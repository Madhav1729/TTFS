// APEX cycle testbench: schedules the FTP inner-product loop over 16 TPPEs, checks every output.
//   apex_tb LAYER_FILE PARAM_FILE OUT_DIR [--saif FILE]
#include "Vapex_top.h"
#include "verilated.h"
#ifdef SAIF_TRACE
#include "verilated_saif_c.h"
#endif
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <sys/stat.h>
#include <vector>

static const int NT = 16, T = T_CFG, BLOCK = 128, MW = MW_CFG, FB = FB_CFG;   // apex15
static_assert(T <= 16, "spike words are held in 16 bits");
static int n_in, m_out, P, K;
static std::vector<int8_t> wt;            // m x n
static std::vector<uint8_t> lev;          // P x n
static std::vector<int32_t> thr;          // m x K
static std::vector<uint8_t> down;
static std::vector<int64_t> theta, mem0;
static std::vector<int> flip, expect_lv;  // expect_lv: P x m

static Vapex_top *top;
#ifdef SAIF_TRACE
static VerilatedSaifC *tfp;
#endif
static uint64_t now;

struct Stats {
    uint64_t cycles = 0, matches = 0, corrections = 0, da_row_reads = 0, idle_rows = 0;   // apex15: one INT4 group
    uint64_t b_mask_beats = 0, b_data_beats = 0, param_beats = 0;
    uint64_t a_mask_beats = 0, a_data_beats = 0, chunks = 0, chunk_tppe_matches = 0;
    uint64_t gc_read_beats = 0, gc_write_beats = 0, outputs = 0, mismatches = 0;
    uint64_t silent_chunks_skipped_data = 0, go_wait_b = 0, go_wait_a = 0, go_wait_ready = 0;
    uint64_t skipped_chunks = 0, lane_busy_max = 0, fiber_beats_checked = 0, fiber_errors = 0;
} st;
static std::vector<std::vector<uint64_t>> bank_log;     // beats written per bank in the current group

static void tick() {
    top->clk = 0; top->eval();
#ifdef SAIF_TRACE
    if (tfp) tfp->dump(now * 10000);              // 1 ps units, 10 ns per cycle (rescaled for DC)
#endif
    top->clk = 1; top->eval();
#ifdef SAIF_TRACE
    if (tfp) tfp->dump(now * 10000 + 5000);
#endif
    ++now;
}

static void load(const char *lpath, const char *ppath) {
    FILE *f = fopen(lpath, "r"); assert(f);
    assert(fscanf(f, "%d %d %d %d", &n_in, &m_out, &P, &K) == 4 && K == T && m_out <= 1023);   // apex15: any P
    wt.resize((size_t)m_out * n_in); thr.resize((size_t)m_out * K); down.resize(m_out); lev.resize((size_t)P * n_in);
    for (auto &x : wt) { int v; assert(fscanf(f, "%d", &v) == 1); assert(v >= -8 && v <= 7); x = (int8_t)v; }
    for (int j = 0; j < m_out; ++j) {
        for (int t = 0; t < K; ++t) assert(fscanf(f, "%d", &thr[(size_t)j * K + t]) == 1);
        int d; assert(fscanf(f, "%d", &d) == 1); down[j] = (uint8_t)d;
    }
    for (auto &x : lev) { int v; assert(fscanf(f, "%d", &v) == 1); assert(v >= 0 && v <= T); x = (uint8_t)v; }
    fclose(f);
    f = fopen(ppath, "r"); assert(f);
    int mm, fb; assert(fscanf(f, "%d %d", &mm, &fb) == 2 && mm == m_out && fb == FB);
    theta.resize(m_out); mem0.resize(m_out); flip.resize(m_out);
    for (int j = 0; j < m_out; ++j) {
        long long a, b; int d; assert(fscanf(f, "%lld %lld %d", &a, &b, &d) == 3);
        theta[j] = a; mem0[j] = b; flip[j] = d; assert(d == down[j]);
        assert(a > 0 && a < (1 << (MW - 1)) && b >= -(1 << (MW - 1)) && b < (1 << (MW - 1)));
    }
    fclose(f);
    expect_lv.resize((size_t)P * m_out);
    for (int p = 0; p < P; ++p) for (int j = 0; j < m_out; ++j) {
        int64_t v = 0;
        for (int i = 0; i < n_in; ++i) v += (int64_t)wt[(size_t)j * n_in + i] * lev[(size_t)p * n_in + i];
        int t = 0; while (t < K && (down[j] ? v > thr[(size_t)j * K + t] : v < thr[(size_t)j * K + t])) ++t;
        int lv = t < K ? K - t : 0;
        // same level from the PASC parameters (floor((d V 2^F + mem0) / theta), clipped)
        int64_t num = (flip[j] ? -v : v) * (1 << fb) + mem0[j];
        int64_t q = num >= 0 ? num / theta[j] : -((-num + theta[j] - 1) / theta[j]);
        int lp = q < 0 ? 0 : (q > T ? T : (int)q);
        assert(lp == lv);
        expect_lv[(size_t)p * m_out + j] = lv;
    }
}

// one chunk of one row / of one filter
struct AChunk { uint64_t bm[2]; std::vector<uint16_t> words; };
struct BChunk { uint64_t bm[2]; std::vector<int> vals; };
static AChunk a_chunk(int p, int g) {
    AChunk c{}; c.bm[0] = c.bm[1] = 0;
    for (int b = 0; b < BLOCK && g * BLOCK + b < n_in; ++b) {
        int v = lev[(size_t)p * n_in + g * BLOCK + b];
        if (v) { c.bm[b / 64] |= 1ull << (b % 64); c.words.push_back((uint16_t)((1u << v) - 1)); }   // front-loaded spike train (PASCAL stage 3)
    }
    return c;
}
static BChunk b_chunk(int j, int g) {
    BChunk c{}; c.bm[0] = c.bm[1] = 0;
    for (int b = 0; b < BLOCK && g * BLOCK + b < n_in; ++b) {
        int w = wt[(size_t)j * n_in + g * BLOCK + b];
        if (w) { c.bm[b / 64] |= 1ull << (b % 64); c.vals.push_back(w); }
    }
    return c;
}
static uint64_t pack_b(const BChunk &c, int beat, int wb) {
    int per = 64 / wb; uint64_t v = 0;
    for (int k = 0; k < per; ++k) {
        size_t i = (size_t)beat * per + k;
        if (i < c.vals.size()) v |= ((uint64_t)(c.vals[i] & ((1 << wb) - 1))) << (k * wb);
    }
    return v;
}
// apex15: T-bit words packed back to back; beat = stream bits [64 beat, 64 beat + 64)
static uint64_t pack_words(const std::vector<uint16_t> &w, size_t first, size_t count, int beat) {
    uint64_t v = 0;
    for (size_t i = 0; i < count; ++i) {
        long lo = (long)i * T - 64L * beat;                     // word position relative to the beat
        if (lo + T <= 0 || lo >= 64) continue;
        uint64_t x = w[first + i];
        v |= lo >= 0 ? x << lo : x >> (-lo);
    }
    return v;
}
static uint64_t pack_a(const AChunk &c, int beat) { return pack_words(c.words, 0, c.words.size(), beat); }
static int ceil_div(int a, int b) { return (a + b - 1) / b; }

// apex15: data-A modelled for energy as two banks (even / odd) of 32-bit rows; a 64-bit beat writes
// one row in each, a T-bit word read touches one row, or two adjacent rows when it crosses a 32-bit row
static int rows_for_word(int idx) { int o = (idx * T) % 32; return o + T > 32 ? 2 : 1; }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    assert(argc >= 4);
    const char *saif = nullptr;
    for (int i = 4; i + 1 < argc; ++i) if (!strcmp(argv[i], "--saif")) saif = argv[i + 1];
    load(argv[1], argv[2]);
    top = new Vapex_top;
#ifdef SAIF_TRACE
    tfp = nullptr;
    if (saif) { Verilated::traceEverOn(true); tfp = new VerilatedSaifC; top->trace(tfp, 99); tfp->open(saif); }
#else
    (void)saif;
#endif
    // idle inputs
    auto idle = [&]() {
        top->b_we = 0; top->b_addr = 0;
        top->p_we = 0; top->go = 0; top->go_first = 0; top->go_last = 0; top->row_reset = 0;
        for (int r = 0; r < NT; ++r) { top->a_we_in[r] = 0; top->a_done_in[r] = 0; }
    };
    idle(); top->rst = 1; top->m_cfg = m_out;
    for (int r = 0; r < NT; ++r) { top->a_sel[r] = r; top->w_src[r] = r; top->bank_rdata[r] = 0; }
    for (int k = 0; k < 4; ++k) tick();
    top->rst = 0; tick();

    const int G = ceil_div(n_in, BLOCK), NG = ceil_div(P, NT);
    // skip weight chunks with no non-zero weight, except a channel's first / last (they frame the output)
    std::vector<int> clist; std::vector<char> zero_b((size_t)m_out * G, 0);
    for (int j = 0; j < m_out; ++j) for (int g = 0; g < G; ++g) {
        zero_b[(size_t)j * G + g] = b_chunk(j, g).vals.empty();
        if (!zero_b[(size_t)j * G + g] || g == 0 || g == G - 1) clist.push_back(j * G + g);
    }
    const int NC = (int)clist.size(), skipped = m_out * G - NC;
    std::vector<int> got(NT, 0);
    for (int grp = 0; grp < NG; ++grp) {
        const int active = P - grp * NT < NT ? P - grp * NT : NT;     // apex15: rows >= active are idle
        st.idle_rows += NT - active;
        // compressor rows start a new set of output fibers; static row -> bank mapping for the group
        idle(); top->row_reset = 1;
        for (int r = 0; r < NT; ++r) { top->a_sel[r] = (r + grp) % NT; top->w_src[(r + grp) % NT] = r; }
        tick(); idle();
        for (int r = 0; r < NT; ++r) got[r] = 0;
        bank_log.assign(NT, {});
        st.gc_read_beats += 3ull * skipped; st.skipped_chunks += skipped;
        // per-TPPE A-port state
        struct APort { int bm_next = 0; uint64_t bm_ready_at = 0; int bm_beat = 0; bool zeroed = false;
                       std::deque<std::pair<int,int>> data;   // (chunk, ctx) waiting for data-A
                       int data_beat = 0; AChunk cur_bm, cur_data; int cur_data_chunk = -1; };
        std::vector<APort> ap(NT);
        int c_go = 0;                      // next chunk to start
        int b_beat = 0, b_need = 0; bool b_started = false; uint64_t b_ready_at = 0;
        BChunk bc{}; bool p_done = false;
        std::vector<AChunk> next_a(NT);
        auto log_writes = [&]() {
            for (int b = 0; b < NT; ++b) if (top->bank_we[b]) {
                int r = (b - grp % NT + NT) % NT;                      // row writing bank b this group
                if (r >= active) continue;                             // idle row: not a real output
                ++st.gc_write_beats; bank_log[b].push_back(top->bank_wdata[b]);
            }
        };
        while (true) {
            // ---- outputs of the previous edge -------------------------------------------------
            for (int r = 0; r < NT; ++r) if ((top->out_valid >> r) & 1) {
                int p = grp * NT + r, j = got[r]++;
                assert(j < m_out);
                if (r >= active) continue;
                int lv = expect_lv[(size_t)p * m_out + j];
                uint16_t want = (uint16_t)((1u << lv) - 1);
                if (top->out_spikes[r] != want) {
                    if (st.mismatches < 10) fprintf(stderr, "MISMATCH grp %d row %d ch %d: got %04x want %04x\n", grp, r, j, (unsigned)top->out_spikes[r], (unsigned)want);
                    ++st.mismatches;
                }
                ++st.outputs;
            }
            log_writes();
            for (int r = 0; r < NT; ++r) {
                if ((top->st_match >> r) & 1) ++st.matches;
                if ((top->st_corr >> r) & 1) ++st.corrections;
            }
            bool all_out = true; for (int r = 0; r < NT; ++r) all_out &= got[r] == m_out;
            if (c_go == NC && all_out) break;
            idle();
            // ---- B bus ---------------------------------------------------------------------------
            if (c_go < NC) {
                int n = clist[c_go] / G, g = clist[c_go] % G;
                bool all_free = top->b_free == 0xFFFF;
                if (!b_started && all_free) {
                    bc = b_chunk(n, g); b_started = true; b_beat = 0; p_done = false;
                    b_need = 2 + ceil_div((int)bc.vals.size(), 16);
                    st.gc_read_beats += 3 + ceil_div((int)bc.vals.size() * 4, 64);       // fiber-B: bitmask, pointer, INT4 values
                }
                if (b_started && b_beat < b_need) {
                    top->b_we = 1; top->b_addr = (uint8_t)b_beat;
                    top->b_data = b_beat < 2 ? bc.bm[b_beat] : pack_b(bc, b_beat - 2, 4);
                    if (b_beat < 2) ++st.b_mask_beats; else ++st.b_data_beats;
                    if (g == 0 && !p_done) {
                        top->p_we = 1; top->p_theta = (uint32_t)theta[n];
                        top->p_mem0 = (uint32_t)(mem0[n] & ((1 << MW) - 1)); top->p_flip = flip[n];
                        p_done = true; ++st.param_beats; ++st.gc_read_beats;
                    }
                    ++b_beat;
                    if (b_beat == b_need) b_ready_at = now + 1;
                }
            }
            // ---- A ports -------------------------------------------------------------------------
            for (int r = 0; r < NT; ++r) {
                APort &a = ap[r];
                int p = grp * NT + r;
                int bank = (r + grp) % NT;
                bool idle_row = r >= active;
                if (idle_row && a.bm_next < NC && a.bm_next <= c_go && a.bm_beat < 2 && !a.zeroed) {
                    // apex15: idle TPPE, bm-A cleared once per group (2 port beats, no cache read)
                    top->a_we_in[r] = 1; top->a_addr_in[r] = (uint8_t)a.bm_beat; top->a_ctx_in[r] = 0;
                    top->bank_rdata[bank] = 0;
                    if (++a.bm_beat == 2) { a.bm_ready_at = now + 2; next_a[r] = AChunk{}; next_a[r].bm[0] = next_a[r].bm[1] = 0; a.zeroed = true; }
                } else if (a.bm_next < NC && a.bm_next <= c_go && a.bm_beat == 0 && (zero_b[clist[a.bm_next]] || idle_row)) {
                    a.bm_beat = 2; a.bm_ready_at = now; next_a[r] = AChunk{};          // nothing to fetch
                    next_a[r].bm[0] = next_a[r].bm[1] = 0;
                } else if (a.bm_next < NC && a.bm_next <= c_go && a.bm_beat < 2) {
                    if (a.bm_beat == 0) { a.cur_bm = a_chunk(p, clist[a.bm_next] % G); st.gc_read_beats += 3; }
                    top->a_we_in[r] = 1; top->a_addr_in[r] = (uint8_t)a.bm_beat; top->a_ctx_in[r] = 0;
                    top->bank_rdata[bank] = a.cur_bm.bm[a.bm_beat]; ++st.a_mask_beats;
                    if (++a.bm_beat == 2) { a.bm_ready_at = now + 2; next_a[r] = a.cur_bm; }
                } else if (!a.data.empty()) {
                    auto [ch, ctx] = a.data.front();
                    if (a.cur_data_chunk != ch) { a.cur_data = a_chunk(p, clist[ch] % G); a.cur_data_chunk = ch; a.data_beat = 0;
                                                  st.gc_read_beats += ceil_div((int)a.cur_data.words.size() * T, 64); }
                    int need = ceil_div((int)a.cur_data.words.size() * T, 64);           // apex15: packed T-bit words
                    top->a_we_in[r] = 1; top->a_addr_in[r] = (uint8_t)(2 + a.data_beat); top->a_ctx_in[r] = (uint8_t)ctx;
                    top->bank_rdata[bank] = pack_a(a.cur_data, a.data_beat); ++st.a_data_beats;
                    if (++a.data_beat == need) { top->a_done_in[r] = 1; a.data.pop_front(); }
                }
            }
            // ---- go ------------------------------------------------------------------------------
            if (c_go < NC && b_started && b_beat == b_need && now >= b_ready_at) {
                bool a_ok = true;
                for (int r = 0; r < NT; ++r) a_ok &= ap[r].bm_next == c_go && ap[r].bm_beat == 2 && now >= ap[r].bm_ready_at;
                if (!a_ok) ++st.go_wait_a;
                else if (top->ready != 0xFFFF) ++st.go_wait_ready;
                else {
                    int g = clist[c_go] % G;
                    top->go = 1; top->go_first = g == 0; top->go_last = g == G - 1;
                    ++st.chunks;
                    int mx = 0;
                    for (int r = 0; r < NT; ++r) {
                        const AChunk &ac = next_a[r];
                        bool hm = (ac.bm[0] & bc.bm[0]) || (ac.bm[1] & bc.bm[1]);
                        assert(hm == (bool)((top->has_match >> r) & 1));
                        int mt = __builtin_popcountll(ac.bm[0] & bc.bm[0]) + __builtin_popcountll(ac.bm[1] & bc.bm[1]);
                        st.chunk_tppe_matches += mt; if (mt > mx) mx = mt;
                        // apex15: data-A rows read by this chunk's corrections (one per match)
                        for (int b = 0; b < BLOCK; ++b) {
                            uint64_t bit = 1ull << (b % 64);
                            if (!((ac.bm[b / 64] & bc.bm[b / 64]) & bit)) continue;
                            int idx = __builtin_popcountll(ac.bm[0] & (b < 64 ? bit - 1 : ~0ull))
                                      + (b >= 64 ? __builtin_popcountll(ac.bm[1] & (bit - 1)) : 0);
                            st.da_row_reads += rows_for_word(idx);
                        }
                        if (hm) ap[r].data.push_back({c_go, (top->next_ctx >> r) & 1});
                        else ++st.silent_chunks_skipped_data;
                        ap[r].bm_next = c_go + 1; ap[r].bm_beat = 0;
                    }
                    st.lane_busy_max += mx;
                    ++c_go; b_started = false;
                }
            } else if (c_go < NC) ++st.go_wait_b;
            tick();
            if (now > 400000000ull) { fprintf(stderr, "timeout\n"); return 2; }
        }
        // let the compressor drain
        idle();
        for (int k = 0; k < 24; ++k) { tick(); log_writes(); }
        // output fibers: per 128 channels, the non-silent spike words packed back to back (apex15),
        // then the bitmask (2 beats) and the pointer (beat address of the fiber's data)
        for (int r = 0; r < active; ++r) {
            int p = grp * NT + r, b = (r + grp) % NT;
            std::vector<uint64_t> want;
            for (int c0 = 0; c0 < m_out; c0 += BLOCK) {
                uint64_t mk[2] = {0, 0}; size_t start = want.size();
                std::vector<uint16_t> words;
                for (int j = c0; j < c0 + BLOCK && j < m_out; ++j) {
                    int lv = expect_lv[(size_t)p * m_out + j];
                    if (!lv) continue;
                    mk[(j - c0) / 64] |= 1ull << ((j - c0) % 64);
                    words.push_back((uint16_t)((1u << lv) - 1));
                }
                int nb = ceil_div((int)words.size() * T, 64);
                for (int k = 0; k < nb; ++k) want.push_back(pack_words(words, 0, words.size(), k));
                want.push_back(mk[0]); want.push_back(mk[1]); want.push_back(start);
            }
            st.fiber_beats_checked += want.size();
            if (want != bank_log[b]) {
                if (st.fiber_errors < 3) {
                    fprintf(stderr, "FIBER grp %d row %d: %zu beats written, %zu expected\n", grp, r, bank_log[b].size(), want.size());
                    for (size_t k = 0; k < want.size() || k < bank_log[b].size(); ++k)
                        fprintf(stderr, "   %2zu got %016" PRIx64 " want %016" PRIx64 "\n", k, k < bank_log[b].size() ? bank_log[b][k] : 0, k < want.size() ? want[k] : 0);
                }
                ++st.fiber_errors;
            }
        }
    }
    st.cycles = now;
#ifdef SAIF_TRACE
    if (tfp) { tfp->close(); delete tfp; }
#endif
    top->final();
    mkdir(argv[3], 0755);
    std::string out = std::string(argv[3]) + "/results.json";
    FILE *o = fopen(out.c_str(), "w"); assert(o);
    fprintf(o, "{\n \"P\": %d, \"n\": %d, \"m\": %d, \"T\": %d, \"tppes\": %d, \"idle_rows\": %" PRIu64 ",\n", P, n_in, m_out, T, NT, st.idle_rows);
    fprintf(o, " \"cycles\": %" PRIu64 ", \"chunks\": %" PRIu64 ", \"skipped_chunks\": %" PRIu64 ", \"outputs\": %" PRIu64 ", \"mismatches\": %" PRIu64 ",\n", st.cycles, st.chunks, st.skipped_chunks, st.outputs, st.mismatches);
    fprintf(o, " \"matches\": %" PRIu64 ", \"corrections\": %" PRIu64 ", \"da_row_reads\": %" PRIu64 ",\n", st.matches, st.corrections, st.da_row_reads);
    fprintf(o, " \"b_mask_beats\": %" PRIu64 ", \"b_data_beats\": %" PRIu64 ", \"param_beats\": %" PRIu64 ",\n", st.b_mask_beats, st.b_data_beats, st.param_beats);
    fprintf(o, " \"a_mask_beats\": %" PRIu64 ", \"a_data_beats\": %" PRIu64 ", \"gc_read_beats\": %" PRIu64 ", \"gc_write_beats\": %" PRIu64 ",\n",
            st.a_mask_beats, st.a_data_beats, st.gc_read_beats, st.gc_write_beats);
    fprintf(o, " \"chunk_tppe_matches\": %" PRIu64 ", \"sum_chunk_max_matches\": %" PRIu64 ", \"skipped_data_a\": %" PRIu64 ",\n",
            st.chunk_tppe_matches, st.lane_busy_max, st.silent_chunks_skipped_data);
    fprintf(o, " \"go_wait_b\": %" PRIu64 ", \"go_wait_a\": %" PRIu64 ", \"go_wait_ready\": %" PRIu64 ",\n \"fiber_beats_checked\": %" PRIu64 ", \"fiber_errors\": %" PRIu64 "\n}\n", st.go_wait_b, st.go_wait_a, st.go_wait_ready, st.fiber_beats_checked, st.fiber_errors);
    fclose(o);
    printf("layer %s: P %d n %d m %d  cycles %" PRIu64 "  outputs %" PRIu64 "  mismatches %" PRIu64 "  fiber errors %" PRIu64 "  matches %" PRIu64 " (%.1f%% of 16 x cycles)\n",
           argv[1], P, n_in, m_out, st.cycles, st.outputs, st.mismatches, st.fiber_errors, st.matches,
           100.0 * st.matches / (16.0 * st.cycles));
    delete top;
    return (st.mismatches || st.fiber_errors) ? 1 : 0;
}
