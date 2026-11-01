#define _POSIX_C_SOURCE 200809L
/* Cycle-level driver for the SparTen and TTFS RTL on one conv layer: memory ports, scheduling,
 * statistics, and an exact check of every output against an integer reference. */
#include "bridge.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int n, m, P, K, latency = 1, engines = 3, CIDX;
static int reuse, g_h, g_w, g_c; /* --geometry */
static int zskip;                /* --zskip 1: all-zero TTFS weight columns not written */
static int wenc;                 /* --wenc list: per column / unit chunk the cheaper of
                                    mask format and (index, value) list format, 1 flag bit */
#define LG 8                     /* lanes per TTFS weight-bank group */
static int8_t *wt;
static int32_t *thr;
static uint8_t *down, *lev;
static int32_t *expected;
static int *first_expected;
static int32_t *got_sum[2];
static int *got_level[2]; /* level = K - tick, 0 = silent */
#define W_(j, i) wt[(size_t)(j) * n + (i)]
#define L_(p, i) lev[(size_t)(p) * n + (i)]
static FILE *trace_file;
static const char *names[2] = {"sparten", "ttfs"};

typedef struct {
    uint64_t cycles, compute_cycles, control_cycles, stall_cycles;
    uint64_t weight_reads, arithmetic, w_beats, a_beats, table_beats, output_beats;
    /* SparTen buffer / network events */
    uint64_t filter_mask_writes, filter_value_writes, filter_mask_reads, input_writes, jobs, net_moves,
        sum_rmw, sum_first_writes, collector_pixels, threshold_writes, output_values;
    /* TTFS buffer events */
    uint64_t column_writes, column_reads, membrane_writes, membrane_reads, event_pushes, passes,
        compare_pixels, output_spikes, events_read, event_read_beats, events_dropped, issue_cycles;
    /* input reuse (patch fetches on the A port, reads of the local patch buffer) */
    uint64_t patch_beats, patch_entries, patch_nnz, patch_max_bits, local_beats, local_read_bits;
    /* TTFS weight banks split into LANES/LG lane groups; a group is read / written only if non-zero */
    uint64_t column_group_reads, column_group_writes, column_writes_skipped;
} Stats;
static Stats stats[2];
static int pop64(uint64_t v) {
    return __builtin_popcountll(v);
}
static int beats(long bits) {
    return (int)((bits + 63) / 64);
}
static int grp_size(int grp) {
    return P - grp * OB < OB ? P - grp * OB : OB;
}
static int expected_level(int p, int j) {
    int t = first_expected[(size_t)p * m + j];
    return t < K ? K - t : 0;
}

static void load(const char *path) {
    FILE *f = fopen(path, "r");
    assert(f);
    assert(fscanf(f, "%d %d %d %d", &n, &m, &P, &K) == 4);
    assert(n > 0 && m > 0 && P > 0 && K == STEPS);
    wt = malloc((size_t)m * n);
    thr = malloc(sizeof(int32_t) * (size_t)m * K);
    down = malloc((size_t)m);
    lev = malloc((size_t)P * n);
    assert(wt && thr && down && lev);
    for (size_t x = 0; x < (size_t)m * n; ++x) {
        int v;
        assert(fscanf(f, "%d", &v) == 1);
        wt[x] = (int8_t)v;
    }
    for (int j = 0; j < m; ++j) {
        for (int t = 0; t < K; ++t) assert(fscanf(f, "%d", &thr[(size_t)j * K + t]) == 1);
        int d;
        assert(fscanf(f, "%d", &d) == 1 && (d == 0 || d == 1));
        down[j] = (uint8_t)d;
    }
    for (size_t x = 0; x < (size_t)P * n; ++x) {
        int v;
        assert(fscanf(f, "%d", &v) == 1);
        assert(v >= 0 && v <= 255 && (v <= K || !(engines & 2)));
        lev[x] = (uint8_t)v;
    }
    fclose(f);
    expected = calloc((size_t)P * m, 4);
    first_expected = malloc(sizeof(int) * (size_t)P * m);
    for (int e = 0; e < 2; ++e) {
        got_sum[e] = calloc((size_t)P * m, 4);
        got_level[e] = malloc(sizeof(int) * (size_t)P * m);
        for (size_t x = 0; x < (size_t)P * m; ++x) got_level[e][x] = -1;
    }
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < m; ++j) {
            int64_t v = 0;
            for (int i = 0; i < n; ++i) v += (int64_t)W_(j, i) * L_(p, i);
            assert(v > INT32_MIN && v < INT32_MAX);
            expected[(size_t)p * m + j] = (int32_t)v;
            int t = 0;
            while (t < K && (down[j] ? v > thr[(size_t)j * K + t] : v < thr[(size_t)j * K + t])) ++t;
            first_expected[(size_t)p * m + j] = t;
        }
}

/* ---- pipelined memory ports ------------------------------------------------------------- */
enum { ACT_NONE, SP_MASK, SP_ROW, SP_INPUT, SP_TABLE, TT_COLS, TT_EVENTS, TT_TABLE };
enum { GATE_NONE, GATE_FILTER, GATE_INPUT, GATE_STABLE, GATE_WCHUNK, GATE_EVENTS, GATE_TTABLE, GATE_PATCH };
enum { PATCH = 100 }; /* patch beat (act) */
typedef struct {
    int act, gate, a, b, c, d;
} Beat;
typedef struct {
    Beat *q;
    size_t head, len, cap;
    Beat fly[256];
    long at[256];
    int fh, fn;
} Port;
static Port port[2][3]; /* [2] = local patch-buffer port */
static long now;
static void push_beat(Port *p, Beat b) {
    if (p->len == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 1024;
        p->q = realloc(p->q, p->cap * sizeof(Beat));
        assert(p->q);
    }
    p->q[p->len++] = b;
}
static int port_idle(const Port *p) {
    return p->head == p->len && p->fn == 0;
}
static int ports_idle(int e) {
    return port_idle(&port[e][0]) && port_idle(&port[e][1]) && port_idle(&port[e][2]);
}

/* the input patch of pixels [p0, p0+np) and chunk inputs [lo, hi): distinct in-bounds
 * (channel, position) entries and their non-zeros; returns its bits (SparseMap + values) */
static int *stamp, stamp_id;
static long patch_bits(int e, int p0, int np, int lo, int hi) {
    if (!stamp) {
        stamp = calloc((size_t)g_c * g_h * g_w, sizeof(int));
        assert(stamp);
    }
    ++stamp_id;
    long entries = 0, nnz = 0;
    for (int x = 0; x < np; ++x) {
        int p = p0 + x, y = p / g_w, xx = p % g_w;
        for (int i = lo; i < hi; ++i) {
            int c = i / 9, k = i % 9, yy = y + k / 3 - 1, xc = xx + k % 3 - 1;
            if (yy < 0 || yy >= g_h || xc < 0 || xc >= g_w) {
                assert(!L_(p, i));
                continue;
            }
            int *s = &stamp[((size_t)c * g_h + yy) * g_w + xc];
            if (*s != stamp_id) {
                *s = stamp_id;
                ++entries;
                nnz += L_(p, i) != 0;
            }
        }
    }
    long bits = entries + nnz * SPT_A;
    stats[e].patch_entries += (uint64_t)entries;
    stats[e].patch_nnz += (uint64_t)nnz;
    if ((uint64_t)bits > stats[e].patch_max_bits) stats[e].patch_max_bits = (uint64_t)bits;
    return bits;
}

/* ==== SparTen ================================================================================ */
static int G, NG, s_tiles;
static int (*tile_filter)[NF]; /* tile, local q -> global filter or -1 */
static int (*pair)[UNITS][2];  /* (tile*G + g), unit, slot -> local q or -1 */
typedef struct {
    int k, grp, g, job0, njobs;
} FChunk;
typedef struct {
    int fc, p;
} Job;
static FChunk *fcs;
static Job *jobs;
static int nfc, njob;
static int *fw_left, *ja_left, *emits_upto; /* cumulative expected emits through job j */
static int *sp_left;                        /* patch beats still to arrive, per filter chunk */
static int f_done, j_done, j_next, running, s_tables_done = -1, s_table_left;
static long takes_total;
/* groups: (tile, grp) in order; gi = k*NG + grp; bank = gi % 2 */
static int *grp_last_job, *grp_collected, compute_group_done = -1, collect_next = 0, collecting = -1, settle;
static int thq[1 << 14], thq_head, thq_len;
static int nnz_chunk(int j, int g) {
    int c = 0;
    for (int b = 0; b < BLOCK && g * BLOCK + b < n; ++b) c += W_(j, g * BLOCK + b) != 0;
    return c;
}
static void s_plan_filters(void) {
    int *order = malloc(sizeof(int) * m), *nnz = malloc(sizeof(int) * m);
    for (int j = 0; j < m; ++j) {
        order[j] = j;
        nnz[j] = 0;
        for (int i = 0; i < n; ++i) nnz[j] += W_(j, i) != 0;
    }
    for (int a = 1; a < m; ++a)
        for (int b = a; b > 0 && nnz[order[b]] > nnz[order[b - 1]]; --b) { /* GB-S layer sort */
            int x = order[b];
            order[b] = order[b - 1];
            order[b - 1] = x;
        }
    G = (n + BLOCK - 1) / BLOCK;
    NG = (P + OB - 1) / OB;
    s_tiles = (m + NF - 1) / NF;
    tile_filter = malloc(sizeof(*tile_filter) * (size_t)s_tiles);
    pair = malloc(sizeof(*pair) * (size_t)s_tiles * G);
    for (int k = 0; k < s_tiles; ++k) {
        int cnt = m - k * NF < NF ? m - k * NF : NF;
        for (int q = 0; q < NF; ++q) tile_filter[k][q] = q < cnt ? order[k * NF + q] : -1;
        for (int g = 0; g < G; ++g) { /* GB-H: per-chunk densest with sparsest */
            int ord[NF], cn[NF];
            for (int q = 0; q < cnt; ++q) {
                ord[q] = q;
                cn[q] = nnz_chunk(tile_filter[k][q], g);
            }
            for (int a = 1; a < cnt; ++a)
                for (int b = a; b > 0 && cn[ord[b]] > cn[ord[b - 1]]; --b) {
                    int x = ord[b];
                    ord[b] = ord[b - 1];
                    ord[b - 1] = x;
                }
            for (int u = 0; u < UNITS; ++u) pair[k * G + g][u][0] = pair[k * G + g][u][1] = -1;
            for (int u = 0, lo = 0, hi = cnt - 1; lo <= hi; ++u, ++lo, --hi) {
                pair[k * G + g][u][0] = ord[lo];
                if (hi > lo) pair[k * G + g][u][1] = ord[hi];
            }
        }
    }
    free(order);
    free(nnz);
}
static void s_filter_chunk(int k, int g, int f, block_t *mask, int vals[UNITS][BLOCK], int *cnt, int *rows,
                           lanes_t *en) {
    *rows = 0;
    *en = 0;
    for (int u = 0; u < UNITS; ++u) {
        memset(&mask[u], 0, sizeof(block_t));
        cnt[u] = 0;
        int q = pair[k * G + g][u][f];
        if (q < 0) continue;
        *en |= 1ull << u;
        int j = tile_filter[k][q];
        for (int b = 0; b < BLOCK && g * BLOCK + b < n; ++b) {
            int w = W_(j, g * BLOCK + b);
            if (w) {
                mask[u].w[b / 64] |= 1ull << (b % 64);
                vals[u][cnt[u]++] = w;
            }
        }
        if (cnt[u] > *rows) *rows = cnt[u];
    }
}
static int s_input_chunk(int p, int g, block_t *mask, int *vals) {
    int count = 0;
    memset(mask, 0, sizeof(block_t));
    memset(vals, 0, sizeof(int) * BLOCK);
    for (int b = 0; b < BLOCK && g * BLOCK + b < n; ++b) {
        int v = L_(p, g * BLOCK + b);
        if (v) {
            mask->w[b / 64] |= 1ull << (b % 64);
            vals[count++] = v;
        }
    }
    return count;
}
static int s_row_beats(const int *cnt, int r) {
    int units = 0;
    for (int u = 0; u < UNITS; ++u) units += cnt[u] > r;
    return beats((long)units * SPT_W);
}
static void s_plan(void) {
    s_plan_filters();
    fcs = malloc(sizeof(FChunk) * (size_t)s_tiles * NG * G);
    jobs = malloc(sizeof(Job) * (size_t)s_tiles * P * G);
    int ngroups = s_tiles * NG;
    grp_last_job = malloc(sizeof(int) * (size_t)ngroups);
    grp_collected = calloc((size_t)ngroups, sizeof(int));
    block_t am;
    int av[BLOCK];
    for (int k = 0; k < s_tiles; ++k)
        for (int grp = 0; grp < NG; ++grp) {
            for (int g = 0; g < G; ++g) {
                FChunk c = {k, grp, g, njob, 0};
                for (int x = 0; x < grp_size(grp); ++x) {
                    int p = grp * OB + x;
                    if (g == 0 || s_input_chunk(p, g, &am, av)) {
                        jobs[njob++] = (Job){nfc, p};
                        ++c.njobs;
                    }
                }
                if (c.njobs) fcs[nfc++] = c;
            }
            grp_last_job[k * NG + grp] = njob - 1;
        }
    fw_left = calloc((size_t)nfc, sizeof(int));
    ja_left = calloc((size_t)njob, sizeof(int));
    emits_upto = malloc(sizeof(int) * (size_t)njob);
    sp_left = calloc((size_t)nfc, sizeof(int));
    block_t mask[UNITS];
    static int vals[UNITS][BLOCK];
    int cnt[UNITS], rows;
    lanes_t en;
    int last_tile = -1, cum = 0;
    for (int c = 0; c < nfc; ++c) {
        if (fcs[c].k != last_tile) { /* threshold table of the tile (+1 beat of direction bits) */
            last_tile = fcs[c].k;
            int tb = beats((long)NF * K * 32);
            for (int b = 0; b <= tb; ++b)
                push_beat(&port[SPT][0],
                          (Beat){SP_TABLE, b == 0 ? GATE_STABLE : GATE_NONE, last_tile, b, tb, 0});
        }
        for (int f = 0; f < 2; ++f) {
            s_filter_chunk(fcs[c].k, fcs[c].g, f, mask, vals, cnt, &rows, &en);
            int nu = pop64(en);
            if (!nu) continue;
            long mbits = (long)nu * BLOCK;
            if (wenc) { /* SparseMap or 7-bit position list, per unit */
                mbits = 0;
                for (int u = 0; u < UNITS; ++u)
                    if (en >> u & 1) {
                        long l = 8 + 7L * cnt[u];
                        mbits += 1 + (l < BLOCK ? l : BLOCK);
                    }
            }
            int mb = beats(mbits);
            for (int b = 0; b < mb; ++b)
                push_beat(&port[SPT][0], (Beat){b == mb - 1 ? SP_MASK : ACT_NONE, GATE_FILTER, c, f, 0, 0});
            fw_left[c] += mb;
            for (int r = 0; r < rows; ++r) {
                int rb = s_row_beats(cnt, r);
                for (int b = 0; b < rb; ++b)
                    push_beat(&port[SPT][0],
                              (Beat){b == rb - 1 ? SP_ROW : ACT_NONE, GATE_FILTER, c, f, r, 0});
                fw_left[c] += rb;
            }
        }
        int units_active = 0;
        for (int u = 0; u < UNITS; ++u) units_active += pair[fcs[c].k * G + fcs[c].g][u][0] >= 0;
        if (reuse) { /* the chunk's patch, once for the group's jobs */
            int lo = fcs[c].g * BLOCK, hi = lo + BLOCK < n ? lo + BLOCK : n;
            int pb = beats(patch_bits(SPT, fcs[c].grp * OB, grp_size(fcs[c].grp), lo, hi));
            for (int b = 0; b < pb; ++b) push_beat(&port[SPT][1], (Beat){PATCH, GATE_PATCH, c, 0, 0, 0});
            sp_left[c] = pb;
        }
        for (int j = fcs[c].job0; j < fcs[c].job0 + fcs[c].njobs; ++j) {
            int count = s_input_chunk(jobs[j].p, fcs[c].g, &am, av);
            int ab = beats(BLOCK) + beats((long)count * SPT_A);
            if (reuse) stats[SPT].local_read_bits += (uint64_t)(BLOCK + count * SPT_A);
            for (int b = 0; b < ab; ++b)
                push_beat(&port[SPT][reuse ? 2 : 1],
                          (Beat){b == ab - 1 ? SP_INPUT : ACT_NONE, GATE_INPUT, j, 0, 0, 0});
            ja_left[j] = ab;
            cum += units_active;
            emits_upto[j] = cum;
        }
    }
}
static int s_gate(const Beat *b) {
    if (b->gate == GATE_FILTER) return b->a < f_done + 2;
    if (b->gate == GATE_INPUT) return b->a < j_done + 2 && !sp_left[jobs[b->a].fc];
    if (b->gate == GATE_PATCH) return b->a < f_done + 2;
    if (b->gate == GATE_STABLE)
        return s_tables_done == b->a - 1 && collecting < 0 && (b->a == 0 || grp_collected[b->a * NG - 1]);
    return 1;
}
static void s_arrive(const Beat *b) {
    block_t mask[UNITS];
    static int vals[UNITS][BLOCK];
    int cnt[UNITS], rows, row[UNITS];
    lanes_t en;
    if (b->act == SP_MASK || b->act == SP_ROW) {
        const FChunk *c = &fcs[b->a];
        s_filter_chunk(c->k, c->g, b->b, mask, vals, cnt, &rows, &en);
        if (b->act == SP_MASK) {
            spf_mask_write(en, b->a % 2, b->b, mask);
            stats[SPT].filter_mask_writes += pop64(en);
        } else {
            lanes_t e = 0;
            for (int u = 0; u < UNITS; ++u) {
                row[u] = cnt[u] > b->c ? vals[u][b->c] : 0;
                if (cnt[u] > b->c) e |= 1ull << u;
            }
            spf_value_write(e, b->a % 2, b->b, b->c, row);
            stats[SPT].filter_value_writes += pop64(e);
        }
    } else if (b->act == SP_INPUT) {
        block_t am;
        int av[BLOCK];
        s_input_chunk(jobs[b->a].p, fcs[jobs[b->a].fc].g, &am, av);
        spf_input_write(b->a % 2, &am, av);
        ++stats[SPT].input_writes;
    } else if (b->act == SP_TABLE) { /* each 64-bit beat carries two 32-bit entries */
        int k = b->a;
        if (b->b == 0) s_table_left = NF * K;
        if (b->b < b->c)
            for (int e = 0; e < 2; ++e) {
                int idx = b->b * 2 + e;
                if (idx >= NF * K) break;
                thq[(thq_head + thq_len++) % (1 << 14)] = (k << 16) | idx;
            }
    }
}
static void s_ports_cycle(void) {
    for (int p = 0; p < 3; ++p) {
        Port *Q = &port[SPT][p];
        if (Q->head < Q->len) {
            Beat *b = &Q->q[Q->head];
            if (s_gate(b)) {
                assert(Q->fn < 256);
                int slot = (Q->fh + Q->fn) % 256;
                Q->fly[slot] = *b;
                Q->at[slot] = now + latency - 1;
                ++Q->fn;
                ++Q->head;
                if (p == 0)
                    ++stats[SPT].w_beats;
                else if (p == 1)
                    ++stats[SPT].a_beats;
                else
                    ++stats[SPT].local_beats;
                if (b->act == SP_TABLE) ++stats[SPT].table_beats;
                if (b->act == PATCH) ++stats[SPT].patch_beats;
            }
        }
        while (Q->fn && Q->at[Q->fh] == now) {
            Beat *b = &Q->fly[Q->fh];
            s_arrive(b);
            if (b->gate == GATE_FILTER) --fw_left[b->a];
            if (b->gate == GATE_INPUT) --ja_left[b->a];
            if (b->gate == GATE_PATCH) --sp_left[b->a];
            Q->fh = (Q->fh + 1) % 256;
            --Q->fn;
        }
    }
}
static void tick(int e, const char *phase, lanes_t reads, lanes_t ops) {
    Stats *a = &stats[e];
    a->weight_reads += pop64(reads);
    a->arithmetic += pop64(ops);
    if (!strcmp(phase, "compute"))
        ++a->compute_cycles;
    else if (!strcmp(phase, "control"))
        ++a->control_cycles;
    else
        ++a->stall_cycles;
    if (trace_file) fprintf(trace_file, "%s,%ld,%s\n", names[e], now, phase);
    r5_tick(e);
    ++a->cycles;
    ++now;
}
static void run_s(void) {
    now = 0;
    int ngroups = s_tiles * NG;
    while (collect_next < ngroups || !ports_idle(SPT)) {
        r5_idle(SPT);
        const char *phase = "stall";
        int outgrp = collecting; /* a pixel leaving the collector this cycle belongs here */
        /* threshold table writes, one entry per cycle */
        if (thq_len) {
            int v = thq[thq_head], k = v >> 16, idx = v & 0xffff, q = idx / K, t = idx % K,
                j = tile_filter[k][q];
            spf_threshold(q, t, j >= 0 ? thr[(size_t)j * K + t] : 0, j >= 0 ? down[j] : 0);
            ++stats[SPT].threshold_writes;
            thq_head = (thq_head + 1) % (1 << 14);
            --thq_len;
            if (--s_table_left == 0) s_tables_done = k;
        }
        /* a group is complete when all its partial sums were taken and written */
        if (compute_group_done + 1 < ngroups && j_done > grp_last_job[compute_group_done + 1] &&
            settle == 0 && takes_total >= emits_upto[grp_last_job[compute_group_done + 1]])
            settle = 3;
        if (settle && --settle == 0) ++compute_group_done;
        /* collector: the next complete group, once the tile's table is in */
        if (collecting >= 0 && !spf_col_busy()) {
            grp_collected[collecting] = 1;
            collecting = -1;
            ++collect_next;
        }
        if (collecting < 0 && collect_next <= compute_group_done && s_tables_done >= collect_next / NG) {
            int gi = collect_next, k = gi / NG;
            uint8_t act[NF];
            for (int q = 0; q < NF; ++q) act[q] = tile_filter[k][q] >= 0;
            spf_collect(gi % 2, grp_size(gi % NG), act);
            collecting = gi;
        }
        /* next job */
        if (running)
            phase = "compute";
        else if (j_next < njob && !fw_left[jobs[j_next].fc] && !ja_left[j_next]) {
            const Job *J = &jobs[j_next];
            const FChunk *c = &fcs[J->fc];
            int gi = c->k * NG + c->grp;
            if (gi < 2 || grp_collected[gi - 2]) { /* the bank's previous group has been drained */
                int act2[2 * UNITS], dest2[2 * UNITS], nact = 0;
                for (int u = 0; u < UNITS; ++u)
                    for (int f = 0; f < 2; ++f) {
                        int q = pair[c->k * G + c->g][u][f];
                        act2[2 * u + f] = q >= 0;
                        dest2[2 * u + f] = q >= 0 ? q : 0;
                        nact += q >= 0;
                    }
                spf_start(J->fc % 2, j_next % 2, c->g == 0, gi % 2, J->p - c->grp * OB, act2, dest2);
                running = 1;
                phase = "compute";
                ++stats[SPT].jobs;
                stats[SPT].filter_mask_reads += (uint64_t)nact;
            }
        }
        s_ports_cycle();
        lanes_t reads = spf_reads(), ops = spf_macs();
        long tk = pop64(spf_takes());
        int mv = spf_moves();
        int pix;
        static uint8_t mask[NF], packed[NF];
        static int32_t sums[NF];
        int outv = spf_out(&pix, mask, packed, sums);
        stats[SPT].net_moves += (uint64_t)mv;
        tick(SPT, phase, reads, ops);
        takes_total += tk;
        if (outv) { /* one collected pixel: check sums and levels */
            assert(outgrp >= 0);
            int gi = outgrp, k = gi / NG, p = (gi % NG) * OB + pix, r = 0, nz = 0, act = 0;
            for (int q = 0; q < NF; ++q) {
                int j = tile_filter[k][q];
                if (j < 0) continue;
                ++act;
                got_sum[SPT][(size_t)p * m + j] = sums[q];
                int lv = mask[q] ? packed[r++] : 0;
                nz += mask[q];
                got_level[SPT][(size_t)p * m + j] = lv;
            }
            ++stats[SPT].collector_pixels;
            stats[SPT].output_values += (uint64_t)nz;
            int ob = beats((long)act + (long)nz * SPT_A);
            stats[SPT].output_beats += (uint64_t)ob;
            for (int b = 0; b < ob; ++b) push_beat(&port[SPT][1], (Beat){ACT_NONE, GATE_NONE, 0, 0, 0, 0});
        }
        if (running && !spf_busy()) {
            running = 0;
            int j = j_next++;
            ++j_done;
            if (j + 1 == njob || jobs[j + 1].fc != jobs[j].fc) ++f_done;
        }
        assert(now < 4000000000L);
    }
}

/* ==== TTFS ===================================================================================== */
static int GC, t_tiles;
typedef struct {
    int k, grp, g, pass0, npass, cols, ncols_written;
    long bits;
    int *col_end;
} WChunk;
typedef struct {
    int wc, p, nev, *ev;
} Pass; /* ev: events after the all-zero-column filter */
static WChunk *wcs;
static Pass *passes;
static int nwc, npass;
static int w_done, ps_done, t_tables_done = -1, t_table_left = -1, ev_reserved, t_tiles_compared;
static int *colq, colq_head, colq_len, *evq, evq_head, evq_len;
static int *t_grp_done, *t_grp_compared; /* per group (tile*NG + grp) */
static unsigned char *colnz;             /* tile x input: column has a non-zero weight */
static int *tp_left;                     /* patch beats still to arrive, per weight chunk */
static lanes_t tt_lanes(int k) {
    int c = m - k * LANES < LANES ? m - k * LANES : LANES;
    return c == 64 ? ~0ull : (1ull << c) - 1;
}
static int col_bits(int k, int i) {
    int nz = 0;
    for (int l = 0; l < LANES && k * LANES + l < m; ++l) nz += W_(k * LANES + l, i) != 0;
    if (wenc) {
        int l = 7 + nz * (6 + TT_W);
        return 1 + (l < LANES + nz * TT_W ? l : LANES + nz * TT_W);
    }
    return LANES + nz * TT_W;
}
static int col_groups(int k, int i) { /* non-zero lane groups of a column */
    int g = 0;
    for (int l0 = 0; l0 < LANES; l0 += LG) {
        int any = 0;
        for (int l = l0; l < l0 + LG && k * LANES + l < m; ++l) any |= W_(k * LANES + l, i) != 0;
        g += any;
    }
    return g;
}
static void t_plan(void) {
    GC = (n + CH - 1) / CH;
    NG = (P + OB - 1) / OB;
    t_tiles = (m + LANES - 1) / LANES;
    CIDX = 0;
    while ((1 << CIDX) < CH) ++CIDX;
    wcs = calloc((size_t)t_tiles * NG * GC, sizeof(WChunk));
    passes = calloc((size_t)t_tiles * P * GC, sizeof(Pass));
    t_grp_done = calloc((size_t)t_tiles * NG, sizeof(int));
    t_grp_compared = calloc((size_t)t_tiles * NG, sizeof(int));
    colnz = calloc((size_t)t_tiles * n, 1);
    for (int k = 0; k < t_tiles; ++k)
        for (int i = 0; i < n; ++i)
            for (int l = 0; l < LANES && k * LANES + l < m; ++l)
                if (W_(k * LANES + l, i)) {
                    colnz[(size_t)k * n + i] = 1;
                    break;
                }
    for (int k = 0; k < t_tiles; ++k) {
        int tb = beats((long)pop64(tt_lanes(k)) * 32);
        for (int t = 0; t < K; ++t)
            for (int b = 0; b < tb + (t == 0); ++b)
                push_beat(&port[TTF][0], (Beat){b == tb - (t != 0) ? TT_TABLE : ACT_NONE,
                                                b == 0 && t == 0 ? GATE_TTABLE : GATE_NONE, k, t, 0, 0});
        for (int grp = 0; grp < NG; ++grp)
            for (int g = 0; g < GC; ++g) {
                WChunk c = {k, grp, g, npass, 0, 0, 0, 0, NULL};
                int lo = g * CH, hi = (g + 1) * CH < n ? (g + 1) * CH : n;
                for (int x = 0; x < grp_size(grp); ++x) {
                    int p = grp * OB + x, nall = 0, nev = 0;
                    for (int i = lo; i < hi; ++i)
                        if (L_(p, i)) {
                            ++nall;
                            nev += colnz[(size_t)k * n + i];
                        }
                    /* the event source reads every spike of the pass and drops those whose weight column
                       is all-zero for this tile (one bit per tile x input) */
                    stats[TTF].events_read += (uint64_t)nall;
                    stats[TTF].events_dropped += (uint64_t)(nall - nev);
                    stats[TTF].event_read_beats +=
                        (uint64_t)((nall + (64 / (CIDX + TBITS)) - 1) / (64 / (CIDX + TBITS))); /* TBITS */
                    if (reuse) stats[TTF].local_read_bits += (uint64_t)((hi - lo) + nall * SPT_A);
                    if (g > 0 && !nev) continue;
                    Pass q = {nwc, p, nev, malloc(sizeof(int) * (size_t)(nev ? nev : 1))};
                    int e = 0;
                    for (int t = 0; t < K; ++t)
                        for (int i = lo; i < hi; ++i)
                            if (L_(p, i) == K - t && colnz[(size_t)k * n + i]) {
                                q.ev[e++] = (i - lo) | (t << 16);
                                stats[TTF].column_group_reads += (uint64_t)col_groups(k, i);
                            }
                    passes[npass++] = q;
                    ++c.npass;
                }
                if (!c.npass) continue;
                c.cols = hi - lo;
                c.col_end = malloc(sizeof(int) * (size_t)c.cols);
                for (int i = lo; i < hi; ++i) {
                    c.bits += col_bits(k, i);
                    c.col_end[i - lo] = (int)c.bits;
                    stats[TTF].column_group_writes += (uint64_t)col_groups(k, i);
                }
                int wb = beats(c.bits);
                for (int b = 0; b < wb; ++b)
                    push_beat(&port[TTF][0], (Beat){TT_COLS, b == 0 ? GATE_WCHUNK : GATE_NONE, nwc, b, 0, 0});
                if (reuse) { /* the chunk's patch, once for the group's passes */
                    int pb = beats(patch_bits(TTF, grp * OB, grp_size(grp), lo, hi));
                    for (int b = 0; b < pb; ++b)
                        push_beat(&port[TTF][1], (Beat){PATCH, GATE_PATCH, nwc, 0, 0, 0});
                    tp_left = realloc(tp_left, sizeof(int) * (size_t)(nwc + 1));
                    tp_left[nwc] = pb;
                }
                wcs[nwc++] = c;
            }
    }
    for (int s = 0; s < npass; ++s) {
        int bits_ev = CIDX + TBITS, per = 64 / bits_ev, nb = (passes[s].nev + per - 1) / per; /* TBITS */
        for (int b = 0; b < nb; ++b) {
            int cnt = passes[s].nev - b * per < per ? passes[s].nev - b * per : per;
            push_beat(&port[TTF][reuse ? 2 : 1], (Beat){TT_EVENTS, GATE_EVENTS, s, b * per, cnt, 0});
        }
    }
    colq = malloc(sizeof(int) * (size_t)(CH * 2 + 8));
    evq = malloc(sizeof(int) * (size_t)(EDEPTH + 64));
}
static int t_gate(const Beat *b) {
    if (b->gate == GATE_TTABLE) return b->a == t_tiles_compared && t_table_left <= 0;
    if (b->gate == GATE_WCHUNK) return b->a < w_done + 2;
    if (b->gate == GATE_EVENTS)
        return b->a < ps_done + 2 && ev_reserved + b->c <= EDEPTH + WS &&
               !(reuse && tp_left[passes[b->a].wc]);
    if (b->gate == GATE_PATCH) return b->a < w_done + 2;
    return 1;
}
static void t_arrive(const Beat *b) {
    if (b->act == TT_TABLE) {
        int k = b->a, t = b->b;
        int32_t th[LANES] = {0};
        lanes_t d = 0;
        for (int l = 0; l < LANES && k * LANES + l < m; ++l) {
            th[l] = thr[(size_t)(k * LANES + l) * K + t];
            d |= (lanes_t)down[k * LANES + l] << l;
        }
        tt_threshold(t, th, d);
        ++stats[TTF].threshold_writes;
        if (--t_table_left == 0) t_tables_done = k;
    } else if (b->act == TT_COLS) {
        const WChunk *c = &wcs[b->a];
        long lo = (long)b->b * 64, hi = lo + 64;
        for (int x = 0; x < c->cols; ++x)
            if (c->col_end[x] > lo && c->col_end[x] <= hi)
                colq[(colq_head + colq_len++) % (CH * 2 + 8)] = (b->a << 16) | x;
    } else if (b->act == TT_EVENTS) {
        const Pass *q = &passes[b->a];
        for (int e = 0; e < b->c; ++e) evq[(evq_head + evq_len++) % (EDEPTH + 64)] = q->ev[b->b + e];
    }
}
static void t_ports_cycle(void) {
    for (int p = 0; p < 3; ++p) {
        Port *Q = &port[TTF][p];
        if (Q->head < Q->len) {
            Beat *b = &Q->q[Q->head];
            if (t_gate(b)) {
                assert(Q->fn < 256);
                int slot = (Q->fh + Q->fn) % 256;
                Q->fly[slot] = *b;
                Q->at[slot] = now + latency - 1;
                ++Q->fn;
                ++Q->head;
                if (p == 0)
                    ++stats[TTF].w_beats;
                else if (p == 1)
                    ++stats[TTF].a_beats;
                else
                    ++stats[TTF].local_beats;
                if (b->act == PATCH) {
                    ++stats[TTF].patch_beats;
                    --tp_left[b->a];
                }
                if (b->gate == GATE_EVENTS) ev_reserved += b->c;
                if (b->gate == GATE_TTABLE) t_table_left = K;
                if (p == 0 && b->act != TT_COLS) ++stats[TTF].table_beats;
            }
        }
        while (Q->fn && Q->at[Q->fh] == now) {
            t_arrive(&Q->fly[Q->fh]);
            Q->fh = (Q->fh + 1) % 256;
            --Q->fn;
        }
    }
}
static void run_t(void) {
    now = 0;
    int ngroups = t_tiles * NG;
    enum { PASS, END, DRAIN, DONE } state = PASS;
    int ps = 0, k = -1, grp = -1;
    /* output window (background): one group at a time, per pixel 1 load + K compares */
    int cg = -1, cx = 0, ct = -1, ccount = 0, cmp_next = 0;
    tt_active(tt_lanes(0));
    while (state != DONE || cmp_next < ngroups || cg >= 0 || !ports_idle(TTF) || colq_len) {
        r5_idle(TTF);
        const char *phase = "stall";
        int compared = -1, loaded = -1;
        while (zskip && colq_len) { /* all-zero columns are never read (their spikes are
                                       dropped at the source), so they are not written */
            int v = colq[colq_head];
            WChunk *c = &wcs[v >> 16];
            int i = c->g * CH + (v & 0xffff);
            if (colnz[(size_t)c->k * n + i]) break;
            ++c->ncols_written;
            ++stats[TTF].column_writes_skipped;
            colq_head = (colq_head + 1) % (CH * 2 + 8);
            --colq_len;
        }
        if (colq_len) {
            int v = colq[colq_head];
            WChunk *c = &wcs[v >> 16];
            int x = v & 0xffff, i = c->g * CH + x;
            lanes_t mask = 0;
            int packed[LANES] = {0}, np = 0;
            for (int l = 0; l < LANES && c->k * LANES + l < m; ++l) {
                int w = W_(c->k * LANES + l, i);
                if (w) {
                    mask |= 1ull << l;
                    packed[np++] = w;
                }
            }
            tt_wb_write((v >> 16) % 2, x, mask, packed);
            ++c->ncols_written;
            ++stats[TTF].column_writes;
            colq_head = (colq_head + 1) % (CH * 2 + 8);
            --colq_len;
        }
        { /* up to PUSHW events per cycle into the tile's FIFO */
            int space = EDEPTH - tt_ev_count(), np = 0, idx[PUSHW], tk[PUSHW];
            while (np < PUSHW && np < space && evq_len) {
                int v = evq[evq_head];
                idx[np] = v & 0xffff;
                tk[np] = v >> 16;
                ++np;
                evq_head = (evq_head + 1) % (EDEPTH + 64);
                --evq_len;
            }
            if (np) {
                tt_ev_push(np, idx, tk);
                stats[TTF].event_pushes += (uint64_t)np;
            }
        }
        /* pass state machine */
        if (state == PASS) {
            if (ps >= npass)
                state = DRAIN;
            else {
                const Pass *q = &passes[ps];
                const WChunk *c = &wcs[q->wc];
                int gi = c->k * NG + c->grp;
                if (grp >= 0 && (c->k != k || c->grp != grp))
                    state = DRAIN; /* group finished */
                else if (c->ncols_written == c->cols && (gi < 2 || t_grp_compared[gi - 2])) {
                    k = c->k;
                    grp = c->grp;
                    tt_pass_start(ps % 2, q->wc % 2, gi % 2, q->p - c->grp * OB, c->g == 0, q->nev);
                    ++stats[TTF].passes;
                    phase = "control";
                    state = END;
                }
            }
        } else if (state == END) {
            if (tt_pass_idle() && !tt_ramp_busy()) {
                tt_pass_end();
                ++stats[TTF].membrane_writes;
                ++stats[TTF].membrane_reads;
                phase = "control";
                int wc = passes[ps].wc;
                ++ps;
                ++ps_done;
                if (ps == npass || passes[ps].wc != wc) ++w_done;
                state = PASS;
                if (ps < npass) { /* chain the next pass of the same group */
                    const Pass *r = &passes[ps];
                    const WChunk *c = &wcs[r->wc];
                    if (c->k == k && c->grp == grp && c->ncols_written == c->cols) {
                        tt_pass_start(ps % 2, r->wc % 2, (c->k * NG + c->grp) % 2, r->p - c->grp * OB,
                                      c->g == 0, r->nev);
                        ++stats[TTF].passes;
                        state = END;
                    }
                }
            } else
                phase = "compute";
        } else if (state == DRAIN) { /* the group's last ramp, then hand it to the window */
            if (grp < 0)
                state = ps >= npass ? DONE : PASS;
            else if (!tt_ramp_busy()) {
                t_grp_done[k * NG + grp] = 1;
                grp = -1;
                state = ps >= npass ? DONE : PASS;
            }
        }
        /* output window on the finished group's bank */
        if (cg < 0 && cmp_next < ngroups && t_grp_done[cmp_next] && t_tables_done >= cmp_next / NG) {
            cg = cmp_next;
            cx = 0;
            ct = -1;
        }
        if (cg >= 0) {
            if (ct < 0) {
                tt_cmp_load(cg % 2, cx);
                ++stats[TTF].membrane_reads;
                loaded = cx;
                ct = 0;
                ccount = 0;
            } else {
                tt_compare(cg % 2, cx, ct);
                compared = ct;
            }
        }
        t_ports_cycle();
        int issued = tt_issued(), creads = tt_col_reads();
        ev_reserved -= issued;
        stats[TTF].column_reads += (uint64_t)creads;
        if (issued) {
            phase = "compute";
            ++stats[TTF].issue_cycles;
        }
        lanes_t ops = tt_adds();
        int ocount, oidx[64];
        int outv = tt_out(&ocount, oidx);
        tick(TTF, phase, 0, ops);
        if (outv && !reuse) { /* compacted spike list of one output tick */
            stats[TTF].output_spikes += (uint64_t)ocount;
            int ob = beats((long)ocount * 16);
            stats[TTF].output_beats += (uint64_t)ob;
            for (int b = 0; b < ob; ++b) push_beat(&port[TTF][1], (Beat){ACT_NONE, GATE_NONE, 0, 0, 0, 0});
        }
        if (loaded >= 0) {
            int kk = cg / NG, p = (cg % NG) * OB + loaded;
            for (int l = 0; l < LANES && kk * LANES + l < m; ++l)
                got_sum[TTF][(size_t)p * m + kk * LANES + l] = tt_result(l);
        }
        if (compared >= 0) {
            int kk = cg / NG, p = (cg % NG) * OB + cx;
            lanes_t sp = tt_spikes();
            for (int l = 0; l < LANES; ++l)
                if (sp >> l & 1) {
                    int j = kk * LANES + l;
                    ++ccount;
                    if (got_level[TTF][(size_t)p * m + j] >= 0) {
                        fprintf(stderr, "lane fired twice p%d j%d\n", p, j);
                        exit(1);
                    }
                    got_level[TTF][(size_t)p * m + j] = K - compared;
                }
            if (++ct == K) {
                ct = -1;
                ++stats[TTF].compare_pixels;
                if (reuse) { /* the pixel's output as mask + values, as SparTen */
                    int act = 0, nz = 0;
                    for (int l = 0; l < LANES && kk * LANES + l < m; ++l) {
                        ++act;
                        nz += got_level[TTF][(size_t)p * m + kk * LANES + l] > 0;
                    }
                    stats[TTF].output_spikes += (uint64_t)nz;
                    int ob = beats((long)act + (long)nz * SPT_A);
                    stats[TTF].output_beats += (uint64_t)ob;
                    for (int b = 0; b < ob; ++b)
                        push_beat(&port[TTF][1], (Beat){ACT_NONE, GATE_NONE, 0, 0, 0, 0});
                }
                for (int l = 0; l < LANES && kk * LANES + l < m; ++l)
                    if (got_level[TTF][(size_t)p * m + kk * LANES + l] < 0)
                        got_level[TTF][(size_t)p * m + kk * LANES + l] = 0;
                if (++cx == grp_size(cg % NG)) {
                    t_grp_compared[cg] = 1;
                    ++cmp_next;
                    if (cmp_next % NG == 0) ++t_tiles_compared;
                    cg = -1;
                }
            }
        }
        assert(now < 4000000000L);
    }
}

#define FIELD(f)                                                                                             \
    fprintf(fout, "%s\"" #f "\":%" PRIu64, comma, a->f);                                                     \
    comma = ",";
static void report(const char *directory, uint64_t macs, uint64_t spikes) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/results.json", directory);
    FILE *fout = fopen(path, "wx");
    assert(fout);
    fprintf(fout,
            "{\"zskip\":%d,\"wenc\":%d,\"reuse\":%d,\"n\":%d,\"m\":%d,\"P\":%d,\"K\":%d,\"latency\":%d,"
            "\"units\":%d,\"lanes\":%d,\"block\":%d,"
            "\"ob\":%d,\"ch\":%d,\"edepth\":%d,\"nport\":%d,\"ref_macs\":%" PRIu64 ",\"ref_spikes\":%" PRIu64
            ",\"engines\":{",
            zskip, wenc, reuse, n, m, P, K, latency, UNITS, LANES, BLOCK, OB, CH, EDEPTH, NPORT, macs,
            spikes);
    const char *sep = "";
    for (int e = 0; e < 2; ++e) {
        if (!(engines >> e & 1)) continue;
        Stats *a = &stats[e];
        const char *comma = "";
        fprintf(fout, "%s\"%s\":{", sep, names[e]);
        sep = ",";
        FIELD(cycles);
        FIELD(compute_cycles);
        FIELD(control_cycles);
        FIELD(stall_cycles);
        FIELD(weight_reads);
        FIELD(arithmetic);
        FIELD(w_beats);
        FIELD(a_beats);
        FIELD(table_beats);
        FIELD(output_beats);
        FIELD(filter_mask_writes);
        FIELD(filter_value_writes);
        FIELD(filter_mask_reads);
        FIELD(input_writes);
        FIELD(jobs);
        FIELD(net_moves);
        FIELD(collector_pixels);
        FIELD(threshold_writes);
        FIELD(output_values);
        FIELD(column_writes);
        FIELD(column_reads);
        FIELD(membrane_writes);
        FIELD(membrane_reads);
        FIELD(event_pushes);
        FIELD(passes);
        FIELD(compare_pixels);
        FIELD(output_spikes);
        FIELD(events_read);
        FIELD(event_read_beats);
        FIELD(events_dropped);
        FIELD(issue_cycles);
        FIELD(patch_beats);
        FIELD(patch_entries);
        FIELD(patch_nnz);
        FIELD(patch_max_bits);
        FIELD(local_beats);
        FIELD(local_read_bits);
        FIELD(column_group_reads);
        FIELD(column_group_writes);
        FIELD(column_writes_skipped);
        fprintf(fout, "}");
        printf("%s: cycles=%" PRIu64 " (compute %" PRIu64 ", control %" PRIu64 ", stall %" PRIu64
               ") arithmetic=%" PRIu64 " W_beats=%" PRIu64 " A_beats=%" PRIu64 "\n",
               names[e], a->cycles, a->compute_cycles, a->control_cycles, a->stall_cycles, a->arithmetic,
               a->w_beats, a->a_beats);
    }
    fprintf(fout, "},\"verified\":true}\n");
    fclose(fout);
}

int main(int argc, char **argv) {
    const char *directory = NULL, *problem = NULL;
    int trace = 0;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--trace")) {
            trace = 1;
            continue;
        }
        if (a + 1 == argc) {
            fprintf(stderr, "Missing value for %s\n", argv[a]);
            return 2;
        }
        const char *key = argv[a++], *value = argv[a];
        if (!strcmp(key, "--output"))
            directory = value;
        else if (!strcmp(key, "--problem"))
            problem = value;
        else if (!strcmp(key, "--engines"))
            engines = !strcmp(value, "sparten") ? 1 : !strcmp(value, "ttfs") ? 2 : 3;
        else if (!strcmp(key, "--memory-latency"))
            latency = atoi(value);
        else if (!strcmp(key, "--zskip"))
            zskip = atoi(value);
        else if (!strcmp(key, "--wenc")) {
            if (strcmp(value, "list")) return 2;
            wenc = 1;
        } else if (!strcmp(key, "--geometry")) {
            reuse = sscanf(value, "%dx%dx%d", &g_h, &g_w, &g_c) == 3;
            if (!reuse) return 2;
        } else {
            fprintf(stderr, "Unknown argument %s\n", key);
            return 2;
        }
    }
    if (!directory || !problem || latency < 1 || latency > 100) {
        fprintf(stderr, "Use --problem FILE --output NEW_DIRECTORY [--engines both|sparten|ttfs "
                        "--memory-latency 1..100 --geometry HxWxC --wenc list --zskip 1 --trace]\n");
        return 2;
    }
    if (mkdir(directory, 0777)) {
        perror(directory);
        return 2;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/cycles.csv", directory);
    if (trace) {
        trace_file = fopen(path, "wx");
        assert(trace_file);
        fprintf(trace_file, "engine,cycle,phase\n");
    }
    load(problem);
    if (reuse && (n != 9 * g_c || P != g_h * g_w)) {
        fprintf(stderr, "--geometry %dx%dx%d does not match n=%d P=%d\n", g_h, g_w, g_c, n, P);
        return 2;
    }
    r5_open(directory, trace);
    for (int e = 0; e < 2; ++e) {
        r5_idle(e);
        r5_reset(e, 1);
        r5_tick(e);
        r5_reset(e, 0);
    }
    if (engines & 1) {
        s_plan();
        run_s();
    }
    if (engines & 2) {
        t_plan();
        run_t();
    }
    uint64_t macs = 0, spikes = 0;
    for (int p = 0; p < P; ++p)
        for (int i = 0; i < n; ++i)
            if (L_(p, i))
                for (int j = 0; j < m; ++j) macs += W_(j, i) != 0;
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < m; ++j) {
            size_t x = (size_t)p * m + j;
            spikes += first_expected[x] < K;
            for (int e = 0; e < 2; ++e) {
                if (!(engines >> e & 1)) continue;
                if (got_sum[e][x] != expected[x] || got_level[e][x] != expected_level(p, j)) {
                    fprintf(stderr, "Mismatch %s p%d j%d: sum %d vs %d, level %d vs %d\n", names[e], p, j,
                            got_sum[e][x], expected[x], got_level[e][x], expected_level(p, j));
                    return 1;
                }
            }
        }
    if ((engines & 1) && (stats[SPT].arithmetic != macs || stats[SPT].weight_reads != macs)) {
        fprintf(stderr, "SparTen MACs %" PRIu64 " != %" PRIu64 "\n", stats[SPT].arithmetic, macs);
        return 1;
    }
    if ((engines & 2) && stats[TTF].arithmetic != macs) {
        fprintf(stderr, "TTFS adds %" PRIu64 " != %" PRIu64 "\n", stats[TTF].arithmetic, macs);
        return 1;
    }
    report(directory, macs, spikes);
    r5_close();
    if (trace_file) fclose(trace_file);
    return 0;
}
