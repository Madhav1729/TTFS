#include "bridge.h"
#include "Vsparten_full.h"
#include "Vttfs_iv.h"
typedef Vttfs_iv Vttfs;
static const char *ttfs_name = "ttfs_iv";
#include "verilated.h"
#ifdef SAIF_TRACE
#include "verilated_saif_c.h"
typedef VerilatedSaifC Trace; // switching activity for Design Compiler (read_saif)
static const char *suffix = ".saif";
#else
#include "verilated_vcd_c.h"
typedef VerilatedVcdC Trace;
static const char *suffix = ".vcd";
#endif
#include <memory>
#include <string>
#include <type_traits>

static std::unique_ptr<VerilatedContext> context[2];
static std::unique_ptr<Vsparten_full> q;
static std::unique_ptr<Vttfs> s;
static std::unique_ptr<Trace> waves[2];

// Verilator maps a port to CData/SData/IData/QData or VlWide<N> by width.
template <class T> static void put(T &port, unsigned bit, unsigned width, uint64_t value) {
    static_assert(std::is_integral<T>::value, "scalar port");
    for (unsigned k = 0; k < width; ++k) {
        T one = T(T(1) << (bit + k));
        if ((value >> k) & 1)
            port = T(port | one);
        else
            port = T(port & ~one);
    }
}
template <std::size_t N> static void put(VlWide<N> &port, unsigned bit, unsigned width, uint64_t value) {
    for (unsigned k = 0; k < width; ++k) {
        unsigned b = bit + k;
        uint32_t one = 1u << (b % 32);
        if ((value >> k) & 1)
            port[b / 32] |= one;
        else
            port[b / 32] &= ~one;
    }
}
template <class T> static uint64_t get(const T &port, unsigned bit, unsigned width) {
    static_assert(std::is_integral<T>::value, "scalar port");
    return (uint64_t(port) >> bit) & (width == 64 ? ~0ull : ((1ull << width) - 1));
}
template <std::size_t N> static uint64_t get(const VlWide<N> &port, unsigned bit, unsigned width) {
    uint64_t value = 0;
    for (unsigned k = 0; k < width; ++k) {
        unsigned b = bit + k;
        value |= uint64_t((port[b / 32] >> (b % 32)) & 1u) << k;
    }
    return value;
}
static uint64_t bits(int value, int width) {
    return uint64_t(value) & ((1ull << width) - 1);
}
static int log2c(int v) {
    int b = 0;
    while ((1 << b) < v) ++b;
    return b;
}

extern "C" {
void r5_open(const char *directory, int trace) {
    for (int e = 0; e < 2; ++e) {
        context[e] = std::make_unique<VerilatedContext>();
        context[e]->traceEverOn(trace);
    }
    q = std::make_unique<Vsparten_full>(context[0].get(), "sparten_full");
    s = std::make_unique<Vttfs>(context[1].get(), ttfs_name);
    if (trace)
        for (int e = 0; e < 2; ++e) {
            waves[e] = std::make_unique<Trace>();
            if (e == 0)
                q->trace(waves[e].get(), 6);
            else
                s->trace(waves[e].get(), 6);
            std::string path = std::string(directory) + "/" + (e == 0 ? "sparten_full" : ttfs_name) + suffix;
            waves[e]->open(path.c_str());
        }
}
void r5_close(void) {
    q->final();
    s->final();
    for (auto &w : waves)
        if (w) {
            w->close();
            w.reset();
        }
    q.reset();
    s.reset();
}
void r5_tick(int e) {
    // 10 ns clock in the trace (rescaled for other clocks when converted for DC)
    for (int level = 0; level < 2; ++level) {
        if (e == 0) {
            q->clk = level;
            q->eval();
        } else {
            s->clk = level;
            s->eval();
        }
        if (waves[e]) waves[e]->dump(context[e]->time());
        context[e]->timeInc(5000);
    }
}
void r5_reset(int e, int value) {
    if (e == 0)
        q->reset = value;
    else
        s->reset = value;
}
void r5_idle(int e) {
    if (e == 0) {
        q->mask_write = 0;
        q->value_write = 0;
        q->in_write = 0;
        q->start = 0;
        q->collect = 0;
        q->th_write = 0;
    } else {
        s->wb_write = 0;
        s->ev_push_n = 0;
        s->pass_start = 0;
        s->pass_end = 0;
        s->threshold_write = 0;
        s->cmp_load = 0;
        s->compare = 0;
    }
}

void spf_mask_write(lanes_t enables, int buffer, int filter, const block_t *masks) {
    q->mask_write = 0;
    put(q->mask_write, 0, UNITS, enables);
    q->wbuf = buffer;
    q->wfil = filter;
    for (int u = 0; u < UNITS; ++u)
        for (int b = 0; b < BLOCK; b += 64) put(q->mask_data, u * BLOCK + b, 64, masks[u].w[b / 64]);
}
void spf_value_write(lanes_t enables, int buffer, int filter, int address, const int *values) {
    q->value_write = 0;
    put(q->value_write, 0, UNITS, enables);
    q->wbuf = buffer;
    q->wfil = filter;
    q->value_addr = address;
    for (int u = 0; u < UNITS; ++u) put(q->value_data, u * SPT_W, SPT_W, bits(values[u], SPT_W));
}
void spf_input_write(int buffer, const block_t *mask, const int *values) {
    q->in_write = 1;
    q->in_buf = buffer;
    for (int b = 0; b < BLOCK; b += 64) put(q->in_mask, b, 64, mask->w[b / 64]);
    for (int i = 0; i < BLOCK; ++i) put(q->in_data, i * SPT_A, SPT_A, bits(values[i], SPT_A));
}
void spf_start(int buffer, int in_buffer, int first, int bank, int slot, const int *active2,
               const int *dest2) {
    int fid = log2c(NF);
    q->start = 1;
    q->buf_sel = buffer;
    q->in_sel = in_buffer;
    q->first = first;
    q->bank = bank;
    q->slot = slot;
    for (int u = 0; u < UNITS; ++u)
        for (int f = 0; f < 2; ++f) {
            put(q->active, 2 * u + f, 1, uint64_t(active2[2 * u + f]));
            put(q->dest, (2 * u + f) * fid, fid, bits(dest2[2 * u + f], fid));
        }
}
void spf_collect(int bank, int pixels, const uint8_t *active_nf) {
    q->collect = 1;
    q->collect_bank = bank;
    q->collect_pixels = pixels;
    for (int f = 0; f < NF; ++f) put(q->collect_active, f, 1, active_nf[f]);
}
void spf_threshold(int fil, int t, int32_t value, int down) {
    q->th_write = 1;
    q->th_fil = fil;
    q->th_t = t;
    q->th_data = uint32_t(value);
    q->th_down = down;
}
int spf_busy(void) {
    q->eval();
    return q->busy;
}
int spf_holding(void) {
    q->eval();
    return q->holding;
}
int spf_net_busy(void) {
    q->eval();
    return q->net_busy;
}
int spf_col_busy(void) {
    q->eval();
    return q->col_busy;
}
int spf_out(int *pixel, uint8_t *mask_nf, uint8_t *packed_nf, int32_t *sums_nf) {
    q->eval();
    if (!q->out_valid) return 0;
    *pixel = q->out_pixel;
    for (int f = 0; f < NF; ++f) {
        mask_nf[f] = uint8_t(get(q->out_mask, f, 1));
        packed_nf[f] = uint8_t(get(q->out_packed, f * SPT_A, SPT_A));
        sums_nf[f] = int32_t(uint32_t(get(q->out_sums, f * 32, 32)));
    }
    return 1;
}
lanes_t spf_reads(void) {
    q->eval();
    return get(q->read_weight, 0, UNITS);
}
lanes_t spf_macs(void) {
    q->eval();
    return get(q->do_mac, 0, UNITS);
}
lanes_t spf_takes(void) {
    q->eval();
    return get(q->unit_done, 0, UNITS);
}
int spf_moves(void) {
    q->eval();
    return __builtin_popcountll(get(q->moves, 0, NPORT));
}

void tt_active(lanes_t active) {
    s->active = 0;
    put(s->active, 0, LANES, active);
}
void tt_wb_write(int buffer, int address, lanes_t mask, const int *packed) {
    s->wb_write = 1;
    s->wb_buf = buffer;
    s->wb_addr = address;
    s->wb_mask = 0;
    put(s->wb_mask, 0, LANES, mask);
    for (int l = 0; l < LANES; ++l) put(s->wb_packed, l * TT_W, TT_W, bits(packed[l], TT_W));
}
void tt_ev_push(int n, const int *index, const int *tick) {
    int ci = log2c(CH), ew = ci + TBITS;
    s->ev_push_n = n;
    for (int k = 0; k < n; ++k)
        put(s->ev_push_data, k * ew, ew, uint64_t(index[k]) | (uint64_t(tick[k]) << ci));
}
int tt_ev_count(void) {
    s->eval();
    return int(s->ev_count);
}
int tt_pass_idle(void) {
    s->eval();
    return s->pass_idle;
}
int tt_ramp_busy(void) {
    s->eval();
    return s->ramp_busy;
}
void tt_pass_start(int set, int wbuf, int bank, int slot, int first, int len) {
    s->pass_start = 1;
    s->pass_set = set;
    s->pass_wbuf = wbuf;
    s->pass_bank = bank;
    s->pass_slot = slot;
    s->pass_first = first;
    s->pass_len = len;
}
void tt_pass_end(void) {
    s->pass_end = 1;
}
void tt_threshold(int timestep, const int32_t *data, lanes_t down) {
    s->threshold_write = 1;
    s->th_t = timestep;
    s->threshold_down = 0;
    put(s->threshold_down, 0, LANES, down);
    for (int l = 0; l < LANES; ++l) put(s->threshold_data, l * 32, 32, uint32_t(data[l]));
}
void tt_cmp_load(int bank, int slot) {
    s->cmp_load = 1;
    s->cmp_bank = bank;
    s->cmp_slot = slot;
}
void tt_compare(int bank, int slot, int timestep) {
    s->compare = 1;
    s->cmp_bank = bank;
    s->cmp_slot = slot;
    s->cmp_t = timestep;
}
lanes_t tt_spikes(void) {
    return get(s->spike_mask, 0, LANES);
}
int tt_out(int *count, int *indices) {
    s->eval();
    if (!s->out_valid) return 0;
    int lr = log2c(LANES);
    *count = int(s->out_count);
    for (int i = 0; i < *count; ++i) indices[i] = int(get(s->out_idx, i * lr, lr));
    return 1;
}
int32_t tt_result(int lane) {
    s->eval();
    return int32_t(uint32_t(get(s->results, lane * 32, 32)));
}
lanes_t tt_adds(void) {
    s->eval();
    return get(s->do_add, 0, LANES);
}
int tt_issued(void) {
    s->eval();
    return int(s->st_issued);
}
int tt_col_reads(void) {
    s->eval();
    return int(s->st_reads);
}
}
