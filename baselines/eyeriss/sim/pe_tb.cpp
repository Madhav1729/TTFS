// Testbench of one Eyeriss PE on a sample of a layer; checks every output psum.
//   pe_tb --layer L.txt --f F --p 16 --q 4 --rows 4 --passes 12 [--vcd out.saif] [--json out.json]
#include "Veyeriss_pe.h"
#include "verilated.h"
#ifdef SAIF_TRACE
#include "verilated_saif_c.h"
typedef VerilatedSaifC Trace;
#else
#include "verilated_vcd_c.h"
typedef VerilatedVcdC Trace;
#endif
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <vector>

static void die(const char *m) { std::fprintf(stderr, "error: %s\n", m); std::exit(2); }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    std::string layer, vcd, json;
    int F = 0, P_ = 16, Q = 4, ROWS = 4, PASSES = 3, PIN = 1, SEED = 1, IDLE = 16;
    double stall = 0.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
        if (a == "--layer") layer = nx();
        else if (a == "--f") F = std::stoi(nx());
        else if (a == "--p") P_ = std::stoi(nx());
        else if (a == "--q") Q = std::stoi(nx());
        else if (a == "--rows") ROWS = std::stoi(nx());
        else if (a == "--passes") PASSES = std::stoi(nx());
        else if (a == "--pin") PIN = std::stoi(nx());
        else if (a == "--stall") stall = std::stod(nx());
        else if (a == "--idle") IDLE = std::stoi(nx());
        else if (a == "--seed") SEED = std::stoi(nx());
        else if (a == "--vcd") vcd = nx();
        else if (a == "--json") json = nx();
        else if (a[0] == '+') continue;
        else die(("unknown option " + a).c_str());
    }
    const int S = 3, U = 1, PSW = 20;
    if (F < 1) die("--f (output row width) is required");
    if (P_ < 1 || P_ > 24 || Q < 1 || Q * S > 12 || P_ * Q * S > 224) die("bad p/q");
    FILE *f = std::fopen(layer.c_str(), "r");
    if (!f) die("cannot open layer");
    int n, m, P, K;
    if (std::fscanf(f, "%d %d %d %d", &n, &m, &P, &K) != 4) die("bad header");
    std::vector<int> w((size_t)m * n), x((size_t)P * n);
    for (auto &v : w) if (std::fscanf(f, "%d", &v) != 1) die("short weights");
    for (int j = 0; j < m; ++j) for (int t = 0; t <= K; ++t) { int d; if (std::fscanf(f, "%d", &d) != 1) die("short th"); }
    for (auto &v : x) if (std::fscanf(f, "%d", &v) != 1) die("short inputs");
    std::fclose(f);
    const int C = n / 9;
    const int Wrow = (F - 1) * U + S;
    auto wt = [&](int fi, int c, int ky, int s) { return w[(size_t)fi * n + (ky * 3 + s) * C + c]; };

    Veyeriss_pe *pe = new Veyeriss_pe;
    Trace *tr = nullptr;
    if (!vcd.empty()) { Verilated::traceEverOn(true); tr = new Trace; pe->trace(tr, 99); tr->open(vcd.c_str()); }
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> u01(0, 1);
    uint64_t t = 0;
    auto tick = [&]() {
        // VCD time unit 1 ps, 5 ns clock (200 MHz), so the SAIF toggle rates are per real 5 ns cycle
        pe->clk = 0; pe->eval(); if (tr) tr->dump(5000 * t);
        pe->clk = 1; pe->eval(); if (tr) tr->dump(5000 * t + 2500);
        ++t;
    };
    pe->reset = 1; pe->start = 0; pe->f_valid = 0; pe->i_valid = 0; pe->pi_valid = 0; pe->po_ready = 0;
    for (int k = 0; k < 4; ++k) tick();
    pe->reset = 0;
    pe->cfg_p = P_; pe->cfg_q = Q; pe->cfg_s = S; pe->cfg_u = U; pe->cfg_f = F; pe->cfg_rows = ROWS; pe->cfg_pin = PIN;

    long errors = 0, outputs = 0, macs = 0, zero_macs = 0, pass_cycles_total = 0;
    std::vector<long> pass_cycles;
    for (int pass = 0; pass < PASSES; ++pass) {
        const int f0 = (pass * P_) % m, c0 = (pass * Q) % C, ky = pass % 3;
        // streams
        std::deque<std::vector<int>> fbeats;
        std::vector<int> wl;
        for (int s = 0; s < S; ++s) for (int c = 0; c < Q; ++c) for (int fi = 0; fi < P_; ++fi)
            wl.push_back((c0 + c < C && f0 + fi < m) ? wt(f0 + fi, c0 + c, ky, s) : 0);
        for (size_t k = 0; k < wl.size(); k += 4) {
            std::vector<int> b(4, 0);
            for (size_t j = 0; j < 4 && k + j < wl.size(); ++j) b[j] = wl[k + j];
            fbeats.push_back(b);
        }
        std::vector<std::vector<std::vector<int>>> a(ROWS, std::vector<std::vector<int>>(Q, std::vector<int>(Wrow, 0)));
        std::deque<int> ivals;
        for (int r = 0; r < ROWS; ++r) {
            for (int col = 0; col < Wrow; ++col) for (int c = 0; c < Q; ++c) {
                int v = 0;
                if (col >= 1 && col <= F && c0 + c < C) v = x[(size_t)(((r * F + col - 1) % P)) * n + 4 * C + c0 + c];
                a[r][c][col] = v;
                ivals.push_back(v);
            }
        }
        std::deque<std::vector<long>> pin_beats;
        std::deque<long> expect;
        for (int r = 0; r < ROWS; ++r) for (int xo = 0; xo < F; ++xo) {
            std::vector<long> pins(P_, 0);
            for (int fi = 0; fi < P_; ++fi) {
                long acc = 0, below = 0;
                for (int s = 0; s < S; ++s) for (int c = 0; c < Q; ++c) {
                    int av = a[r][c][xo * U + s];
                    int wv = wl[(s * Q + c) * P_ + fi];
                    acc += (long)wv * av;
                    if (c0 + c < C && f0 + fi < m) below += (long)wt(f0 + fi, c0 + c, (ky + 1) % 3, s) * av;
                }
                pins[fi] = PIN ? below : 0;
                expect.push_back(acc + pins[fi]);
            }
            for (int k = 0; PIN && k < P_; k += 4) {
                std::vector<long> b(4, 0);
                for (int j = 0; j < 4 && k + j < P_; ++j) b[j] = pins[k + j];
                pin_beats.push_back(b);
            }
        }
        // count MACs (gated ones included) for the record
        for (int r = 0; r < ROWS; ++r) for (int xo = 0; xo < F; ++xo) for (int s = 0; s < S; ++s) for (int c = 0; c < Q; ++c) {
            macs += P_;
            if (a[r][c][xo * U + s] == 0) zero_macs += P_;
        }
        // run the pass
        while (pe->busy) tick();
        pe->start = 1; tick(); pe->start = 0;
        uint64_t t0 = t;
        size_t outs_expected = expect.size(), outs_seen = 0;
        while (outs_seen < outs_expected) {
            if (t - t0 > 10000000) die("timeout");
            // drive inputs (values set before the rising edge, consumed if ready)
            pe->f_valid = !fbeats.empty() && u01(rng) >= stall;
            if (pe->f_valid) { uint32_t d = 0; for (int j = 0; j < 4; ++j) d |= (uint32_t)(fbeats.front()[j] & 0xF) << (4 * j); pe->f_data = d; }
            pe->i_valid = !ivals.empty() && u01(rng) >= stall;
            if (pe->i_valid) pe->i_data = ivals.front();
            pe->pi_valid = PIN && !pin_beats.empty() && u01(rng) >= stall;
            if (pe->pi_valid) {
                for (int j = 0; j < 4; ++j) {
                    long v = pin_beats.front()[j] & ((1L << PSW) - 1);
                    int lo = j * PSW;
                    for (int b = 0; b < PSW; ++b) {
                        int bit = lo + b, wd = bit / 32, off = bit % 32;
                        if ((v >> b) & 1) pe->pi_data[wd] |= (1u << off); else pe->pi_data[wd] &= ~(1u << off);
                    }
                }
            }
            pe->po_ready = u01(rng) >= stall;
            pe->eval();
            bool f_take = pe->f_valid && pe->f_ready, i_take = pe->i_valid && pe->i_ready;
            bool p_take = pe->pi_valid && pe->pi_ready, o_take = pe->po_valid && pe->po_ready;
            if (o_take) {
                const int lanes = std::min(4, P_ - (int)(outs_seen % P_));
                for (int j = 0; j < lanes; ++j) {
                    long v = 0;
                    for (int b = 0; b < PSW; ++b) { int bit = j * PSW + b; v |= (long)((pe->po_data[bit / 32] >> (bit % 32)) & 1) << b; }
                    if (v & (1L << (PSW - 1))) v -= (1L << PSW);
                    if (v != expect.front()) {
                        if (errors < 10) std::fprintf(stderr, "pass %d out %zu lane %d: got %ld want %ld\n", pass, outs_seen, j, v, expect.front());
                        ++errors;
                    }
                    expect.pop_front();
                    ++outs_seen;
                    ++outputs;
                }
            }
            tick();
            if (f_take) fbeats.pop_front();
            if (i_take) ivals.pop_front();
            if (p_take) pin_beats.pop_front();
        }
        while (pe->busy) tick();
        pass_cycles.push_back((long)(t - t0));
        pass_cycles_total += (long)(t - t0);
        if (!fbeats.empty() || !ivals.empty() || !pin_beats.empty()) { std::fprintf(stderr, "unconsumed input\n"); ++errors; }
    }
    // idle tail (activity of a waiting PE)
    for (int k = 0; k < IDLE; ++k) tick();
    if (tr) { tr->close(); delete tr; }
    long ideal = 0;
    for (int pass = 0; pass < PASSES; ++pass) ideal += (long)P_ * Q * S + (long)ROWS * F * (P_ * Q * S + P_);
    std::printf("F %d p %d q %d rows %d passes %d: cycles %ld (model %ld), outputs %ld, MACs %ld (zero-gated %.1f%%), errors %ld\n",
                F, P_, Q, ROWS, PASSES, pass_cycles_total, ideal, outputs, macs, macs ? 100.0 * zero_macs / macs : 0.0, errors);
    if (!json.empty()) {
        FILE *o = std::fopen(json.c_str(), "w");
        std::fprintf(o, "{\"F\": %d, \"p\": %d, \"q\": %d, \"rows\": %d, \"passes\": %d, \"cycles\": %ld, \"model_cycles\": %ld, "
                        "\"outputs\": %ld, \"macs\": %ld, \"zero_macs\": %ld, \"errors\": %ld, \"stall\": %g, \"pin\": %d, \"vcd_cycles\": %llu}\n",
                     F, P_, Q, ROWS, PASSES, pass_cycles_total, ideal, outputs, macs, zero_macs, errors, stall, PIN, (unsigned long long)t);
        std::fclose(o);
    }
    delete pe;
    return errors ? 1 : 0;
}
