#!/usr/bin/env python3
"""Analytical row-stationary model of Eyeriss v1 (JSSC 2017): mappings, cycles, data movement."""

import argparse
import itertools
import json
import math
import os

ROWS, COLS = 12, 14
FSPAD, ISPAD, PSPAD = 224, 12, 24
GLB_BYTES = 100 * 1024  # ifmap + psum part of the 108 kB GLB
LINK = 64 * 60 / 200 / 64  # 64-b DRAM words per core cycle (64-b link at 60 MHz, core 200 MHz)


def placements(R, e):
    """(sets that fit, set columns per set row, rows used per set). Too-wide sets are split into
    segments stacked vertically (AlexNet CONV2), so a set uses R*seg rows and ceil(e/seg) columns."""
    seg = math.ceil(e / COLS)
    w = math.ceil(e / seg)
    h = R * seg
    if h > ROWS:
        return 0, 0, h
    return (ROWS // h) * (COLS // w), COLS // w, h


def pass_model(L, mp, b_if, b_ps):
    """Cycles and access counts of one layer under one mapping. Returns None if the mapping is illegal."""
    R, S, U, E, F, C, M, Hp, Wp, N = (L[k] for k in "R S U E F C M H W N".split())
    p, q, r, t, e, n, m = (mp[k] for k in "p q r t e n m".split())
    if p * q * S > FSPAD or q * S > ISPAD or p > PSPAD or e > E or n > N:
        return None
    fit, cols_per_row, h = placements(R, e)
    if fit < r * t or r > ROWS // h:
        return None
    rows_in = (e - 1) * U + R  # ifmap rows feeding a strip of e output rows
    if n * q * r * rows_in * Wp * b_if + n * m * e * F * b_ps > GLB_BYTES:
        return None
    pe = r * t * R * e
    step_mac = p * q * S + p
    step_if = r * rows_in * q * U
    step_ps = math.ceil(t * e * p / 4)
    step = max(step_mac, step_if, step_ps)
    # the filter GIN delivers 4 weights per cycle, but a PE writes one weight per cycle into its
    # spad (16-b SRAM port in Fig. 12; confirmed by the RTL in rtl/eyeriss_pe.sv)
    fload = max(math.ceil(p * t * q * r * R * S / 4), p * q * S)
    ramp = R * p + 16  # psum ripple up the set, pipeline fill and drain (RTL: 16)
    n_c, n_m, n_e, n_n = math.ceil(C / (q * r)), math.ceil(M / (p * t)), math.ceil(E / e), math.ceil(N / n)
    passes = n_c * n_m * n_e * n_n
    cyc_pass = fload + n * F * step + ramp
    cycles = passes * cyc_pass
    # data movement (counts of values)
    macs = N * M * E * F * C * R * S
    glb_if_rd = passes * n * q * r * rows_in * Wp
    glb_ps_wr = passes * n * p * t * e * F
    glb_ps_rd = glb_ps_wr * (n_c - 1) / n_c
    w_total = M * C * R * S
    glb_w = passes * p * t * q * r * R * S  # filter GLB -> spads
    dram_w = w_total * n_n * n_e  # each pass's filters come in from DRAM
    ifmap_strip = N * C * min(Hp, n_e * rows_in) * Wp
    dram_if = ifmap_strip * math.ceil(M / m)  # re-read once per group of m ofmap channels
    dram_of = N * M * E * F
    busy = passes * n * F * step_mac * pe  # PE-cycles spent on MACs and psum adds (zero ifmaps included)
    return dict(
        pe=pe,
        busy=busy,
        idle=cycles * pe - busy,
        passes=passes,
        step=step,
        step_mac=step_mac,
        fload=fload,
        cyc_pass=cyc_pass,
        cycles=cycles,
        util=macs / (cycles * ROWS * COLS),
        macs=macs,
        glb_if_rd=glb_if_rd,
        glb_ps_rd=glb_ps_rd,
        glb_ps_wr=glb_ps_wr,
        glb_w=glb_w,
        dram_w=dram_w,
        dram_if=dram_if,
        dram_of=dram_of,
        ifmap_strip=ifmap_strip,
    )


def search(L, b_if, b_ps, score):
    """(cycles, energy) of the fastest legal mapping (energy breaks ties), exhaustive over p, q, r, t, n, e."""
    best = None
    R, S, E, F, M, N, U, Wp = (L[k] for k in "R S E F M N U W".split())
    widths = sorted({math.ceil(E / k) for k in range(1, E + 1)})
    for p, q, e, n in itertools.product(range(1, PSPAD + 1), range(1, 5), widths, range(1, N + 1)):
        if p * q * S > FSPAD or q * S > ISPAD:
            continue
        fit, _, h = placements(R, e)
        if fit == 0:
            continue
        rows_in = (e - 1) * U + R
        for r in range(1, ROWS // h + 1):
            for t in range(1, fit // r + 1):
                room = GLB_BYTES - n * q * r * rows_in * Wp * b_if
                m = int(room // (n * e * F * b_ps)) // (p * t) * (p * t)
                m = min(m, math.ceil(M / (p * t)) * p * t)
                if m < p * t:
                    continue
                mp = dict(p=p, q=q, r=r, t=t, e=e, n=n, m=m)
                res = pass_model(L, mp, b_if, b_ps)
                if res is None:
                    continue
                s = score(L, res)
                if best is None or s < best[0]:
                    best = (s, mp, res)
    return best


def conv(name, H, W, R, S, E, F, C, M, U, N, pad=0):
    return dict(name=name, H=H + 2 * pad, W=W + 2 * pad, R=R, S=S, E=E, F=F, C=C, M=M, U=U, N=N)


# AlexNet, Table III (H/W already padded); mapping parameters as published.
ALEX = [
    (conv("CONV1", 227, 227, 11, 11, 55, 55, 3, 96, 4, 4), dict(m=96, n=1, e=7, p=16, q=1, r=1, t=2)),
    (conv("CONV2", 31, 31, 5, 5, 27, 27, 48, 256, 1, 4), dict(m=64, n=1, e=27, p=16, q=2, r=1, t=1)),
    (conv("CONV3", 15, 15, 3, 3, 13, 13, 256, 384, 1, 4), dict(m=64, n=4, e=13, p=16, q=4, r=1, t=4)),
    (conv("CONV4", 15, 15, 3, 3, 13, 13, 192, 384, 1, 4), dict(m=64, n=4, e=13, p=16, q=3, r=2, t=2)),
    (conv("CONV5", 15, 15, 3, 3, 13, 13, 192, 256, 1, 4), dict(m=64, n=4, e=13, p=16, q=3, r=2, t=2)),
]
# Table V: processing latency (ms), active PEs, GLB accesses (MB), DRAM accesses (MB), power (mW), zeros
ALEX_MEAS = [
    (16.5, 154, 18.5, 5.0, 332, 0.0001),
    (39.2, 135, 77.6, 4.0, 288, 0.387),
    (21.8, 156, 50.2, 3.0, 266, 0.725),
    (16.0, 156, 37.4, 2.1, 235, 0.793),
    (10.0, 156, 24.9, 1.3, 236, 0.776),
]

VGG16 = [
    ("CONV1-1", 224, 3, 64),
    ("CONV1-2", 224, 64, 64),
    ("CONV2-1", 112, 64, 128),
    ("CONV2-2", 112, 128, 128),
    ("CONV3-1", 56, 128, 256),
    ("CONV3-2", 56, 256, 256),
    ("CONV3-3", 56, 256, 256),
    ("CONV4-1", 28, 256, 512),
    ("CONV4-2", 28, 512, 512),
    ("CONV4-3", 28, 512, 512),
    ("CONV5-1", 14, 512, 512),
    ("CONV5-2", 14, 512, 512),
    ("CONV5-3", 14, 512, 512),
]
VGG16_MS = [38.0, 810.6, 405.3, 810.8, 204.0, 408.1, 408.1, 105.1, 210.0, 210.0, 48.3, 48.5, 48.5]

# ---- energy at 45 nm (pJ): measured Eyeriss chip energy per active PE-cycle (Fig. 16), scaled ------
A65, B65 = 6.54, 3.98
TECH = (45 / 65) * (1.1 / 1.0) ** 2  # C*V^2 scaling to NanGate45 typical 1.1 V
INT4 = 0.78  # width scaling, per component of Fig. 16(b) (see README)


def e_pe_cycle(nz, clock=True):
    e = (A65 + B65 * nz) * TECH * INT4
    if not clock:
        e *= 1 - clock_share(nz)
    return e


def clock_share(nz):
    return 0.45 + (0.329 - 0.45) * (nz - 0.224) / (1 - 0.224)


def e_layer(res, nz, clock=True):
    """A busy PE-cycle costs the measured per-PE energy; a PE waiting for filters or psums costs only
    its clock share (zero without the clock tree). Favourable to Eyeriss: the measured power was
    averaged over cycles that already included such waits."""
    busy = res["busy"] * e_pe_cycle(nz, clock)
    idle = res["idle"] * (e_pe_cycle(nz) * clock_share(nz) if clock else 0.0)
    return busy + idle


GLOB_BEAT = 50.8999  # pJ per 64-b beat of the original 256 kB global buffer (CACTI 45 nm)
DRAM_PJ_BIT = 3.97  # HBM2, as the original headline


def rlc_words(values, density, level_bits):
    """64-b RLC words: 5-b run + level per pair, last bit marks the end; runs over 31 add a pair."""
    pairs_per_word = (64 - 1) // (5 + level_bits)
    nnz = values * density
    zeros = values - nnz
    extra = max(0.0, zeros / 32 - nnz)  # zero runs longer than 31 need an extra pair
    return (nnz + extra) / pairs_per_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--layers",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "sparten_ref", "data", "vgg11_p128", "layers.json"
        ),
    )
    ap.add_argument("--reference-energy", required=True, help="energy_final_45nm_wireload_artifact.json")
    a = ap.parse_args()
    os.makedirs(a.output)
    out = {}
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- 1. AlexNet validation ---------------------------------------------------------------------
    say("1. AlexNet CONV1-5, batch 4, paper mappings (Table III), 16-b data. Model vs measured (Table V).")
    say(
        f"{'layer':6} {'passes':>6} {'step':>5} {'fload':>5} {'PEs':>8} {'latency ms':>16} {'GLB MB':>14} {'DRAM MB':>14}"
    )
    rows = []
    for (L, mp), (lat, pes, glb, dram, pw, zero) in zip(ALEX, ALEX_MEAS):
        res = pass_model(L, mp, 2, 2)
        ms = res["cycles"] / 200e3
        glb_mb = (
            2
            * (
                res["glb_if_rd"]
                + res["dram_if"]
                + res["glb_ps_rd"]
                + res["glb_ps_wr"]
                + res["glb_w"]
                + res["dram_w"]
            )
            / 1e6
        )
        nz_in = 1 - zero
        d_if = rlc_words(res["dram_if"], nz_in, 16) * 8 if L["name"] != "CONV1" else 2 * res["dram_if"]
        d_of = rlc_words(res["dram_of"], 1 - ALEX_MEAS[min(len(ALEX_MEAS) - 1, len(rows) + 1)][5], 16) * 8
        dram_mb = (2 * res["dram_w"] + d_if + d_of) / 1e6
        say(
            f"{L['name']:6} {res['passes']:>6} {res['step']:>5} {res['fload']:>5} {res['pe']:>4}/{pes:<3} "
            f"{ms:7.1f}/{lat:<7.1f} {glb_mb:6.1f}/{glb:<6.1f} {dram_mb:6.2f}/{dram:<6.1f}"
        )
        rows.append(
            dict(
                layer=L["name"],
                mapping=mp,
                model=res,
                model_ms=ms,
                meas_ms=lat,
                model_glb_MB=glb_mb,
                meas_glb_MB=glb,
                model_dram_MB=dram_mb,
                meas_dram_MB=dram,
            )
        )
    tot_m = sum(x["model_ms"] for x in rows)
    say(f"total processing latency: model {tot_m:.1f} ms, measured 103.5 ms")
    out["alexnet"] = rows

    # ---- 2. VGG-16 with searched mappings ------------------------------------------------------------
    say("\n2. VGG-16 CONV layers, batch 3, mappings from this model's search, 16-b data.")
    say(f"{'layer':8} {'p q r t e n m':>22} {'PEs':>4} {'util':>5} {'latency ms':>16}")

    def score16(L, res):
        nz = 0.5
        return (
            res["cycles"],
            res["cycles"] * res["pe"] * e_pe_cycle(nz)
            + (res["dram_w"] + res["dram_if"] + res["dram_of"]) / 4 * GLOB_BEAT,
        )

    rows = []
    for (name, hw, c, mm), meas in zip(VGG16, VGG16_MS):
        L = conv(name, hw, hw, 3, 3, hw, hw, c, mm, 1, 3, pad=1)
        s, mp, res = search(L, 2, 2, score16)
        ms = res["cycles"] / 200e3
        say(
            f"{name:8} {mp['p']:>3}{mp['q']:>3}{mp['r']:>3}{mp['t']:>3}{mp['e']:>4}{mp['n']:>3}{mp['m']:>4} "
            f"{res['pe']:>4} {res['util']:5.2f} {ms:7.1f}/{meas:<7.1f}"
        )
        rows.append(dict(layer=name, mapping=mp, model=res, model_ms=ms, meas_ms=meas))
    say(f"total: model {sum(x['model_ms'] for x in rows):.0f} ms, measured 3755.2 ms")
    out["vgg16"] = rows

    # ---- 3. VGG-11 CIFAR-10, INT4 ----------------------------------------------------------------------
    lay = {d["layer"]: d for d in json.load(open(a.layers))}
    ref_rows = {r["layer"]: r for r in json.load(open(a.reference_energy))["rows"]}
    b_if, b_ps = 0.5, 64 / 3 / 8  # 4-b ifmaps; 20-b psums, three per 64-b GLB word
    say("\n3. VGG-11 CIFAR-10 layers 1-7, INT4 weights and activations, 20-b psums, 45 nm.")
    say("   Mapping: lowest energy (on chip + off-core). 'fastest' = cycles of the fastest legal mapping.")
    res_all = {}
    for N in (1, 4, 16):
        say(f"\n   batch N = {N}")
        say(
            f"   {'layer':5} {'p q r t e n m':>22} {'PEs':>4} {'step':>5} {'fload':>5} {'passes':>6} {'util':>5} "
            f"{'cycles/img':>10} {'fastest':>10} {'uJ/img (clk)':>12} {'(no clk)':>9} {'glob uJ':>8}"
        )
        tot = dict(cycles=0, fast_cycles=0, e_clk=0, e_noclk=0, glob=0, dram_bytes=0, macs=0)
        per = []
        for l in range(1, 8):
            d = lay[l]
            C, M, Hs = d["n"] // 9, d["m"], d["H"]
            L = conv(f"L{l}", Hs, Hs, 3, 3, Hs, Hs, C, M, 1, N, pad=1)
            nz = d["input_density"]

            def offcore(res, nz=nz):
                return res["dram_w"] / 16 + rlc_words(res["dram_if"], nz, 4) + res["dram_of"] / 16

            fast = search(L, b_if, b_ps, lambda L, res: (res["cycles"], e_layer(res, nz)))[2]
            s, mp, res = search(
                L, b_if, b_ps, lambda L, res: (e_layer(res, nz) + offcore(res) * GLOB_BEAT, 0)
            )
            # off-core traffic in 64-b words: INT4 weights 16 per word, RLC fmaps
            nz_out = lay[l + 1]["input_density"] if l < 7 else 0.2
            words = (
                res["dram_w"] / 16 + rlc_words(res["dram_if"], nz, 4) + rlc_words(res["dram_of"], nz_out, 4)
            )
            e_clk = e_layer(res, nz) / N
            e_noclk = e_layer(res, nz, clock=False) / N
            glob = words * GLOB_BEAT / N
            tot["cycles"] += res["cycles"] / N
            tot["fast_cycles"] += fast["cycles"] / N
            tot["e_clk"] += e_clk
            tot["e_noclk"] += e_noclk
            tot["glob"] += glob
            tot["dram_bytes"] += words * 8 / N
            tot["macs"] += res["macs"] / N
            say(
                f"   {l:5} {mp['p']:>3}{mp['q']:>3}{mp['r']:>3}{mp['t']:>3}{mp['e']:>4}{mp['n']:>3}{mp['m']:>4} "
                f"{res['pe']:>4} {res['step']:>5} {res['fload']:>5} {res['passes']:>6} {res['util']:5.2f} "
                f"{res['cycles'] / N:>10,.0f} {fast['cycles'] / N:>10,.0f} {e_clk / 1e6:>12.2f} {e_noclk / 1e6:>9.2f} "
                f"{glob / 1e6:>8.2f}"
            )
            per.append(
                dict(
                    layer=l,
                    mapping=mp,
                    model=res,
                    fastest=fast,
                    nz=nz,
                    e_clk_pJ=e_clk,
                    e_noclk_pJ=e_noclk,
                    glob_pJ=glob,
                    words=words / N,
                )
            )
        say(
            f"   total: {tot['cycles']:,.0f} cycles/img (fastest mapping {tot['fast_cycles']:,.0f}), {tot['macs'] / 1e6:.1f} M dense MACs/img; on-chip "
            f"{tot['e_clk'] / 1e6:.1f} uJ (with clock tree) / {tot['e_noclk'] / 1e6:.1f} uJ (without); "
            f"global {tot['glob'] / 1e6:.1f} uJ; off-core {tot['dram_bytes'] / 1e3:.0f} kB/img"
        )
        res_all[N] = dict(layers=per, total=tot)
    out["vgg11"] = {str(k): v for k, v in res_all.items()}

    # ---- comparison with the original 45 nm TTFS / SparTen (layers 1-7) ----------------------------------
    T = {k: sum(ref_rows[l]["ttfs"][k] for l in range(1, 8)) for k in ("logic", "local", "glob")}
    Sp = {k: sum(ref_rows[l]["sparten"][k] for l in range(1, 8)) for k in ("logic", "local", "glob")}
    Tc = sum(ref_rows[l]["cycles"]["ttfs"] for l in range(1, 8))
    Sc = sum(ref_rows[l]["cycles"]["sparten"] for l in range(1, 8))
    # Layer-3 power-only rerun without wire load (cad): TTFS 54.4 -> 15.0 mW, SparTen unit 2.23 -> 2.12 mW.
    t_fix, s_fix = 15.0 / 54.4, 2.12 / 2.23
    ttfs_as = sum(T.values())
    ttfs_fix = T["logic"] * t_fix + T["local"] + T["glob"]
    spr_as = sum(Sp.values())
    spr_fix = Sp["logic"] * s_fix + Sp["local"] + Sp["glob"]
    say("\n4. Energy per image, layers 1-7, 45 nm, on chip + global-buffer traffic (uJ). Lower is better.")
    say(
        f"   the original TTFS     : {ttfs_as / 1e6:7.1f} as reported (clock wire-load artifact), "
        f"{ttfs_fix / 1e6:7.1f} with the layer-3 no-wire-load correction; {Tc:,.0f} cycles"
    )
    say(
        f"   the original SparTen  : {spr_as / 1e6:7.1f} as reported, {spr_fix / 1e6:7.1f} corrected; {Sc:,.0f} cycles"
    )
    for N in (1, 4, 16):
        t = res_all[N]["total"]
        lo, hi = t["e_noclk"] + t["glob"], t["e_clk"] + t["glob"]
        say(
            f"   Eyeriss N={N:<2}     : {lo / 1e6:7.1f} (no clock tree) .. {hi / 1e6:7.1f} (with clock tree); "
            f"{t['cycles']:,.0f} cycles (fastest {t['fast_cycles']:,.0f})  -> Eyeriss/TTFS(corrected) {lo / ttfs_fix:.1f}x .. {hi / ttfs_fix:.1f}x"
        )
    out["compare"] = dict(
        ttfs_reported=ttfs_as,
        ttfs_corrected=ttfs_fix,
        sparten_reported=spr_as,
        sparten_corrected=spr_fix,
        ttfs_cycles=Tc,
        sparten_cycles=Sc,
    )
    json.dump(out, open(os.path.join(a.output, "results.json"), "w"), indent=1, default=float)
    open(os.path.join(a.output, "report.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
