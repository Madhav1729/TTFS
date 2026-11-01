#!/usr/bin/env python3
"""Eyeriss v1 INT4 energy per image on VGG-16 (CIFAR-100): row-stationary model + PE power + memories."""

import argparse
import csv
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("eyeriss_model", os.path.join(HERE, "model.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

ap = argparse.ArgumentParser()
ap.add_argument("--problems", required=True)
ap.add_argument("--activity", required=True)
ap.add_argument("--synth", required=True, help="directory with the PE's area.rpt and timing.rpt")
ap.add_argument("--cacti", nargs="+", required=True)
ap.add_argument("--workloads", nargs="+", default=["w_0.6", "w_0.3", "w_loas"])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
ap.add_argument("--json", required=True)
a = ap.parse_args()

# CACTI rows; a size/width present in several files must agree (same sweep settings)
C = {}
for path in a.cacti:
    for r in csv.DictReader(open(path)):
        if r["ok"] == "1" and r["tech_um"] == "0.045":
            k = (int(r["size_B"]), int(r["width_B"]))
            v = tuple(float(r[f]) for f in ("read_nJ", "write_nJ", "leak_mW", "area_mm2"))
            assert C.get(k, v) == v, f"CACTI files disagree on {k}"
            C[k] = v
rd = lambda s, w: C[(s, w)][0] * 1e3  # pJ
wr = lambda s, w: C[(s, w)][1] * 1e3
leak_c = lambda s, w: C[(s, w)][2]  # mW
area_c = lambda s, w: C[(s, w)][3]  # mm2

timing = open(os.path.join(a.synth, "timing.rpt")).read()
WNS = min(0.0, min(float(x) for x in re.findall(r"slack \((?:MET|VIOLATED)\)\s+([-\d.]+)", timing)))
CLK = 5.0 - WNS  # ns; dynamic energy at 5 ns, leakage over the real clock
NPE = M.ROWS * M.COLS
GLB_BANKS = 27
GB = 262144
B_IF, B_PS = 0.5, 64 / 3 / 8  # GLB bytes per ifmap (4 b) and per psum (20 b, 3 per word)


def pe_area():
    vals = {}
    for line in open(os.path.join(a.synth, "area.rpt")):
        m = re.match(r"^(\S+)\s+([\d.]+)\s+[\d.]+\s", line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    return (vals["eyeriss_pe"] - vals["fspad"]) * 1e-6


act = json.load(open(a.activity))
cal = act["calibration"]
IDLE_DYN, IDLE_LEAK = cal["idle"]["dyn_uW"] * 1e-3, cal["idle"]["leak_uW"] * 1e-3  # mW
BUSY_LEAK = cal["busy_leak_uW"] * 1e-3
dyn = {w: {r["layer"]: r["dyn_uW"] * 1e-3 for r in rows} for w, rows in act["workloads"].items()}
win = {w: {r["layer"]: r["window_density"] for r in rows} for w, rows in act["workloads"].items()}
lays = {w: json.load(open(os.path.join(a.problems, w, "layers.json")))["layers"] for w in a.workloads}
names = [L["layer"] for L in lays[a.workloads[0]]]
for w in a.workloads:  # activations are the same image in every workload
    assert [L["layer"] for L in lays[w]] == names
    assert all(L["input_density"] == L0["input_density"] for L, L0 in zip(lays[w], lays[a.workloads[0]]))


def cost(res, mp, w, i):
    """Energy parts (pJ, whole batch) of layer i of workload w under one mapping."""
    L = lays[w][i]
    name = L["layer"]
    nz_in = L["input_density"]
    # stored ofmap density: the next layer's input; after a 2x2 max-pool that is an upper bound (a pooled
    # value is non-zero if any of its 4 is); the last layer's output is bounded by 1
    nz_out = lays[w][i + 1]["input_density"] if i + 1 < len(names) else 1.0
    idle_pe_cycles = (NPE - res["pe"]) * res["cycles"]
    dyn_e = (res["busy"] * dyn[w][name] + res["idle"] * IDLE_DYN) * 5.0
    leak_e = ((res["busy"] * BUSY_LEAK + res["idle"] * IDLE_LEAK) + idle_pe_cycles * IDLE_LEAK) * CLK
    logic = dyn_e + leak_e
    noc = 0.02 * logic
    p, q, S = mp["p"], mp["q"], 3
    fsp = (
        res["macs"] * win[w][name] * rd(128, 1)
        + res["passes"] * res["pe"] * p * q * S * wr(128, 1)
        + NPE * res["cycles"] * leak_c(128, 1) * CLK
    )
    glb_rd = res["glb_if_rd"] / 16 + res["glb_ps_rd"] / 3 + res["glb_w"] / 16 + res["dram_of"] / 3
    glb_wr = res["dram_if"] / 16 + res["glb_ps_wr"] / 3 + res["dram_w"] / 16
    glb = glb_rd * rd(4096, 8) + glb_wr * wr(4096, 8) + GLB_BANKS * res["cycles"] * leak_c(4096, 8) * CLK
    g_w = res["dram_w"] / 16
    g_in = M.rlc_words(res["dram_if"], nz_in, 4)
    g_out = M.rlc_words(res["dram_of"], nz_out, 4)
    glob = (g_w + g_in) * rd(GB, 8) + g_out * wr(GB, 8)
    return dict(
        logic=logic,
        logic_dyn=dyn_e,
        logic_leak=leak_e,
        spads=fsp,
        glb=glb,
        glob=glob,
        noc=noc,
        total=logic + noc + fsp + glb + glob,
        glob_beats=g_w + g_in + g_out,
        glob_w_beats=g_w,
        glob_in_beats=g_in,
        glob_out_beats=g_out,
    )


def mappings(L):
    """Every legal mapping of model.search's enumeration (m: most ofmap channels whose psums fit)."""
    R, S, E, F, Mo, Nb, U, Wp = (L[k] for k in "R S E F M N U W".split())
    widths = sorted({-(-E // k) for k in range(1, E + 1)})
    for p in range(1, M.PSPAD + 1):
        for q in range(1, 5):
            if p * q * S > M.FSPAD or q * S > M.ISPAD:
                continue
            for e in widths:
                fit, _, h = M.placements(R, e)
                if fit == 0:
                    continue
                rows_in = (e - 1) * U + R
                for n in range(1, Nb + 1):
                    for r in range(1, M.ROWS // h + 1):
                        for t in range(1, fit // r + 1):
                            room = M.GLB_BYTES - n * q * r * rows_in * Wp * B_IF
                            m = int(room // (n * e * F * B_PS)) // (p * t) * (p * t)
                            m = min(m, -(-Mo // (p * t)) * p * t)
                            if m < p * t:
                                continue
                            mp = dict(p=p, q=q, r=r, t=t, e=e, n=n, m=m)
                            res = M.pass_model(L, mp, B_IF, B_PS)
                            if res is not None:
                                yield mp, res


ref = a.workloads[0]
area_pe = NPE * (pe_area() + area_c(128, 1))
area_glb = GLB_BANKS * area_c(4096, 8)
out = dict(
    clock_ns=CLK,
    wns_ns=WNS,
    area_mm2_no_glb=area_pe,
    area_mm2_glb=area_glb,
    pe_logic_area_mm2=pe_area(),
    fspad_area_mm2=area_c(128, 1),
    batches={},
)
for N in a.batches:
    per_layer = []
    for i, name in enumerate(names):
        L0 = lays[ref][i]
        Lg = M.conv(name, L0["H"], L0["W"], 3, 3, L0["H"], L0["W"], L0["n"] // 9, L0["m"], 1, N, pad=1)
        best = {w: None for w in a.workloads}  # each workload's own optimum (check)
        fastest = None
        for mp, res in mappings(Lg):
            for w in a.workloads:
                c = cost(res, mp, w, i)
                if best[w] is None or c["total"] < best[w][0]:
                    best[w] = (c["total"], mp, res)
                if w == ref and (
                    fastest is None or (res["cycles"], c["total"]) < (fastest[1]["cycles"], fastest[0])
                ):
                    fastest = (c["total"], res, mp)
        _, mp, res = best[ref]
        row = dict(
            layer=name,
            geometry=L0["geometry"],
            C=L0["n"] // 9,
            M=L0["m"],
            mapping=mp,
            pe=res["pe"],
            passes=res["passes"],
            step=res["step"],
            fload=res["fload"],
            cycles_per_img=res["cycles"] / N,
            macs_per_img=res["macs"] / N,
            util=res["util"],
            fastest=dict(mapping=fastest[2], cycles_per_img=fastest[1]["cycles"] / N),
            workloads={},
        )
        for w in a.workloads:
            c = cost(res, mp, w, i)
            row["workloads"][w] = dict(
                parts_uJ_per_img={k: v / N / 1e6 for k, v in c.items() if not k.endswith("beats")},
                glob_beats_per_img=c["glob_beats"] / N,
                own_best=dict(
                    mapping=best[w][1],
                    energy_uJ_per_img=best[w][0] / N / 1e6,
                    cycles_per_img=best[w][2]["cycles"] / N,
                ),
                pe_dyn_mW=dyn[w][name],
                window_density=win[w][name],
                weight_density=lays[w][i]["weight_density"],
                input_density=lays[w][i]["input_density"],
            )
        per_layer.append(row)
        print(
            f"N={N:<2} {name:12} p{mp['p']} q{mp['q']} r{mp['r']} t{mp['t']} e{mp['e']} n{mp['n']} m{mp['m']}  "
            f"{res['pe']:3d} PEs  {res['cycles'] / N:10,.0f} cyc/img  util {res['util']:.2f}  "
            + "  ".join(
                f"{w} {row['workloads'][w]['parts_uJ_per_img']['total']:6.2f} uJ" for w in a.workloads
            ),
            flush=True,
        )
    tot = {}
    for w in a.workloads:
        t = {
            k: sum(r["workloads"][w]["parts_uJ_per_img"][k] for r in per_layer)
            for k in per_layer[0]["workloads"][w]["parts_uJ_per_img"]
        }
        t["own_best_total"] = sum(r["workloads"][w]["own_best"]["energy_uJ_per_img"] for r in per_layer)
        t["glob_beats"] = sum(r["workloads"][w]["glob_beats_per_img"] for r in per_layer)
        tot[w] = t
    cyc = sum(r["cycles_per_img"] for r in per_layer)
    out["batches"][str(N)] = dict(
        cycles_per_img=cyc,
        fastest_cycles_per_img=sum(r["fastest"]["cycles_per_img"] for r in per_layer),
        latency_us_per_img=cyc * CLK * 1e-3,
        macs_per_img=sum(r["macs_per_img"] for r in per_layer),
        energy_uJ_per_img=tot,
        layers=per_layer,
    )
    print(
        f"N={N}: {cyc:,.0f} cycles/img, {cyc * CLK * 1e-3:,.0f} us; "
        + "; ".join(
            f"{w} {tot[w]['total']:.1f} uJ (own-optimum mappings {tot[w]['own_best_total']:.1f})"
            for w in a.workloads
        )
    )
# one row per (batch, workload): the numbers that go in the comparison table
out["summary"] = [
    dict(
        batch=N,
        workload=w,
        cycles_per_img=b["cycles_per_img"],
        clock_ns=CLK,
        latency_us_per_img=b["latency_us_per_img"],
        area_mm2_no_glb=area_pe,
        **{
            f"{k}_uJ": b["energy_uJ_per_img"][w][k]
            for k in ("logic", "logic_dyn", "logic_leak", "spads", "glb", "glob", "noc", "total")
        },
        glob_beats_per_img=b["energy_uJ_per_img"][w]["glob_beats"],
    )
    for N, b in ((int(k), v) for k, v in out["batches"].items())
    for w in a.workloads
]
print(f"clock {CLK:.2f} ns (WNS {WNS:.2f}); area without GLB {area_pe:.3f} mm2, GLB {area_glb:.3f} mm2")
json.dump(out, open(a.json, "w"), indent=1)
