#!/usr/bin/env python3
"""PE switching activity on the VGG-16 layers and the toggle-fit power estimate it implies."""

import argparse
import json
import os
import re
import subprocess

import vcd_toggles

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL4 = os.path.join(HERE, "..", "..", "hardware", "final4")
CAD = os.path.join(FINAL4, "results", "cad_eyeriss", "reports_nowl")
ACT = os.path.join(FINAL4, "eyeriss_int4", "runs", "activity_01")
PE_TB = os.path.join(HERE, "build", "pe_tb")
WORKLOADS = ["w_0.6", "w_0.3", "w_loas"]


def pe_power(tag):
    """(dynamic uW, leakage uW) of eyeriss_pe without the filter spad, from a DC report_power -hier."""
    text = open(os.path.join(CAD, f"power_{tag}.rpt")).read()
    assert "Dynamic Power Units = 1uW" in text and "Leakage Power Units = 1nW" in text

    def row(name):
        m = re.search(
            rf"^\s*{name}\s+(?:\(\S+\)\s+)?([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
            text,
            re.M,
        )
        sw, it, lk, _ = (float(m.group(k)) for k in range(1, 5))
        return sw + it, lk * 1e-3

    d, l = row("eyeriss_pe")
    fd, fl = row("fspad")
    return d - fd, l - fl


def lstsq(A, y):
    """Least squares by the normal equations (3 unknowns; no numpy on this machine)."""
    n = len(A[0])
    M = [[sum(r[i] * r[j] for r in A) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(A, y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[p], b[i], b[p] = M[p], M[i], b[p], b[i]
        for k in range(n):
            if k != i:
                f = M[k][i] / M[i][i]
                M[k] = [x - f * z for x, z in zip(M[k], M[i])]
                b[k] -= f * b[i]
    return [b[i] / M[i][i] for i in range(n)]


def calibrate():
    pts = []
    for tag in ["idle"] + [f"layer{l}" for l in range(1, 8)]:
        dyn, leak = pe_power(tag)
        t = vcd_toggles.toggles(os.path.join(ACT, f"{tag}.vcd"))
        pts.append(
            dict(tag=tag, dyn_uW=dyn, leak_uW=leak, rate_data=t["rate_data"], rate_rest=t["rate_rest"])
        )
    c = lstsq([(1.0, p["rate_data"], p["rate_rest"]) for p in pts], [p["dyn_uW"] for p in pts])
    for p in pts:
        p["fit_uW"] = c[0] + c[1] * p["rate_data"] + c[2] * p["rate_rest"]
    busy = pts[1:]
    return dict(
        coef=dict(a_uW=c[0], b_uW_per_data_toggle=c[1], c_uW_per_other_toggle=c[2]),
        points=pts,
        max_abs_err_uW=max(abs(p["fit_uW"] - p["dyn_uW"]) for p in pts),
        max_rel_err_busy=max(abs(p["fit_uW"] - p["dyn_uW"]) / p["dyn_uW"] for p in busy),
        busy_leak_uW=sum(p["leak_uW"] for p in busy) / len(busy),
        idle=dict(dyn_uW=pts[0]["dyn_uW"], leak_uW=pts[0]["leak_uW"]),
    )


def convert(src, dst):
    """Rewrite one problem file into pe_tb's input order; return statistics of the layer."""
    with open(src) as f:
        n, m, P, K = map(int, f.readline().split())
        rest = f.read().split()
    C = n // 9
    w = rest[: m * n]
    th = rest[m * n : m * n + m * (K + 1)]
    x = rest[m * n + m * (K + 1) :]
    assert len(x) == P * n
    perm = [c * 9 + k for k in range(9) for c in range(C)]  # new index k*C + c <- old c*9 + k
    max_abs = 0
    with open(dst, "w") as o:
        o.write(f"{n} {m} {P} {K}\n")
        for j in range(m):
            row = w[j * n : (j + 1) * n]
            o.write(" ".join(row[i] for i in perm) + "\n")
            max_abs = max(max_abs, sum(abs(int(v)) for v in row))
        for j in range(m):
            o.write(" ".join(th[j * (K + 1) : (j + 1) * (K + 1)]) + "\n")
        nz = 0
        for p in range(P):
            row = x[p * n : (p + 1) * n]
            nz += sum(1 for v in row if v != "0")
            o.write(" ".join(row[i] for i in perm) + "\n")
    return dict(window_density=nz / (P * n), psum_bound=15 * max_abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument(
        "--work", required=True, help="scratch dir for converted files and VCDs (deleted after use)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--passes", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    cal = calibrate()
    print(
        "calibration:",
        json.dumps(cal["coef"]),
        f"max error {cal['max_abs_err_uW']:.1f} uW "
        f"({100 * cal['max_rel_err_busy']:.1f}% of a busy layer)",
    )
    co = cal["coef"]
    res = dict(calibration=cal, rows=a.rows, passes=a.passes, workloads={})
    for wl in WORKLOADS:
        lay = json.load(open(os.path.join(a.problems, wl, "layers.json")))["layers"]
        rows = []
        for L in lay:
            name = L["layer"]
            src = os.path.join(a.problems, wl, f"{name}.txt")
            conv = os.path.join(a.work, f"{wl}_{name}.txt")
            vcd = os.path.join(a.work, f"{wl}_{name}.vcd")
            js = os.path.join(a.out, f"{wl}_{name}.pe_tb.json")
            st = convert(src, conv)
            cmd = [
                PE_TB,
                "--layer",
                conv,
                "--f",
                str(L["W"]),
                "--p",
                "16",
                "--q",
                "4",
                "--rows",
                str(min(a.rows, L["H"])),
                "--passes",
                str(a.passes),
                "--pin",
                "1",
                "--idle",
                "0",
                "--vcd",
                vcd,
                "--json",
                js,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"pe_tb failed on {wl} {name}: {r.stdout} {r.stderr}")
            tb = json.load(open(js))
            t = vcd_toggles.toggles(vcd)
            os.remove(vcd)
            os.remove(conv)
            dyn = (
                co["a_uW"]
                + co["b_uW_per_data_toggle"] * t["rate_data"]
                + co["c_uW_per_other_toggle"] * t["rate_rest"]
            )
            row = dict(
                layer=name,
                F=L["W"],
                weight_density=L["weight_density"],
                input_density=L["input_density"],
                window_density=st["window_density"],
                psum_bound=st["psum_bound"],
                psum_ok=st["psum_bound"] < 2**19,
                tb_macs=tb["macs"],
                tb_zero_macs=tb["zero_macs"],
                tb_errors=tb["errors"],
                tb_cycles=tb["cycles"],
                rate_data=t["rate_data"],
                rate_rest=t["rate_rest"],
                dyn_uW=dyn,
                cmd=" ".join(cmd[1:]),
            )
            rows.append(row)
            print(
                f"{wl:7} {name:12} F {L['W']:2}  win-density {st['window_density']:.3f}  tb nz-MAC "
                f"{1 - tb['zero_macs'] / tb['macs']:.3f}  toggles {t['rate_data']:5.1f}/{t['rate_rest']:4.1f}  "
                f"dyn {dyn:5.1f} uW  psum bound {st['psum_bound']}{'' if row['psum_ok'] else ' (>2^19!)'}",
                flush=True,
            )
        res["workloads"][wl] = rows
    json.dump(res, open(os.path.join(a.out, "activity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
