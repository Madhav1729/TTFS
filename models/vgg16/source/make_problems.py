"""Problem files for the v7 harness from our QAT exports (hardware/sim_runs/export_conv.py), one image.

    python make_problems.py LAYERS_DIR OUT_DIR --img 0 --density natural|0.5|0.1|loas

Every output pixel of the layer, in raster order, with input index i = c*9 + ky*3 + kx (the order of
weights.bin rows), zero padding 1, so that v7 --geometry HxWxC can account input reuse. Format (Madhav's
v6 problem file): "n m P K", m rows of n weights, per channel K thresholds + direction bit, P rows of n
input levels (level = K - tick, 0 = no spike). Weight sparsity study as in Madhav's prune_layers.py: keep
the largest-magnitude fraction `density` of the non-zero 4-bit weights (ties broken by a fixed random
order, since only the integer weights are exported), everything else unchanged; a hardware workload, not
a retrained network. The harness checks each layer exactly against its own (pruned) weights.
"loas": LoAS's lottery-ticket VGG-16 per-layer densities, as quoted in Madhav's prune_layers.py; the two
layers it does not list (64->64 and 128->128) take the next layer's value.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LOAS = {"features.3": 0.1762, "features.7": 0.1762, "features.10": 0.1258, "features.14": 0.1258,
        "features.17": 0.0741, "features.20": 0.0641, "features.24": 0.0325, "features.27": 0.0085,
        "features.30": 0.0110, "features.34": 0.0033, "features.37": 0.0018, "features.40": 0.0024}

ap = argparse.ArgumentParser()
ap.add_argument("layers", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--img", type=int, default=0)
ap.add_argument("--density", default="natural")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
summary = []
for d in sorted((x for x in a.layers.iterdir() if x.is_dir()), key=lambda x: json.loads((x / "meta.json").read_text())["index"]):
    m = json.loads((d / "meta.json").read_text())
    cin, cout, h, w, T = m["cin"], m["cout"], m["h"], m["w"], m["T"]
    n, P = 9 * cin, h * w
    wt = np.fromfile(d / "weights.bin", np.int8).reshape(cout, n).astype(np.int32)
    target = None if a.density == "natural" else LOAS[d.name] if a.density == "loas" else float(a.density)
    if target is not None:
        nz = np.flatnonzero(wt)
        keep = round(target * wt.size)
        if keep < nz.size:
            tie = np.random.default_rng(m["index"]).permutation(nz.size)
            order = np.lexsort((tie, -np.abs(wt.flat[nz])))      # largest magnitude first, random among equals
            drop = nz[order[keep:]]
            wt.flat[drop] = 0
    thr = np.fromfile(d / "thresholds.bin", np.int32).reshape(cout, T)
    down = np.fromfile(d / "down.bin", np.uint8)
    ticks = np.fromfile(d / f"img{a.img}_ticks.bin", np.uint8).reshape(cin, h, w).astype(np.int32)
    lev = np.where(ticks < T, T - ticks, 0)
    pad = np.pad(lev, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((P, cin, 9), np.int32)                      # im2col, i = c*9 + ky*3 + kx
    for ky in range(3):
        for kx in range(3):
            cols[:, :, ky * 3 + kx] = pad[:, ky:ky + h, kx:kx + w].reshape(cin, P).T
    cols = cols.reshape(P, n)
    with open(a.out / f"{d.name}.txt", "w") as f:
        f.write(f"{n} {cout} {P} {T}\n")
        np.savetxt(f, wt, fmt="%d")
        np.savetxt(f, np.concatenate([thr, down[:, None].astype(np.int32)], axis=1), fmt="%d")
        np.savetxt(f, cols, fmt="%d")
    rec = dict(layer=d.name, index=m["index"], n=n, m=cout, H=h, W=w, C=cin, P=P, K=T,
               geometry=f"{h}x{w}x{cin}", weight_density=float((wt != 0).mean()),
               natural_weight_density=m["weight_density"], input_density=float((lev != 0).mean()),
               macs_effectual=int(((cols != 0).astype(np.int64) @ (wt != 0).T.astype(np.int64)).sum()))
    summary.append(rec)
    print(rec["layer"], f"w-density {rec['weight_density']:.4f} in-density {rec['input_density']:.3f}", flush=True)
(a.out / "layers.json").write_text(json.dumps(dict(source=str(a.layers), img=a.img, density=a.density, layers=summary), indent=1) + "\n")
