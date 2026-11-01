"""Problem files from the VGG-16 per-layer exports (models/vgg16/layers), pruned to a target weight density."""

import json

import numpy as np

from . import config

LAYERS = config.ROOT / "models" / "vgg16" / "layers"
LOAS_DENSITY = {
    "features.3": 0.1762, "features.7": 0.1762, "features.10": 0.1258, "features.14": 0.1258,
    "features.17": 0.0741, "features.20": 0.0641, "features.24": 0.0325, "features.27": 0.0085,
    "features.30": 0.0110, "features.34": 0.0033, "features.37": 0.0018, "features.40": 0.0024,
}  # fmt: skip


def target_density(layer, density):
    if density == "natural":
        return None
    return LOAS_DENSITY[layer] if density == "loas" else float(density)


def prune(weights, target, seed):
    nonzero = np.flatnonzero(weights)
    keep = round(target * weights.size)
    if keep >= nonzero.size:
        return weights
    tie = np.random.default_rng(seed).permutation(nonzero.size)
    order = np.lexsort((tie, -np.abs(weights.flat[nonzero])))
    pruned = weights.copy()
    pruned.flat[nonzero[order[keep:]]] = 0
    return pruned


def im2col(levels):
    """P x (cin*9) input levels of a 3x3, pad-1 conv, input index c*9 + ky*3 + kx."""
    cin, h, w = levels.shape
    pad = np.pad(levels, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((h * w, cin, 9), np.int32)
    for ky in range(3):
        for kx in range(3):
            cols[:, :, ky * 3 + kx] = pad[:, ky : ky + h, kx : kx + w].reshape(cin, h * w).T
    return cols.reshape(h * w, cin * 9)


def export_layer(d, out, density, image):
    meta = json.loads((d / "meta.json").read_text())
    cin, cout, h, w, steps = meta["cin"], meta["cout"], meta["h"], meta["w"], meta["T"]
    n = 9 * cin
    weights = np.fromfile(d / "weights.bin", np.int8).reshape(cout, n).astype(np.int32)
    target = target_density(d.name, density)
    if target is not None:
        weights = prune(weights, target, meta["index"])
    thresholds = np.fromfile(d / "thresholds.bin", np.int32).reshape(cout, steps)
    down = np.fromfile(d / "down.bin", np.uint8)
    ticks = np.fromfile(d / f"img{image}_ticks.bin", np.uint8).reshape(cin, h, w).astype(np.int32)
    levels = np.where(ticks < steps, steps - ticks, 0)
    cols = im2col(levels)
    with open(out / f"{d.name}.txt", "w") as f:
        f.write(f"{n} {cout} {h * w} {steps}\n")
        np.savetxt(f, weights, fmt="%d")
        np.savetxt(f, np.concatenate([thresholds, down[:, None].astype(np.int32)], axis=1), fmt="%d")
        np.savetxt(f, cols, fmt="%d")
    effectual = int(((cols != 0).astype(np.int64) @ (weights != 0).T.astype(np.int64)).sum())
    return {
        "layer": d.name, "index": meta["index"], "n": n, "m": cout, "H": h, "W": w, "C": cin, "P": h * w,
        "K": steps, "geometry": f"{h}x{w}x{cin}", "weight_density": float((weights != 0).mean()),
        "natural_weight_density": meta["weight_density"], "input_density": float((levels != 0).mean()),
        "macs_effectual": effectual,
    }  # fmt: skip


def export(density, out, layers=LAYERS, image=0):
    out.mkdir(parents=True, exist_ok=False)
    dirs = sorted(
        (x for x in layers.iterdir() if x.is_dir()),
        key=lambda x: json.loads((x / "meta.json").read_text())["index"],
    )
    summary = [export_layer(d, out, density, image) for d in dirs]
    record = {"source": str(layers), "img": image, "density": density, "layers": summary}
    (out / "layers.json").write_text(json.dumps(record, indent=1) + "\n")
    return out
