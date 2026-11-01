"""Problem files: full-layer inputs and the 32-pixel samples used for switching activity."""

import json

from . import config

FULL = config.DATA / "problems"
SAMPLED = config.RESULTS / "problems_p32"


def geometry(workload, layer):
    layers = json.loads((FULL / workload / "layers.json").read_text())["layers"]
    return next(r["geometry"] for r in layers if r["layer"] == layer)


def sample(src, dst, pixels=config.SAMPLE_PIXELS):
    """Keep the first `pixels` output pixels: header, weights and thresholds unchanged."""
    tok = src.read_text().split()
    n, m, p, k = map(int, tok[:4])
    keep = min(p, pixels)
    w_end = 4 + n * m
    t_end = w_end + m * (k + 1)
    blocks = [f"{n} {m} {keep} {k}", " ".join(tok[4:w_end]), " ".join(tok[w_end:t_end])]
    blocks.append(" ".join(tok[t_end : t_end + keep * n]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(blocks) + "\n")


def sample_all():
    for w in config.WORKLOADS:
        for layer in config.LAYERS:
            sample(FULL / w / f"{layer}.txt", SAMPLED / w / f"{layer}.txt")
    return SAMPLED


def layer_info(workload, layer):
    layers = json.loads((FULL / workload / "layers.json").read_text())["layers"]
    return next(r for r in layers if r["layer"] == layer)


def to_channel_minor(src, dst):
    """Reorder inputs and weights from i = c*9 + k to i = k*C + c (the Eyeriss PE testbench's order)."""
    tok = src.read_text().split()
    n, m, p, k = map(int, tok[:4])
    c = n // 9
    perm = [ch * 9 + kk for kk in range(9) for ch in range(c)]
    w_end = 4 + n * m
    t_end = w_end + m * (k + 1)
    rows = [f"{n} {m} {p} {k}"]
    for j in range(m):
        row = tok[4 + j * n : 4 + (j + 1) * n]
        rows.append(" ".join(row[i] for i in perm))
    for j in range(m):
        rows.append(" ".join(tok[w_end + j * (k + 1) : w_end + (j + 1) * (k + 1)]))
    for q in range(p):
        row = tok[t_end + q * n : t_end + (q + 1) * n]
        rows.append(" ".join(row[i] for i in perm))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + "\n")
