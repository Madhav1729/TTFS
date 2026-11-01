"""BERT-base encoder layer (Otters, layer 5) in five spike codes and two QNN baselines, on the RTL simulators."""

import json

import numpy as np

from . import build, config

PROFILE = config.DATA / "lm" / "profile_otters.json"
LM = config.RESULTS / "lm"
TWIN_DELTA_PJ = {"w1": 0.0, "w4": 0.301 - 0.090}  # QNN twin: 4x4 MAC cells vs TTFS lane adder cells, per op


def root(weights):
    return LM / weights


BERT_WEIGHTS = config.DATA / "lm" / "bert_layer5_w4.npz"
WEIGHT_SITES = {
    "qkv": "attention.self.query",
    "attn_out": "attention.output.dense",
    "ffn1": "intermediate.dense",
    "ffn2": "output.dense",
}


OTTERS_CHECKPOINT = config.ROOT / "models" / "otters" / "checkpoint" / "pytorch_model.bin"


def quantize_otters(checkpoint=OTTERS_CHECKPOINT, layer=5):
    """Real BERT encoder weights (Otters latent weights) to 4 bits per output channel, LSQ initial step."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out = {}
    for name, site in WEIGHT_SITES.items():
        w = state[f"bert.encoder.layer.{layer}.{site}.weight"].float()
        step = 2 * w.abs().mean(1, keepdim=True) / np.sqrt(7)
        out[name] = torch.clamp(torch.round(w / step), -8, 7).to(torch.int8).numpy()
    BERT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BERT_WEIGHTS, **out)
    return {k: round(float((v == 0).mean()), 3) for k, v in out.items()}


def bert_weights(name, weights):
    """Layer-5 weight matrix: 4-bit, or its sign for the binary (Otters W1) case."""
    w4 = np.load(BERT_WEIGHTS)[name].astype(int)
    return w4 if weights == "w4" else np.where(w4 >= 0, 1, -1)


TOKENS = 32
SEQUENCE = 128
D, FFN = 768, 3072
MATMULS = {  # name: (inputs, outputs, profiled site, count per layer)
    "qkv": (D, D, "attention.self.query", 3),
    "attn_out": (D, D, "attention.output.dense", 1),
    "ffn1": (D, FFN, "intermediate.dense", 1),
    "ffn2": (FFN, D, None, 1),
}
CODES = ("offset", "bipolar", "tworail", "bitser", "rate")
# ticks per code: sign-magnitude codes carry |q| <= 8 (T = 2^(n-1)); the offset code needs 2^n - 1
CODE_STEPS = {"offset": 15, "bipolar": 8, "tworail": 8, "bitser": 8, "rate": 8}
QNN_CODES = {"qnn": 15, "offset": 15}
FFN2_SILENT = 0.5


def tick_distribution(site, layer=5):
    hist = json.loads(PROFILE.read_text())["hist"][f"bert.encoder.layer.{layer}.{site}"]
    p = np.array(hist, dtype=float)
    return p / p.sum()


def signed_levels(site, shape, rng):
    """Signed q in -8..7 drawn from the profile; FFN2's input (unprofiled, post-activation) is unsigned."""
    if site is None:
        q = rng.geometric(0.35, size=shape).clip(1, 7)
        return np.where(rng.random(shape) < FFN2_SILENT, 0, q)
    ticks = rng.choice(16, size=shape, p=tick_distribution(site))
    offset = np.where(ticks == 15, 0, 15 - ticks)
    return offset - 8


def encode(q, w, code):
    """(levels P x n', weights m x n') of one code; levels are unsigned ticks-to-go (0 = silent)."""
    mag, pos, neg = np.abs(q), np.maximum(q, 0), np.maximum(-q, 0)
    if code == "offset":
        return (q if q.min() >= 0 else q + 8), w
    if code == "bipolar":
        return mag, w
    if code == "tworail":
        return np.concatenate([pos, neg], 1), np.concatenate([w, np.clip(-w, -8, 7)], 1)
    if code == "bitser":
        bits = [((mag >> b) & 1) * (1 << b) for b in range(4)]
        return np.concatenate(bits, 1), np.concatenate([w] * 4, 1)
    if code == "rate":
        copies = [(mag > k).astype(int) for k in range(8)]
        return np.concatenate(copies, 1), np.concatenate([w] * 8, 1)
    raise KeyError(code)


def thresholds(levels, w, steps):
    """Per-channel thresholds at quantiles of its sums, so about half the outputs are silent."""
    v = levels.astype(np.int64) @ w.T.astype(np.int64)
    qs = np.linspace(0.97, 0.5, steps)
    th = np.quantile(v, qs, axis=0).T.round().astype(np.int64)
    th = np.maximum.accumulate(th[:, ::-1], axis=1)[:, ::-1]
    return th.clip(-32767, 32767)


def write_problem(path, levels, w, steps):
    n, m, p = w.shape[1], w.shape[0], levels.shape[0]
    th = thresholds(levels, w, steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{n} {m} {p} {steps}\n")
        np.savetxt(f, w, fmt="%d")
        np.savetxt(f, np.concatenate([th, np.zeros((m, 1), np.int64)], 1), fmt="%d")
        np.savetxt(f, levels, fmt="%d")


def make_problems(weights, seed=0):
    rng = np.random.default_rng(seed)
    stats = {}
    for name, (n, m, site, _) in MATMULS.items():
        q = signed_levels(site, (TOKENS, n), rng)
        w = bert_weights(name, weights)
        stats[name] = {
            "silent": float((q == 0).mean()),
            "mean_abs": float(np.abs(q).mean()),
            "bits": float(sum(((np.abs(q) >> b) & 1).mean() for b in range(4))),
        }
        for code in CODES:
            levels, wc = encode(q, w, code)
            write_problem(root(weights) / "problems" / code / f"{name}.txt", levels, wc, CODE_STEPS[code])
        write_problem(root(weights) / "problems" / "qnn" / f"{name}.txt", np.abs(q), w, QNN_CODES["qnn"])
    (root(weights) / "activation_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def _simulate(job):
    import subprocess

    weights, engine, code, name = job
    out = root(weights) / "sims" / engine / code / name
    out.parent.mkdir(parents=True, exist_ok=True)
    problem = root(weights) / "problems" / code / f"{name}.txt"
    flags = config.SIM_FLAGS[engine]
    steps = CODE_STEPS[code] if engine == "ttfs" else QNN_CODES[code]
    sim = build.sim_dir(1, steps) / "sim"
    cmd = [str(sim), "--problem", str(problem), "--output", str(out), *flags]
    subprocess.run(["rm", "-rf", str(out)], check=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = json.loads((out / "results.json").read_text())
    if not result["verified"]:
        raise RuntimeError(f"output check failed: {job}")
    return job, result["engines"][engine]


def simulate_all(weights, jobs=8):
    from concurrent.futures import ThreadPoolExecutor

    work = [(weights, "ttfs", c, n) for c in CODES for n in MATMULS]
    work += [(weights, "sparten", c, n) for c in QNN_CODES for n in MATMULS]
    with ThreadPoolExecutor(jobs) as pool:
        return dict(pool.map(_simulate, work))


EXPANDED = {"tworail", "bitser", "rate"}


def _stats(weights, engine, code, name):
    path = root(weights) / "sims" / engine / code / name / "results.json"
    return json.loads(path.read_text())["engines"][engine]


def fair_ttfs(weights, code, name, cacti):
    """Stats and memory of one code with bipolar's weight storage, weight traffic and per-pass overhead."""
    from . import memory

    s, ref = dict(_stats(weights, "ttfs", code, name)), _stats(weights, "ttfs", "bipolar", name)
    if code in EXPANDED:
        overhead = ref["cycles"] - ref["arithmetic"] / config.LANES
        s["cycles"] = round(s["arithmetic"] / config.LANES + overhead)
    e = memory.ttfs(s, cacti, CODE_STEPS[code])
    if code in EXPANDED:
        e_ref = memory.ttfs(ref, cacti, CODE_STEPS["bipolar"])
        e["wbanks"], e["glb_w"] = e_ref["wbanks"], e_ref["glb_w"]
    return s, sum(e.values())


def energy_table(weights):
    """Energy per encoder layer and 128-token sequence for every code / baseline, plus attention's common part."""
    from . import calibrate, memory
    from .cacti import Cacti

    cacti, models = Cacti(), calibrate.fit()
    clocks = {"ttfs": 5.36, "sparten": 5.26}
    scale = SEQUENCE / TOKENS
    rows = {}
    for engine, codes in (("ttfs", CODES), ("sparten", tuple(QNN_CODES))):
        for code in codes:
            cycles = logic = mem = ops = 0.0
            for name, (_, _, _, count) in MATMULS.items():
                if engine == "ttfs":
                    s, e_mem = fair_ttfs(weights, code, name, cacti)
                    p = calibrate.power(models, "lane", s).scale(config.LANES)
                    p = p + calibrate.power(models, "front", s)
                else:
                    s = _stats(weights, engine, code, name)
                    p = calibrate.power(models, "unit", s).scale(config.UNITS)
                    p = p + calibrate.power(models, "peri", s)
                    e_mem = sum(memory.sparten(s, cacti).values()) + memory.sparten_table(s, cacti)
                k = count * scale
                cycles += k * s["cycles"]
                logic += k * p.energy_pJ(s["cycles"], clocks[engine], config.PERIOD_NS)
                mem += k * e_mem
                ops += k * s["arithmetic"]
            rows[(engine, code)] = {
                "cycles": cycles,
                "latency_us": cycles * clocks[engine] * 1e-3,
                "logic_uJ": logic * 1e-6,
                "mem_uJ": mem * 1e-6,
                "energy_uJ": (logic + mem) * 1e-6,
                "ops": ops,
            }
    sp = rows[("sparten", "qnn")]
    attention_uJ = 2 * SEQUENCE * SEQUENCE * D * sp["energy_uJ"] / sp["ops"]
    twin = dict(rows[("ttfs", "bipolar")])
    twin["logic_uJ"] += twin["ops"] * TWIN_DELTA_PJ[weights] * 1e-6
    twin["energy_uJ"] = twin["logic_uJ"] + twin["mem_uJ"]
    rows[("twin", "bipolar")] = twin
    return rows, attention_uJ


NAMES = {
    ("ttfs", "bipolar"): "TTFS bipolar",
    ("ttfs", "tworail"): "TTFS two-rail unipolar",
    ("ttfs", "offset"): "TTFS offset unipolar",
    ("ttfs", "bitser"): "Bit-serial, one event per non-zero bit (same lanes)",
    ("ttfs", "rate"): "Rate / count spikes (same lanes)",
    ("sparten", "qnn"): "SparTen W4A4, zero-skipping QNN",
    ("sparten", "offset"): "SparTen W4A4, no zeros to skip (offset inputs)",
    ("twin", "bipolar"): "QNN twin: same dataflow, MAC lanes",
}


def write_table(weights):
    rows, attention = energy_table(weights)
    ref = rows[("ttfs", "bipolar")]
    lines = [
        "| Design | Latency (us) | Logic (uJ) | Memory (uJ) | Linear layers (uJ) | vs TTFS bipolar | "
        "With attention (uJ) | vs TTFS bipolar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in NAMES.items():
        r = rows[key]
        total = r["energy_uJ"] + attention
        lines.append(
            f"| {name} | {r['latency_us']:.0f} | {r['logic_uJ']:.1f} | {r['mem_uJ']:.1f} | "
            f"{r['energy_uJ']:.1f} | {r['energy_uJ'] / ref['energy_uJ']:.2f}x | {total:.1f} | "
            f"{total / (ref['energy_uJ'] + attention):.2f}x |"
        )
    note = (
        f"\nWeights {weights.upper()}. One BERT-base encoder layer, 128 tokens. Attention QK^T and PV: {attention:.1f} uJ for every design "
        "(SparTen energy per MAC). Cycles and accesses from the RTL simulators; logic power from fits to the "
        "measured CNN power runs (flow/calibrate.py). Every TTFS-lane code uses the same weight storage and "
        "weight traffic and bipolar's per-pass overhead; codes differ only in their event stream. QNN twin: "
        "bipolar's events, cycles and memory with a 4x4 MAC per op instead of the lane add "
        f"(+{TWIN_DELTA_PJ[weights]:.3f} pJ per op, measured cell energies).\n"
    )
    (root(weights) / "table.md").write_text("\n".join(lines) + note)
    (root(weights) / "table.json").write_text(
        json.dumps({"attention_uJ": attention, "rows": {NAMES[k]: v for k, v in rows.items()}}, indent=1)
    )
    return "\n".join(lines) + note
