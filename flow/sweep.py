"""Weight-density sweep of TTFS-64 (gated) and SparTen: energy per image split by cause, from fitted power models."""

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from . import calibrate, config, memory, problems
from .cacti import Cacti

DENSITIES = ["natural", "0.6", "0.5", "0.3", "0.1", "0.05", "loas"]
OUT = config.RESULTS / "sweep"
E_OP_PJ = {"sparten": 0.301, "ttfs": 0.090}  # arithmetic cells per op, from the measured DC reports
CLOCK_NS = {"sparten": 5.26, "ttfs": 5.36}
FLAGS = {"ttfs": ["--engines", "ttfs", "--wenc", "list", "--zskip", "1"], "sparten": ["--engines", "sparten", "--wenc", "list"]}
LOCAL = {"sparten": ("fbuf", "ibuf", "sums", "patch", "table"), "ttfs": ("wbanks", "membranes", "patch", "sorter", "thresholds")}


def problem_dir(density):
    recorded = config.DATA / "problems" / f"w_{density}"
    return recorded if recorded.is_dir() else problems.SAMPLED.parent / "problems" / f"w_{density}"


def _run(job):
    density, engine, layer = job
    out = OUT / "sims" / f"w_{density}" / engine / layer
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdir = problem_dir(density)
    info = next(r for r in json.loads((pdir / "layers.json").read_text())["layers"] if r["layer"] == layer)
    cmd = [str(config.BUILD / "sim_gate1" / "sim"), "--problem", str(pdir / f"{layer}.txt"), "--output", str(out),
           "--geometry", info["geometry"], *FLAGS[engine]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = json.loads((out / "results.json").read_text())
    if not result["verified"]:
        raise RuntimeError(f"output check failed: {job}")
    return job, result["engines"][engine]


def simulate(jobs=8):
    work = [(d, e, layer) for d in DENSITIES for e in ("sparten", "ttfs") for layer in config.LAYERS]
    with ThreadPoolExecutor(jobs) as pool:
        return dict(pool.map(_run, work))


def load():
    """Stats of earlier sweep simulations."""
    return {(d, e, layer): json.loads((OUT / "sims" / f"w_{d}" / e / layer / "results.json").read_text())["engines"][e]
            for d in DENSITIES for e in ("sparten", "ttfs") for layer in config.LAYERS}


def _power(models, engine, s):
    if engine == "ttfs":
        return calibrate.power(models, "lane", s).scale(config.LANES) + calibrate.power(models, "front", s)
    return calibrate.power(models, "unit", s).scale(config.UNITS) + calibrate.power(models, "peri", s)


def _memory(engine, s, cacti):
    if engine == "ttfs":
        return memory.ttfs(s, cacti)
    e = memory.sparten(s, cacti)
    e["table"] = memory.sparten_table(s, cacti)
    return e


def breakdown(stats):
    """{density: {engine: energy parts in uJ, cycles, ops}} for every density."""
    cacti, models = Cacti(), calibrate.fit()
    out = {}
    for d in DENSITIES:
        density = json.loads((problem_dir(d) / "layers.json").read_text())["layers"]
        w = sum(r["weight_density"] * r["n"] * r["m"] for r in density) / sum(r["n"] * r["m"] for r in density)
        out[d] = {"weight_density": w}
        for engine in ("sparten", "ttfs"):
            parts = dict(arithmetic=0.0, other_logic=0.0, local_sram=0.0, global_buffer=0.0)
            cycles = ops = 0
            layers = {}
            for layer in config.LAYERS:
                s = stats[(d, engine, layer)]
                logic = _power(models, engine, s).energy_pJ(s["cycles"], CLOCK_NS[engine], config.PERIOD_NS)
                arith = s["arithmetic"] * E_OP_PJ[engine]
                mem = _memory(engine, s, cacti)
                parts["arithmetic"] += arith
                parts["other_logic"] += logic - arith
                parts["local_sram"] += sum(v for k, v in mem.items() if k in LOCAL[engine])
                parts["global_buffer"] += sum(v for k, v in mem.items() if k.startswith("glb"))
                cycles += s["cycles"]
                ops += s["arithmetic"]
                layers[layer] = {"ops": s["arithmetic"], "w_beats": s["w_beats"], "cycles": s["cycles"]}
            out[d][engine] = {**{k: v * 1e-6 for k, v in parts.items()}, "cycles": cycles, "ops": ops,
                              "layers": layers}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(out, indent=1))
    return out
