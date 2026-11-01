"""Cycle simulations: full layers (cycles, memory counts) and 32-pixel samples (switching activity)."""

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from . import config, problems

ACTIVITY = config.RESULTS / "activity"
FULL = config.RESULTS / "sims"
TRACES = {"ttfs": "ttfs_iv.saif", "sparten": "sparten_full.saif", "apex": "apex.saif", "eyeriss": "pe.saif"}


def simulator(gate):
    return config.BUILD / f"sim_gate{gate}" / "sim"


def activity_dir(gate, engine, workload, layer):
    return ACTIVITY / f"gate{gate}" / engine / workload / layer


def _run(sim, problem, out, flags):
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(sim), "--problem", str(problem), "--output", str(out), *flags]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = json.loads((out / "results.json").read_text())
    if not result["verified"]:
        raise RuntimeError(f"output check failed: {problem}")
    return result


def _activity_run(gate, engine, workload, layer):
    out = activity_dir(gate, engine, workload, layer)
    problem = problems.SAMPLED / workload / f"{layer}.txt"
    result = _run(simulator(gate), problem, out, [*config.SIM_FLAGS[engine], "--trace"])
    for engine_name, trace in TRACES.items():
        if engine_name != engine:
            (out / trace).unlink(missing_ok=True)
    return result["engines"][engine]


def _full_run(gate, engine, workload, layer):
    out = config.BUILD / "full_runs" / engine / workload / layer
    flags = ["--geometry", problems.geometry(workload, layer), *config.FULL_SIM_FLAGS[engine]]
    result = _run(simulator(gate), problems.FULL / workload / f"{layer}.txt", out, flags)
    target = FULL / workload / engine / f"{layer}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "results.json", target)
    shutil.rmtree(out)
    return result["engines"][engine]


def _each_layer(job, gate, engine, jobs):
    work = [(w, layer) for w in config.WORKLOADS for layer in config.LAYERS]
    with ThreadPoolExecutor(jobs) as pool:
        stats = pool.map(lambda wl: job(gate, engine, *wl), work)
        return {f"{w}_{layer}": s for (w, layer), s in zip(work, stats)}


def activity(gate, engine, jobs=8):
    problems.sample_all()
    return _each_layer(_activity_run, gate, engine, jobs)


def full_layers(engine, jobs=8):
    return _each_layer(_full_run, 1, engine, jobs)


def same_schedule(a, b):
    """True if two sets of runs took the same cycles and did the same arithmetic on every layer."""
    return all(a[t][k] == b[t][k] for t in a for k in ("cycles", "arithmetic"))


def _apex_run(problem, params, out, trace=None):
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(config.BUILD / "apex" / "sim"), str(problem), str(params), str(out)]
    if trace:
        cmd += ["--saif", str(trace)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = json.loads((out / "results.json").read_text())
    if result["mismatches"] or result["fiber_errors"]:
        raise RuntimeError(f"APEX output check failed: {problem}")
    return result


def _apex_activity(gate, engine, workload, layer):
    out = activity_dir(0, "apex", workload, layer)
    params = config.APEX / "params" / workload / f"{layer}.txt"
    problem = problems.SAMPLED / workload / f"{layer}.txt"
    return _apex_run(problem, params, out, out.parent / f"{layer}.saif.tmp")


def _apex_full(gate, engine, workload, layer):
    out = config.BUILD / "full_runs" / "apex" / workload / layer
    params = config.APEX / "params" / workload / f"{layer}.txt"
    result = _apex_run(problems.FULL / workload / f"{layer}.txt", params, out)
    target = FULL / workload / "apex" / f"{layer}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "results.json", target)
    shutil.rmtree(out)
    return result


def apex_activity(jobs=8):
    problems.sample_all()
    runs = _each_layer(_apex_activity, 0, "apex", jobs)
    for w in config.WORKLOADS:
        for layer in config.LAYERS:
            tmp = activity_dir(0, "apex", w, layer).parent / f"{layer}.saif.tmp"
            tmp.rename(activity_dir(0, "apex", w, layer) / TRACES["apex"])
    return runs


def apex_full_layers(jobs=8):
    return _each_layer(_apex_full, 0, "apex", jobs)


def _eyeriss_activity(gate, engine, workload, layer):
    """PE testbench on a sample of the layer: 4 image rows x 12 passes (48 channels, 192 filters)."""
    out = activity_dir(0, "eyeriss", workload, layer)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    info = problems.layer_info(workload, layer)
    converted = config.BUILD / "eyeriss_problems" / workload / f"{layer}.txt"
    problems.to_channel_minor(problems.FULL / workload / f"{layer}.txt", converted)
    cmd = [
        str(config.BUILD / "eyeriss" / "pe_tb"),
        "--layer",
        str(converted),
        "--f",
        str(info["W"]),
        "--p",
        "16",
        "--q",
        "4",
        "--rows",
        str(min(4, info["H"])),
        "--passes",
        "12",
        "--pin",
        "1",
        "--idle",
        "0",
        "--vcd",
        str(out / TRACES["eyeriss"]),
        "--json",
        str(out / "results.json"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = json.loads((out / "results.json").read_text())
    if result.get("errors", 0):
        raise RuntimeError(f"Eyeriss PE check failed: {workload} {layer}")
    return result


def eyeriss_activity(jobs=8):
    return _each_layer(_eyeriss_activity, 0, "eyeriss", jobs)
