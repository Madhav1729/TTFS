"""Energy, latency and area of every design per workload, and the final comparison table."""

import json

from . import config, designs, eyeriss
from .cacti import Cacti


def layer_stats(design, workload, layer):
    path = design.reports.sims(workload, design.engine, layer)
    return design.stats(json.loads(path.read_text()))


def evaluate(design, workload, cacti):
    clock = design.clock_ns()
    cycles = logic = mem = 0.0
    for layer in config.LAYERS:
        s = layer_stats(design, workload, layer)
        cycles += s["cycles"]
        logic += design.power(f"{workload}_{layer}").energy_pJ(s["cycles"], clock, config.PERIOD_NS)
        mem += sum(design.memory_pJ(s, cacti).values())
    return {
        "design": design.name,
        "workload": workload,
        "cycles": cycles,
        "clock_ns": clock,
        "latency_us": cycles * clock * 1e-3,
        "logic_uJ": logic * 1e-6,
        "mem_uJ": mem * 1e-6,
        "energy_uJ": (logic + mem) * 1e-6,
        "area_mm2": design.area_mm2(cacti),
        "measured": True,
    }


def baselines(workload):
    rows = json.loads((config.DATA / "baselines.json").read_text())["rows"]
    return [dict(r, measured=False) for r in rows if r["workload"] == workload]


def build_rows(source):
    cacti = Cacti()
    rows = []
    for w in config.WORKLOADS:
        rows += [evaluate(d, w, cacti) for d in designs.all_designs(source)]
        rows += measured_baselines(source, w, cacti)
        rows += baselines(w)
    return rows


def measured_baselines(source, workload, cacti):
    """APEX and Eyeriss rows with measured logic power, when this source has their power reports."""
    reports = designs.Reports(config.SOURCES[source])
    rows = []
    if (reports.paths["power"] / "power" / "apex_peri").is_dir():
        rows.append(evaluate(designs.APEX(reports), workload, cacti))
    if (reports.paths["power"] / "power" / "eyeriss_pe").is_dir():
        rows.append(eyeriss.measured_row(reports, workload))
    return rows


def compare(rows, reference, tolerance=1e-9):
    """(workload, design, key, new, recorded) for numbers differing by more than a relative tolerance."""
    keys = ("cycles", "latency_us", "logic_uJ", "mem_uJ", "energy_uJ", "area_mm2")
    recorded = {(r["workload"], r["design"]): r for r in reference}
    diffs = []
    for a in rows:
        b = recorded.get((a["workload"], a["design"]))
        if b is None:
            continue
        for k in keys:
            if abs(a[k] - b[k]) > tolerance * max(1.0, abs(b[k])):
                diffs.append((a["workload"], a["design"], k, a[k], b[k]))
    return diffs


def gains(row, ref):
    energy = row["energy_uJ"] / ref["energy_uJ"]
    latency = row["latency_us"] / ref["latency_us"]
    return energy, latency, energy * latency


def markdown(rows, reference="TTFS-64 S/U, gated lanes"):
    head = [
        "| Sparsity | Design | Latency (ms) | Logic (uJ) | Memory (uJ) | Energy (uJ) | Area (mm2) | "
        "TTFS gain: energy | latency | EDP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = []
    for w, label in config.WORKLOADS.items():
        group = [r for r in rows if r["workload"] == w]
        ref = next(r for r in group if r["design"] == reference)
        for r in group:
            e, l, edp = gains(r, ref)
            mark = "" if r["measured"] else " *"
            cells = ["-", "-", "-"] if r is ref else [f"{e:.2f}x", f"{l:.2f}x", f"{edp:.2f}x"]
            lines.append(
                f"| {label} | {r['design']}{mark} | {r['latency_us'] / 1e3:.2f} | {r['logic_uJ']:.1f} | "
                f"{r['mem_uJ']:.1f} | {r['energy_uJ']:.1f} | {r['area_mm2']:.3f} | "
                + " | ".join(cells)
                + " |"
            )
    note = "\n* logic power estimated (not a power run); all other rows measured with SAIF activity.\n"
    return "\n".join(head + lines) + note


def write(rows, stem="final"):
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / f"{stem}.json").write_text(json.dumps(rows, indent=1) + "\n")
    table = markdown(rows)
    (config.RESULTS / f"{stem}.md").write_text(table)
    return table
