"""Linear power models fitted on the measured CNN power runs: dynamic = a + b x activity rate, leakage = mean."""

import json

import numpy as np

from . import config, dc, designs, simulate

ACTIVITY_RATES = {
    "lane": lambda s: s["arithmetic"] / (s["cycles"] * config.LANES),
    "front": lambda s: s["event_pushes"] / s["cycles"],
    "unit": lambda s: s["arithmetic"] / (s["cycles"] * config.UNITS),
    "peri": lambda s: s["arithmetic"] / (s["cycles"] * config.UNITS),
}


def _points():
    reports = designs.Reports(config.SOURCES["recorded"])
    pts = {k: [] for k in ACTIVITY_RATES}
    for w in config.WORKLOADS:
        for layer in config.LAYERS:
            tag = f"{w}_{layer}"
            t = json.loads((simulate.activity_dir(1, "ttfs", w, layer) / "results.json").read_text())[
                "engines"
            ]["ttfs"]
            s = json.loads((simulate.activity_dir(0, "sparten", w, layer) / "results.json").read_text())
            s = s["engines"]["sparten"]
            lane = reports.mean_block("iv_lane_gated", tag, config.LANE_SAMPLES)
            front = dc.block_power(reports.power("iv_front", tag), "ttfs_iv", config.SRAM_BLOCKS["iv_front"])
            unit = reports.mean_block("spf_unit", tag, config.UNIT_SAMPLES)
            peri = dc.block_power(
                reports.power("spf_peri", tag), "spf_periphery", config.SRAM_BLOCKS["spf_peri"]
            )
            peri = peri - dc.cell_totals(reports.power("spf_table", tag))
            for key, stats, p in (
                ("lane", t, lane),
                ("front", t, front),
                ("unit", s, unit),
                ("peri", s, peri),
            ):
                pts[key].append((ACTIVITY_RATES[key](stats), p.dynamic, p.leakage))
    return pts


def fit():
    models = {}
    for key, pts in _points().items():
        x, dyn, leak = (np.array(v) for v in zip(*pts))
        b, a = np.polyfit(x, dyn, 1)
        pred = a + b * x
        r2 = 1 - ((dyn - pred) ** 2).sum() / ((dyn - dyn.mean()) ** 2).sum()
        models[key] = {
            "a_mW": a,
            "b_mW": b,
            "leak_mW": leak.mean(),
            "r2": r2,
            "rate_range": [x.min(), x.max()],
        }
    return models


def power(models, key, stats):
    m = models[key]
    return dc.Power(m["a_mW"] + m["b_mW"] * ACTIVITY_RATES[key](stats), m["leak_mW"])
