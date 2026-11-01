"""Eyeriss rows with the PE's measured power in place of the toggle-fit estimate (array cycles stay modelled)."""

import json
import subprocess
import sys

from . import config, dc

SCRIPT = config.ROOT / "baselines" / "eyeriss" / "energy16.py"
CALIBRATION = config.RECORDED / "eyeriss" / "activity.json"


def _measured_activity(reports, out):
    """Calibration file with each layer's busy dynamic power replaced by the DC power run's value."""
    act = json.loads(CALIBRATION.read_text())
    for workload, rows in act["workloads"].items():
        for row in rows:
            report = reports.power("eyeriss_pe", f"{workload}_{row['layer']}")
            row["dyn_uW"] = (
                dc.block_power(report, "eyeriss_pe", config.SRAM_BLOCKS["eyeriss_pe"]).dynamic * 1e3
            )
    out.write_text(json.dumps(act, indent=1))
    return out


def _run(reports):
    config.RESULTS.mkdir(exist_ok=True)
    activity = _measured_activity(reports, config.RESULTS / "eyeriss_activity_measured.json")
    result = config.RESULTS / "eyeriss_measured.json"
    cacti = sorted(str(p) for p in (config.DATA / "cacti").glob("*.csv"))
    synth = reports.paths["synth"] / "synth" / "eyeriss_pe"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--problems",
        str(config.DATA / "problems"),
        "--activity",
        str(activity),
        "--synth",
        str(synth),
        "--cacti",
        *cacti,
        "--batches",
        "1",
        "--json",
        str(result),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads(result.read_text())


def measured_row(reports, workload):
    summary = next(s for s in _run(reports)["summary"] if s["batch"] == 1 and s["workload"] == workload)
    return {
        "design": "Eyeriss INT4, measured PE",
        "workload": workload,
        "cycles": summary["cycles_per_img"],
        "clock_ns": summary["clock_ns"],
        "latency_us": summary["latency_us_per_img"],
        "logic_uJ": summary["logic_uJ"],
        "mem_uJ": summary["total_uJ"] - summary["logic_uJ"],
        "energy_uJ": summary["total_uJ"],
        "area_mm2": summary["area_mm2_no_glb"],
        "measured": True,
    }
