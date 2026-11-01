"""Design Compiler jobs: SAIF extraction, staging with the folder's own netlists, launch and retrieval."""

import shutil
import tarfile
import tempfile
from pathlib import Path

from . import config, dchost, saif, simulate

# power piece -> (SAIF source piece, netlist, simulator engine, gate of the activity run)
PIECES = {
    "iv_lane": ("iv_lane", "iv_lane", "ttfs", 0),
    "iv_lane_gated": ("iv_lane", "iv_lane_gated", "ttfs", 1),
    "iv_front": ("iv_front", "iv_front", "ttfs", 0),
    "spf_unit": ("spf_unit", "spf_unit", "sparten", 0),
    "spf_peri": ("spf_peri", "spf_peri", "sparten", 0),
    "spf_table": ("spf_peri", "spf_peri", "sparten", 0),
    "apex_tppe": ("apex_tppe", "apex_tppe", "apex", 0),
    "apex_peri": ("apex_peri", "apex_peri", "apex", 0),
    "eyeriss_pe": ("eyeriss_pe", "eyeriss_pe", "eyeriss", 0),
}
SAMPLES = {"iv_lane": config.LANE_SAMPLES, "spf_unit": config.UNIT_SAMPLES, "apex_tppe": config.TPPE_SAMPLES}
REMOTES = config.RESULTS / "dc_runs.txt"


def make_saifs(piece, out):
    source, _, engine, gate = PIECES[piece]
    top = config.TOPS[source]
    for w in config.WORKLOADS:
        for layer in config.LAYERS:
            trace = simulate.activity_dir(gate, engine, w, layer) / simulate.TRACES[engine]
            for i in SAMPLES.get(source, [None]):
                name = f"{w}_{layer}" if i is None else f"{w}_{layer}_u{i}"
                path = config.SAIF_PATHS[source].format(i=i)
                scale = config.SAIF_SCALES.get(source, config.SAIF_TIME_SCALE)
                saif.extract(trace, out / f"{name}.saif", path, top, scale)


def _script(lines):
    setup = [config.DC_SETUP] if config.DC_SETUP else []
    head = ["set -e", *setup, f"export LIB=$PWD/lib/{config.LIBRARY_DB}"]
    return "\n".join([*head, *lines, "touch done"]) + "\n"


def _launch(label, bundle, lines):
    shutil.copytree(config.LIBRARY, bundle / "lib")
    remote = dchost.new_run_dir(label)
    (bundle / "run.sh").write_text(_script(lines))
    archive = bundle.parent / f"{label}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=".")
    dchost.push(archive, f"{remote}/bundle.tgz")
    archive.unlink()
    dchost.run(f"cd {remote} && tar xzf bundle.tgz && rm bundle.tgz")
    dchost.launch(remote)
    with open(REMOTES, "a") as log:
        log.write(f"{label} {remote}\n")
    return remote


def _netlist_dir(name, netlists):
    if netlists == "resynth":
        return config.RESULTS / "resynth" / "netlists" / name
    return config.DATA / "netlists" / name


def power(piece, netlists="data"):
    """Power-only run of one piece on its netlist (data/netlists or the resynthesized one)."""
    source, netlist, _, _ = PIECES[piece]
    top = config.TOPS[source]
    cells = f"CELLS='{config.SPARTEN_TABLE_CELLS}' " if piece == "spf_table" else ""
    tcl = "cell_power.tcl" if piece == "spf_table" else "power.tcl"
    saifs = 'SAIFS=$(for f in $PWD/saif/*.saif; do printf "%s,%s " $(basename $f .saif) $f; done)'
    run = f'TOP={top} OUT=reports {cells}SAIFS="$SAIFS" dc_shell -f synth/{tcl} > power.log 2>&1'
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        (bundle / "saif").mkdir(parents=True)
        make_saifs(piece, bundle / "saif")
        shutil.copytree(config.SYNTH, bundle / "synth")
        shutil.copytree(_netlist_dir(netlist, netlists), bundle / "netlist")
        prefix = "power_resynth" if netlists == "resynth" else "power"
        return _launch(f"{prefix}_{piece}", bundle, [saifs, run])


def synth(piece):
    """Synthesize one piece from rtl/ with the parameters in config.SYNTH_PIECES."""
    top, params = config.SYNTH_PIECES[piece]
    sources = " ".join(f"rtl/{f}" for f in config.RTL_FILES)
    run = (
        f"TOP={top} PARAMS='{params}' SRCS='{sources}' PERIOD={config.PERIOD_NS} "
        "dc_shell -f synth/synth.tcl > synthesis.log 2>&1"
    )
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(config.RTL, bundle / "rtl")
        shutil.copytree(config.SYNTH, bundle / "synth")
        return _launch(f"synth_{piece}", bundle, [run])


def launched():
    """{label: remote dir} of the runs started from this folder (latest per label)."""
    runs = {}
    if REMOTES.exists():
        for line in REMOTES.read_text().splitlines():
            label, remote = line.split()
            runs[label] = remote
    return runs


def fetch(label):
    """Copy a finished run's outputs: power reports, or a synthesis's reports and netlist."""
    remote = launched()[label]
    if not dchost.is_done(remote):
        raise RuntimeError(f"{label} ({remote}) has not finished")
    kind, piece = label.split("_", 1)
    if kind == "synth":
        base = config.RESULTS / "resynth"
        dchost.pull_tree(f"{remote}/reports", base / "synth" / piece)
        dchost.pull_tree(f"{remote}/netlist", base / "netlists" / piece)
        return base / "synth" / piece
    if piece.startswith("resynth_"):
        target = config.RESULTS / "resynth" / "power" / piece.removeprefix("resynth_")
    else:
        target = config.RESULTS / "power" / piece
    dchost.pull_tree(f"{remote}/reports", target)
    return target
