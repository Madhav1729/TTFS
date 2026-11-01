"""Paths, workloads and design constants shared by every step."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL, SIM, SYNTH = ROOT / "rtl", ROOT / "sim", ROOT / "synth"
DATA, RESULTS, BUILD = ROOT / "data", ROOT / "results", ROOT / "build"

WORKLOADS = {"w_0.6": "~40 % (0.56)", "w_0.3": "70 % (0.30)", "w_loas": "98 % LoAS"}
LAYERS = [
    "features.3",
    "features.7",
    "features.10",
    "features.14",
    "features.17",
    "features.20",
    "features.24",
    "features.27",
    "features.30",
    "features.34",
    "features.37",
    "features.40",
]

PERIOD_NS = 5.0
STEPS = 15
LANES, UNITS = 64, 32
LANE_SAMPLES = (0, 21, 42, 63)
TPPES = 16
TPPE_SAMPLES = (0, 5, 10, 15)
UNIT_SAMPLES = (0, 20)
SAIF_TIME_SCALE = 0.5
SAMPLE_PIXELS = 32

# each source: where the table reads simulations, power reports and synthesis reports
RECORDED = DATA / "recorded"
SOURCES = {
    "recorded": {"sims": RECORDED, "power": RECORDED, "synth": RECORDED},
    "results": {"sims": RESULTS, "power": RESULTS, "synth": RECORDED},
    "resynth": {"sims": RESULTS, "power": RESULTS / "resynth", "synth": RESULTS / "resynth"},
}

TECH_UM = "0.045"
GLOBAL_BUFFER_B = 262144
PATCH_BYTES = {"sparten": 2048, "ttfs": 4096}

# SRAM-like blocks removed from the netlist power / area and priced with CACTI instead
SRAM_BLOCKS = {
    "iv_front": ["weights", "membranes", "thresholds"],
    "spf_unit": ["fbuf", "aread"],
    "spf_peri": [r"homes_\d+__outsums"],
}
TOPS = {
    "iv_lane": "ttfs_iv_lane",
    "iv_lane_gated": "ttfs_iv_lane",
    "iv_front": "ttfs_iv",
    "spf_unit": "spf_unit",
    "spf_peri": "spf_periphery",
}
SPARTEN_TABLE_CELLS = "*collector/*table_*"
TOPS.update({"apex_tppe": "apex_tppe", "apex_peri": "apex_periphery", "eyeriss_pe": "eyeriss_pe"})
SRAM_BLOCKS.update({"apex_tppe": ["dbufb", "dbufa"], "apex_peri": [], "eyeriss_pe": ["fspad"]})
SAIF_SCALES = {"eyeriss_pe": 1.0}

SAIF_PATHS = {
    "iv_lane": "ttfs_iv/ttfs_iv/g_lanes/lanes[{i}]/lane",
    "iv_front": "ttfs_iv/ttfs_iv",
    "spf_unit": "sparten_full/sparten_full/units[{i}]/pe",
    "spf_peri": "sparten_full/sparten_full",
    "apex_tppe": "TOP/apex_top/t4[{i}]/pe",
    "apex_peri": "TOP/apex_top/peri",
    "eyeriss_pe": "TOP/eyeriss_pe",
}

# activity runs on the 32-pixel samples, and full-layer runs (with input reuse) for cycles and memory counts
SIM_FLAGS = {
    "ttfs": ["--engines", "ttfs", "--wenc", "list", "--zskip", "1"],
    "sparten": ["--engines", "sparten", "--wenc", "list"],
}
FULL_SIM_FLAGS = {
    "ttfs": ["--engines", "ttfs", "--wenc", "list", "--zskip", "1"],
    "sparten": ["--wenc", "list"],
}

VERILATOR = Path.home() / "miniconda3/envs/hwsim/bin/verilator"
SIM_DEFINES = [
    "-DUNITS_CFG=32",
    "-DLANES_CFG=64",
    "-DOB_CFG=32",
    "-DCH_CFG=256",
    "-DEDEPTH_CFG=32",
    "-DNPORT_CFG=4",
    "-DSPTW_CFG=4",
    "-DSPTA_CFG=4",
    "-DWS_CFG=16",
    "-DSAIF_TRACE",
]
SPARTEN_PARAMS = ["-GUNITS=32", "-GOB=32", "-GNPORT=4", "-GW=4", "-GA=4"]
APEX_PARAMS = ["-GT=15", "-GPW=12", "-GCW=12", "-GMW=18", "-GFB=6", "-GPC=8"]
APEX_DEFINES = ["-DT_CFG=15", "-DMW_CFG=18", "-DFB_CFG=6", "-DSAIF_TRACE"]
APEX = ROOT / "baselines" / "apex"
TTFS_PARAMS = [
    "-GLANES=64",
    "-GOB=32",
    "-GCH=256",
    "-GEDEPTH=32",
    "-GWS=16",
    "-GS=4",
    "-GNB=4",
]


def tick_bits(steps):
    """Bits of a tick field (ticks 0..steps-1)."""
    return max(1, (steps - 1).bit_length())


def step_defines(steps):
    return [f"-DSTEPS_CFG={steps}", f"-DTBITS_CFG={tick_bits(steps)}"]


RTL_FILES = ["sparten.sv", "spf_periphery.sv", "ttfs_blocks.sv", "ttfs_iv.sv"]
SYNTH_PIECES = {
    "iv_lane": ("ttfs_iv_lane", "W=4,B=13,ACC=32,T=15,WS=16,GATE=0"),
    "iv_lane_gated": ("ttfs_iv_lane", "W=4,B=13,ACC=32,T=15,WS=16,GATE=1"),
    "iv_front": ("ttfs_iv", "LANES=64,CH=16,OB=2,LANES_IMPL=0,T=15"),
    "spf_unit": ("spf_unit", "BLOCK=128,W=4,A=4,ACC=32,OB=32,NF=64"),
    "spf_peri": ("spf_periphery", "UNITS=32,OB=2,A=4,T=15"),
}

# Design Compiler machine, from the environment (see dc.env.example); no host = run dc_shell locally
DC_HOST = os.environ.get("TTFS_DC_HOST", "")
DC_USER = os.environ.get("TTFS_DC_USER", "")
DC_PORT = os.environ.get("TTFS_DC_PORT", "22")
DC_KEY = os.environ.get("TTFS_DC_KEY", "")
DC_WORKDIR = os.environ.get("TTFS_DC_WORKDIR", "" if DC_HOST else str(BUILD / "dc_runs"))
DC_SETUP = os.environ.get("TTFS_DC_SETUP", "")
LIBRARY = DATA / "lib"
LIBRARY_DB = "ng45_typical.db"


def tags():
    return [f"{w}_{layer}" for w in WORKLOADS for layer in LAYERS]
