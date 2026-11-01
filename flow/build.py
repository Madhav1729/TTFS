"""Verilator build of the cycle simulator (SparTen cluster + TTFS tile + C harness), with SAIF tracing."""

import os
import subprocess

from . import config

VFLAGS = [
    "--cc",
    "--trace-saif",
    "--trace-max-array",
    "256",
    "--assert",
    "-Wall",
    "-Wno-fatal",
    "-Wno-DECLFILENAME",
]
CXXFLAGS = ["-std=c++17", "-O2", "-Wall", "-Wno-unused-parameter", "-Wno-sign-compare"]


def _env():
    env = dict(os.environ)
    env["PATH"] = f"{config.VERILATOR.parent}:{env['PATH']}"
    return env


def verilator_include():
    return config.VERILATOR.parent.parent / "share" / "verilator" / "include"


def _verilate(out, top, params):
    out.mkdir(parents=True, exist_ok=True)
    sources = [str(config.RTL / f) for f in config.RTL_FILES]
    cmd = [str(config.VERILATOR), *VFLAGS, "--top-module", top, *params, "--Mdir", str(out), *sources]
    with open(out / "verilate.log", "w") as log:
        subprocess.run(cmd, check=True, env=_env(), stdout=log, stderr=log)
    subprocess.run(
        ["make", "-s", "-j8", "-C", str(out), "-f", f"V{top}.mk", f"V{top}__ALL.a"],
        check=True,
        env=_env(),
        stdout=subprocess.DEVNULL,
    )
    return out / f"V{top}__ALL.a"


def sim_dir(gate, steps=config.STEPS):
    suffix = "" if steps == config.STEPS else f"_t{steps}"
    return config.BUILD / f"sim_gate{gate}{suffix}"


def build(gate, steps=config.STEPS):
    """Build the simulator; gate selects the TTFS lane (0 = plain, 1 = idle-lane clock gating), steps the window."""
    out = sim_dir(gate, steps)
    include = verilator_include()
    defines = [*config.SIM_DEFINES, *config.step_defines(steps)]
    spt = _verilate(out / "sparten", "sparten_full", [*config.SPARTEN_PARAMS, f"-GT={steps}"])
    ttfs = _verilate(out / "ttfs", "ttfs_iv", [*config.TTFS_PARAMS, f"-GT={steps}", f"-GGATE={gate}"])
    env = _env()
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            *defines,
            f"-I{config.SIM}",
            "-c",
            str(config.SIM / "harness.c"),
            "-o",
            str(out / "harness.o"),
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "c++",
            *CXXFLAGS,
            *defines,
            f"-I{config.SIM}",
            f"-I{out / 'sparten'}",
            f"-I{out / 'ttfs'}",
            f"-I{include}",
            f"-I{include}/vltstd",
            str(config.SIM / "bridge.cpp"),
            *(f"{include}/{f}" for f in ("verilated.cpp", "verilated_saif_c.cpp", "verilated_threads.cpp")),
            str(out / "harness.o"),
            str(spt),
            str(ttfs),
            "-pthread",
            "-o",
            str(out / "sim"),
        ],
        check=True,
        env=env,
        stderr=subprocess.DEVNULL,
    )
    return out / "sim"


def build_apex():
    """APEX array (16 INT4 TPPEs, T = 15) with its C++ testbench, SAIF tracing on."""
    out = config.BUILD / "apex"
    include = verilator_include()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(config.VERILATOR),
        *VFLAGS,
        "--top-module",
        "apex_top",
        *config.APEX_PARAMS,
        "--Mdir",
        str(out / "v"),
        str(config.APEX / "rtl" / "apex.sv"),
    ]
    with open(out / "verilate.log", "w") as log:
        subprocess.run(cmd, check=True, env=_env(), stdout=log, stderr=log)
    subprocess.run(
        ["make", "-s", "-j8", "-C", str(out / "v"), "-f", "Vapex_top.mk", "Vapex_top__ALL.a"],
        check=True,
        env=_env(),
        stdout=subprocess.DEVNULL,
    )
    runtime = [f"{include}/{f}" for f in ("verilated.cpp", "verilated_saif_c.cpp", "verilated_threads.cpp")]
    subprocess.run(
        [
            "c++",
            *CXXFLAGS,
            *config.APEX_DEFINES,
            f"-I{out / 'v'}",
            f"-I{include}",
            f"-I{include}/vltstd",
            str(config.APEX / "sim" / "apex_tb.cpp"),
            *runtime,
            str(out / "v" / "Vapex_top__ALL.a"),
            "-pthread",
            "-o",
            str(out / "sim"),
        ],
        check=True,
        env=_env(),
        stderr=subprocess.DEVNULL,
    )
    return out / "sim"


def build_eyeriss():
    """Eyeriss PE (INT4) with its testbench, SAIF tracing on."""
    out = config.BUILD / "eyeriss"
    out.mkdir(parents=True, exist_ok=True)
    rtl = config.ROOT / "baselines" / "eyeriss" / "rtl" / "eyeriss_pe.sv"
    tb = config.ROOT / "baselines" / "eyeriss" / "sim" / "pe_tb.cpp"
    cmd = [
        str(config.VERILATOR),
        "--cc",
        "--exe",
        "--build",
        "-j",
        "4",
        "--trace-saif",
        "--assert",
        "-Wall",
        "-Wno-fatal",
        "-Wno-DECLFILENAME",
        "-Wno-UNUSEDSIGNAL",
        "-O2",
        "-CFLAGS",
        "-DSAIF_TRACE",
        "--top-module",
        "eyeriss_pe",
        "--Mdir",
        str(out / "obj"),
        "-o",
        "pe_tb",
        str(rtl),
        str(tb),
    ]
    with open(out / "build.log", "w") as log:
        subprocess.run(cmd, check=True, env=_env(), stdout=log, stderr=log)
    (out / "pe_tb").write_bytes((out / "obj" / "pe_tb").read_bytes())
    (out / "pe_tb").chmod(0o755)
    return out / "pe_tb"
