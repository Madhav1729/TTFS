# TTFS-64 vs SparTen, APEX and Eyeriss: hardware comparison

TTFS-64 tile (S/U lanes, idle-lane clock gating) against SparTen W4/A4, APEX (rate-coded SNN, T = 15) and
Eyeriss INT4 on VGG-16 on CIFAR-100 (12 conv layers, 16 activation levels, one image). NanGate 45 nm, CACTI 7 (45 nm LSTP);
logic power from Design Compiler with SAIF switching activity. Plus BERT-base encoder layers (Otters) for the
spike-code comparison (bipolar, two-rail, offset unipolar, bit-serial, rate) at W1A4 and W4A4.

## Layout

| Path | Contents |
|---|---|
| `rtl/` | `ttfs_iv.sv` (tile; lane `GATE`, window `T`), `ttfs_blocks.sv`, `sparten.sv`, `spf_periphery.sv` |
| `sim/` | cycle-level C harness and Verilator bridge; every output is checked exactly |
| `synth/` | DC scripts: `synth.tcl`, `power.tcl`, `cell_power.tcl` |
| `baselines/apex/`, `baselines/eyeriss/` | APEX RTL + testbench + PASC parameters; Eyeriss PE RTL, testbench, array model |
| `conversion/` | QNN -> TTFS conversion (CNN, BERT polarity) with exactness checks |
| `models/` | the trained networks (VGG-16 W4A4 CIFAR-100, Otters BERT W1A4, VGG-16-half W4A4), training / export code, provenance |
| `flow/` | Python package for every step (`python -m flow -h`); `config.py` holds all constants |
| `data/problems/` | full-layer VGG-16 problem files (weights, thresholds, input levels) |
| `data/netlists/` | the synthesized netlists (TTFS lane plain / gated, front end, SparTen unit / periphery, APEX TPPE / periphery, Eyeriss PE) |
| `data/lib/`, `data/cacti/` | NanGate45 typical library; CACTI SRAM energies and areas |
| `data/lm/` | Otters activation profile and the real layer-5 BERT weights, 4-bit |
| `data/recorded/` | simulations, synthesis and power reports behind `results/final.md` |
| `data/baselines.json` | earlier Eyeriss and APEX estimates (shown with `*`) |
| `results/` | everything the flow writes: `final.md`, `lm/w1/table.md`, `lm/w4/table.md` |

## Setup

```
unzip checkpoints.zip                 # from the Drive link below, inside hw_final/
```

Python 3.11+ with numpy (torch for the model steps), Verilator 5.052 (path in `flow/config.py`).

## Reproduce without a Synopsys licence

Builds, simulations, SAIF, energy accounting, BERT and conversion use open tools only. The tables are rebuilt from
the Design Compiler reports in `data/recorded/`.

```
make build                            # Verilator builds: TTFS / SparTen (both lanes, T = 15 and 8), APEX, Eyeriss
make simulate                         # full layers + 32-pixel activity runs (SAIF), APEX, Eyeriss
python -m flow check-gating           # gated lane keeps every cycle and add
make table                            # results/final.md from data/recorded
python -m flow lm --weights w1|w4     # BERT layer: spike codes and QNN baselines
python conversion/verify_polarity.py  # polarity TTFS decodes every level exactly
```

## Regenerate the Design Compiler reports

Needs Design Compiler (U-2022.12 used here) and its licence, on this machine or on a server reachable by ssh with
key-based login (the flow never asks for a password). Fill in the connection settings and load them:

```
cp dc.env.example dc.env              # dc.env is git-ignored
# edit: TTFS_DC_HOST, TTFS_DC_USER, TTFS_DC_PORT, TTFS_DC_KEY, TTFS_DC_WORKDIR, TTFS_DC_SETUP
source dc.env
```

| Variable | Meaning |
|---|---|
| `TTFS_DC_HOST` | ssh host or alias of the DC machine; empty = run `dc_shell` locally |
| `TTFS_DC_USER` | ssh user name (empty = from `~/.ssh/config`) |
| `TTFS_DC_PORT` | ssh port (default 22) |
| `TTFS_DC_KEY` | private key file (empty = `ssh-agent` or `~/.ssh/config`) |
| `TTFS_DC_WORKDIR` | writable directory on the DC machine for run folders (required for a remote host) |
| `TTFS_DC_SETUP` | command that puts `dc_shell` on PATH, e.g. `source /tools/synopsys/setup.sh` |

Then:

```
python -m flow power                  # power-only runs on the netlists in data/netlists (all pieces)
python -m flow fetch                  # repeat until every run has finished -> results/
python -m flow check-reports          # regenerated power reports vs data/recorded
python -m flow compare                # table from the new reports vs results/final.json
python -m flow synth                  # optional: resynthesize the TTFS / SparTen netlists from rtl/
```

## Models

Checkpoints: [`checkpoints.zip`](https://drive.google.com/file/d/1ULRXTGxmAl8MLo_yqgPOb0wExdkZ89UQ/view?usp=sharing) on Google Drive (unzip inside `hw_final/`).
`models/README.md`: the trained networks (VGG-16 W4A4, Otters BERT W1A4), accuracy, checksums, and the
scripts from checkpoint to the inputs in `data/` (`export-problems`, `quantize-otters`, `models/otters/profile.py`).

## Credits

`CREDITS.md`: models, datasets, baseline papers, libraries and tools used, with their sources and terms.
