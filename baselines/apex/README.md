# APEX baseline (LoAS TPPE array + PASC-IF), T = 15

- 16 INT4 TPPEs on one B broadcast bus, rate code with T = 15 timesteps (16 activation levels), global cache in
  16 banks of 16 KB.
- Accumulators: 12-bit pseudo and correction (the workload needs 11 bits), 18-bit PASC membrane, 6 fractional bits.
- PASC stages are 8-cycle multicycle paths at 5 ns (`synth/apex_mcp.tcl`).
- Data-A words are packed back to back (15 bits each), two contexts so the next chunk's pass overlaps the
  corrections.

| Path | Contents |
|---|---|
| `rtl/apex.sv` | TPPE, PASC-IF, periphery (compressor, crossbars), `apex_top` |
| `sim/apex_tb.cpp` | cycle testbench on the problem files; checks every output and fiber exactly |
| `sim/apex_prep.c` | derives per-layer PASC parameters from the thresholds |
| `params/<workload>/<layer>.txt` | PASC parameters used by the testbench |
| `synth/` | multicycle constraints and extra timing report for synthesis |

Build, simulation, power and energy are driven by the flow (`build_apex`, `simulate.apex_*`, power pieces
`apex_tppe` / `apex_peri`, `designs.APEX`).
