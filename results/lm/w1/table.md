| Design | Latency (us) | Logic (uJ) | Memory (uJ) | Linear layers (uJ) | vs TTFS bipolar | With attention (uJ) | vs TTFS bipolar |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFS bipolar | 54943 | 1500.9 | 351.0 | 1851.9 | 1.00x | 2020.3 | 1.00x |
| TTFS two-rail unipolar | 54943 | 1500.9 | 369.0 | 1869.9 | 1.01x | 2038.3 | 1.01x |
| TTFS offset unipolar | 71328 | 1956.7 | 419.0 | 2375.7 | 1.28x | 2544.1 | 1.26x |
| Bit-serial, one event per non-zero bit (same lanes) | 64084 | 1771.8 | 432.2 | 2204.0 | 1.19x | 2372.5 | 1.17x |
| Rate / count spikes (same lanes) | 99569 | 2823.4 | 608.4 | 3431.8 | 1.85x | 3600.3 | 1.78x |
| SparTen W4A4, zero-skipping QNN | 103396 | 2848.5 | 1009.9 | 3858.4 | 2.08x | 4026.8 | 1.99x |
| SparTen W4A4, no zeros to skip (offset inputs) | 132478 | 3686.7 | 1243.5 | 4930.2 | 2.66x | 5098.7 | 2.52x |
| QNN twin: same dataflow, MAC lanes | 54943 | 1500.9 | 351.0 | 1851.9 | 1.00x | 2020.3 | 1.00x |
Weights W1. One BERT-base encoder layer, 128 tokens. Attention QK^T and PV: 168.4 uJ for every design (SparTen energy per MAC). Cycles and accesses from the RTL simulators; logic power from fits to the measured CNN power runs (flow/calibrate.py). Every TTFS-lane code uses the same weight storage and weight traffic and bipolar's per-pass overhead; codes differ only in their event stream. QNN twin: bipolar's events, cycles and memory with a 4x4 MAC per op instead of the lane add (+0.000 pJ per op, measured cell energies).
