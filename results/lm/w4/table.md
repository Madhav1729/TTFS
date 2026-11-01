| Design | Latency (us) | Logic (uJ) | Memory (uJ) | Linear layers (uJ) | vs TTFS bipolar | With attention (uJ) | vs TTFS bipolar |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFS bipolar | 43157 | 1072.1 | 320.6 | 1392.7 | 1.00x | 1571.1 | 1.00x |
| TTFS two-rail unipolar | 43157 | 1072.1 | 338.6 | 1410.7 | 1.01x | 1589.2 | 1.01x |
| TTFS offset unipolar | 55873 | 1393.0 | 388.7 | 1781.6 | 1.28x | 1960.1 | 1.25x |
| Bit-serial, one event per non-zero bit (same lanes) | 49226 | 1251.6 | 401.8 | 1653.5 | 1.19x | 1831.9 | 1.17x |
| Rate / count spikes (same lanes) | 72981 | 1954.4 | 578.0 | 2532.4 | 1.82x | 2710.8 | 1.73x |
| SparTen W4A4, zero-skipping QNN | 76193 | 2005.3 | 740.7 | 2746.0 | 1.97x | 2924.4 | 1.86x |
| SparTen W4A4, no zeros to skip (offset inputs) | 93618 | 2535.4 | 900.7 | 3436.1 | 2.47x | 3614.5 | 2.30x |
| QNN twin: same dataflow, MAC lanes | 43157 | 1153.8 | 320.6 | 1474.4 | 1.06x | 1652.8 | 1.05x |
Weights W4. One BERT-base encoder layer, 128 tokens. Attention QK^T and PV: 178.4 uJ for every design (SparTen energy per MAC). Cycles and accesses from the RTL simulators; logic power from fits to the measured CNN power runs (flow/calibrate.py). Every TTFS-lane code uses the same weight storage and weight traffic and bipolar's per-pass overhead; codes differ only in their event stream. QNN twin: bipolar's events, cycles and memory with a 4x4 MAC per op instead of the lane add (+0.211 pJ per op, measured cell energies).
