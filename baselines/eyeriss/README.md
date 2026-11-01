# Eyeriss v1 baseline, INT4

- Only the processing element is RTL (`rtl/eyeriss_pe.sv`: INT4 weights and activations, 20-bit psums, filter
  scratchpad 224 x 4 b, ifmap and psum scratchpads as registers, zero-ifmap gating).
- Array cycles, NoC and GLB traffic come from the row-stationary model (`model.py`), 12 x 14 PEs, 108 kB GLB.
- `energy16.py` combines the model with the PE power; the flow feeds it the measured PE power per layer
  (`flow/eyeriss.py`). `activity16.py` and `vcd_toggles.py` produced the earlier toggle-fit estimate.

| Path | Contents |
|---|---|
| `rtl/eyeriss_pe.sv` | PE |
| `sim/pe_tb.cpp` | PE testbench on a sample of each layer (4 rows x 12 passes); SAIF with `-DSAIF_TRACE` |
| `model.py` | row-stationary mapping search and pass model |
| `energy16.py` | energy per image: PE logic, scratchpads, GLB, global buffer, NoC (+2 %) |
