# QNN -> TTFS conversion

| Module | Converts | Check |
|---|---|---|
| `cnn.py` (+ `qcnn.py`, `ir.py`) | LSQ W4A4 VGG (BN, ReLU, 4-bit activations) -> integer weights and per-tick thresholds on the integer sum; `run_ttfs` is the integer TTFS reference | `verify(model, graph, x)`: per-layer and chained level mismatches, logits, predictions |
| `bert.py` | Otters W1A4 BERT -> polarity TTFS per quantised linear (signed sites T = 8 with a sign payload, unsigned FFN2 input T = 15) | `verify(graph, capture(model, rte_batch()))`: max output error per site |
| `verify_polarity.py` | event-level simulation of the polarity neuron against the sign-magnitude quantiser | decoded value = source level for every input |

Results (`models/vgg16h/checkpoint.pt`, 64 CIFAR-10 test images; `models/otters`, 8 RTE dev examples):
- VGG-16-half W4A4: 2 isolated level mismatches (float32 ties), 0.014 % of activations differ when chained,
  predictions identical.
- BERT W1A4: 73 sites, worst output error 1.2e-5 of the output range (float32).
- Polarity rule: all cases pass; T = 8 carries 17 levels, the same step as offset-binary at T = 16.

```
python -c "from conversion import cnn, qcnn; ..."      # see the module docstrings
python conversion/verify_polarity.py
```
