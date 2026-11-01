# Models

The hardware flow starts from problem files and activation profiles derived from three trained networks. This folder
holds the networks and the code from checkpoint to the inputs in `data/`.

| Model | Task | Accuracy | Checkpoint | Derived inputs |
|---|---|---|---|---|
| VGG-16-BN W4A4: QAT with distillation from chenyaofo `cifar100_vgg16_bn`; INT4 weights -7..7, 16 activation levels | CIFAR-100 | 71.41 % test (FP32 teacher 74.00 %) | `vgg16/checkpoint/last.pt` | `vgg16/layers/` -> `data/problems/` via `python -m flow export-problems` (36/36 byte-identical) |
| Otters BERT-base W1A4: binary weights, 4-bit TTFS activations, ReLU FFN | GLUE RTE | 67.5 % dev (our evaluation); best 75.0 % in the training log | `otters/checkpoint/pytorch_model.bin` | `data/lm/profile_otters.json` via `models/otters/profile.py`; `data/lm/bert_layer5_w4.npz` via `python -m flow quantize-otters` |
| VGG-16-half W4A4: LSQ QAT, channels halved | CIFAR-10 | 91.39 % test | `vgg16h/checkpoint.pt` | example for `conversion/cnn.py` (exactness check) |

## Checksums

| File | Size (B) | Hash |
|---|---:|---|
| `vgg16/checkpoint/last.pt` | 76572065 | sha256 `60eb2abe5ebfeac144d292427aeb07073ef6e8199ff4de5128a03d7f75c530c7` |
| `otters/checkpoint/pytorch_model.bin` | 438547359 | md5 `9e273844c752e6f6e458e9eb0f60f2e0` |
| `vgg16h/checkpoint.pt` | 14803978 | md5 `7f167b8703c4883387a063abdca54719` |
| `../data/lm/profile_otters.json` | | md5 `e904d7a6ca2b31d0aea881695b3db96f` |
| `../data/lm/bert_layer5_w4.npz` | | md5 `b942e27f2a23356bdf72012a0a8d0a8f` |

## VGG-16 (the CNN workload)

- `vgg16/checkpoint/last.pt`: the exported QAT checkpoint (epoch 30).
- `vgg16/source/`: training (`logQuant/finetuning/train_vgg.py`) and export (`hardware/sim_runs/export_conv.py`) code,
  the exact versions of the training run; `vgg16/run/`: job script, log, input hashes.
- `vgg16/layers/`: per-layer exports (weights, thresholds, direction bits, input ticks, reference sums and outputs for
  CIFAR-100 test images 0 and 1). `python -m flow export-problems` turns them into `data/problems/`, byte-identical.
- Regenerating the exports from the checkpoint needs the CIFAR-100 archive: see `vgg16/README.md`.

## Otters BERT

- `otters/checkpoint/`: `kd_joint` student of run `weight1a4-RTE-argu-a-shifting-mixprevious-cmr` (config, weights, eval log).
- `otters/code/`: the Otters reference implementation (quantised BERT with the TTFS activation quantiser).
- `otters/glue_data/RTE/`: GLUE RTE train / dev.
- `otters/profile.py`: firing-tick histogram of every quantised activation on 16 dev examples; regenerates
  `data/lm/profile_otters.json` exactly.
- `python -m flow quantize-otters`: layer-5 latent weights to 4 bits per output channel (LSQ initial step);
  regenerates `data/lm/bert_layer5_w4.npz` exactly. W1 uses their sign.

## Large files

Checkpoints (`*.pt`, `*.bin`) are not in git. Download [`checkpoints.zip`](https://drive.google.com/file/d/1ULRXTGxmAl8MLo_yqgPOb0wExdkZ89UQ/view?usp=sharing) (505 MB), unzip it inside
`hw_final/`, and check the files with the hashes above.
