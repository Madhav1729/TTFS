# VGG-16 W4A4 checkpoint, per-layer exports and test accuracy

Every problem file in `data/problems/` (`w_0.6`, `w_0.3`, `w_loas`) is made from these exports; `python -m flow export-problems --density 0.6|0.3|loas` regenerates all 36 byte for byte.

**The dataset is CIFAR-100, not CIFAR-10.** The network is chenyaofo's pretrained `cifar100_vgg16_bn`, used as the FP32 teacher and as the starting point for the student.

## Accuracy (CIFAR-100 test set, 10,000 images)

| Model | Test accuracy |
|---|---:|
| FP32 teacher (`cifar100_vgg16_bn-7d8c4031.pt`) | 74.00 % |
| W4/A4 after PTQ, before QAT | 60.68 % |
| **W4/A4 after QAT, final epoch 30 (`checkpoint/last.pt`, the one exported)** | **71.41 %** |
| W4/A4 after QAT, best-dev epoch 27 | 71.47 % |

Training setup:
- QAT with distillation (alpha 0.5, temperature 2), 30 epochs, lr 5e-4, batch 128, seed 42.
- Split: 45,000 train images and 5,000 dev images held out from the training set. The dev images were also seen by the pretrained teacher, so dev accuracy (99 %) is not a generalisation measure.
- Weights: INT4 uniform, -7..7.
- Activations: 4-bit uniform. These are the 16 TTFS levels, T = 15 (tick t means value 15 - t; tick 15 means no spike).

Sources: `run/results/summary.md`, `hw_w4_uniform_a4.json` and `stdout.log`. The line "checkpoint W4 test accuracy 0.7141" in `stdout.log` is the export script re-evaluating this checkpoint.

## Contents

- `checkpoint/last.pt`: the QAT export from `logQuant.finetuning.train_vgg`. sha256 `60eb2abe5ebfeac144d292427aeb07073ef6e8199ff4de5128a03d7f75c530c7`, which matches `last_sha256` in `hw_w4_uniform_a4.json`.
- `source/hardware/sim_runs/export_conv.py`: writes the per-layer exports (`meta.json`, `weights.bin`, `thresholds.bin`, `down.bin`, `img<k>_ticks.bin`, `img<k>_sums.bin`, `img<k>_out.bin`). Its docstring gives the file formats and the code.
- `source/logQuant/`: the modules that `export_conv.py` and `train_vgg.py` import. These are the exact files the aqua run used: they are copied from the run's `source/` directory and their hashes match `run/source_sha256.txt`.
- `source/make_problems.py`: the original exporter (same logic as `flow/export.py`).
  - Usage: `make_problems.py layers OUT --img 0 --density 0.6|0.3|loas`.
  - It keeps the largest-magnitude fraction of the non-zero INT4 weights. This is a hardware workload only; the pruned network was not retrained.
- `layers/`: the exports themselves, for images 0 and 1 of the CIFAR-100 test set (labels 49 and 33). They are byte-identical to the aqua run's `conv_layers/uniform_a4` (121 files compared). Every conv layer except the first (`features.0`, whose input is the signed image) is exported. `model_agreement` in each `meta.json` is 1.0.
- `run/`: the PBS job (`job.pbs`, `task.sh`), `stdout.log`, `results/`, `source_sha256.txt` and `input_sha256.txt` (hashes of the teacher, the dataset archive and the split manifest).

## Reproduce (on a GPU node with PyTorch)

```
export PYTHONPATH=source
python -m logQuant.finetuning.train_vgg --checkpoint cifar100_vgg16_bn-7d8c4031.pt \
    --data-archive cifar-100-python.tar.gz --manifest data_manifest.json --device cuda \
    --configs hw_w4_uniform_a4 --epochs 30 --output-dir results --checkpoint-dir ckpt
python source/hardware/sim_runs/export_conv.py --checkpoint checkpoint/last.pt \
    --data-archive cifar-100-python.tar.gz --output layers --images 0 1 --device cuda
```

- `task.sh` is the exact job script.
- The teacher URL and the dataset URL are in `source/logQuant/image_models/data.py`.
- Remote originals:
  - aqua `/lfs/usrhome/btech/ee23b103/scratch/ANN-SNN/runs/cnn_export_2550934.hn1/`
  - checkpoint: `.../src/logQuant/checkpoints/vgg16_qat_2550934.hn1/hw_w4_uniform_a4/last.pt`
