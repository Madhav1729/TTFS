#!/bin/bash
# QAT of the two configurations the CNN hardware comparison still needs (W4 + uniform A4 for
# SparTen / linear TTFS, W4 + log 3 levels for a 2-bit log TTFS), then export of every conv layer
# (hardware/sim_runs/export_conv.py) for those and for the existing hw_w4_log7 checkpoint.
set -euo pipefail
python_path="$ANN_SNN_ROOT/envs/logquant-cu118/bin/python"
wheelhouse="$ANN_SNN_ROOT/cache/wheels/logquant-20261022"
runtime_dir=$(mktemp -d "/tmp/logquant_${USER}_${PBS_JOBID}_XXXXXX")
printf '%s\n' "$runtime_dir" > "$RUN_DIR/runtime_directory.txt"
TMPDIR="$runtime_dir" "$python_path" -m pip install --no-index --no-deps --no-compile \
    --target "$runtime_dir" "$wheelhouse/"*.whl > "$RUN_DIR/runtime_setup.log" 2>&1
export PYTHONPATH="$runtime_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
mkdir "$RUN_DIR/source"
tar -C "$ANN_SNN_ROOT/src" --exclude='__pycache__' --exclude='.venv' --exclude='results' \
    --exclude='data' --exclude='prepared' --exclude='checkpoints' -cf - logQuant hardware/sim_runs/export_conv.py \
    | tar -C "$RUN_DIR/source" -xf -
cd "$RUN_DIR/source"
sha256sum logQuant/*.py logQuant/image_models/*.py logQuant/finetuning/*.py hardware/sim_runs/export_conv.py \
    > "$RUN_DIR/source_sha256.txt"
checkpoint="$ANN_SNN_ROOT/checkpoints/vgg16-cifar100/cifar100_vgg16_bn-7d8c4031.pt"
dataset="$ANN_SNN_ROOT/data/CIFAR100/cifar-100-python-mirror.tar.gz"
manifest="$ANN_SNN_ROOT/data/finetuning/seed42/data_manifest.json"
ckdir="$ANN_SNN_ROOT/src/logQuant/checkpoints/vgg16_qat_${PBS_JOBID}"
log7="$ANN_SNN_ROOT/src/logQuant/checkpoints/vgg16_qat_2549758.hn1/hw_w4_log7/last.pt"
sha256sum "$checkpoint" "$dataset" "$manifest" "$log7" > "$RUN_DIR/input_sha256.txt"
export_one() {
    "$python_path" hardware/sim_runs/export_conv.py --checkpoint "$2" --data-archive "$dataset" \
        --output "$RUN_DIR/conv_layers/$1" --images 0 1 --device cuda
}
# the existing checkpoint first: an export bug fails the job before the training time is spent
export_one log7 "$log7"
"$python_path" -m logQuant.finetuning.train_vgg --checkpoint "$checkpoint" --data-archive "$dataset" \
    --manifest "$manifest" --device cuda --configs ${QAT_CONFIGS:-hw_w4_uniform_a4 hw_w4_log3} \
    --epochs ${QAT_EPOCHS:-30} --output-dir "$RUN_DIR/results" --checkpoint-dir "$ckdir"
export_one uniform_a4 "$ckdir/hw_w4_uniform_a4/last.pt"
export_one log3 "$ckdir/hw_w4_log3/last.pt"
