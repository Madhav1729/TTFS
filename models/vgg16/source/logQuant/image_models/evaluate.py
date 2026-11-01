"""CIFAR-100 VGG16-BN: FP32, INT16 weights, and uniform/log 4/6/8-bit activations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from ..evaluate import metrics, save_report
from ..weights import load_weights, quantize_weights
from .data import (CHECKPOINT_SHA256, CHECKPOINT_URL, DATA_MD5, DATA_URL, MEAN, STD,
                   MODEL_COMMIT, PREPROCESS_COMMIT, digest, download, load_split, preprocess)
from .model import VGG16BN, instrument_image


@torch.inference_mode()
def predict(model, images, batch_size):
    outputs = []
    for batch in images.split(batch_size):
        logits = model(batch).float().cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite classifier output")
        outputs.append(logits)
    return torch.cat(outputs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="logQuant/image_models/.cache/cifar100_vgg16_bn-7d8c4031.pt")
    p.add_argument("--data-archive", default="logQuant/image_models/.cache/cifar-100-python.tar.gz")
    p.add_argument("--dataset-download-url", default=DATA_URL, help="Provenance only; archive MD5 is always checked")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output-dir", default="logQuant/image_models/results/vgg16_cifar100")
    p.add_argument("--bits", type=int, nargs="+", choices=range(2, 17), default=[4, 6, 8])
    p.add_argument("--schemes", nargs="+", choices=["uniform", "log"], default=["uniform", "log"])
    p.add_argument("--weight-bits", type=int, choices=[8, 16, 32], default=16)
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--calibration-sampling", choices=["random_extrema", "legacy_linspace"],
                   default="random_extrema")
    p.add_argument("--calibration-batch-size", type=int, default=8)
    p.add_argument("--max-eval-samples", type=int, default=0, help="0 evaluates all 10,000 test images")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--collect-stats", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    out = Path(args.output_dir)
    if any((out / name).exists() for name in ("results.json", "summary.csv", "weights.pt")):
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    if min(args.calibration_samples, args.calibration_batch_size, args.batch_size, args.threads) < 1 or args.max_eval_samples < 0:
        raise ValueError("Counts must be positive; max-eval-samples must be nonnegative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.download:
        download(CHECKPOINT_URL, args.checkpoint)
        download(DATA_URL, args.data_archive)
    if digest(args.data_archive, "md5") != DATA_MD5:
        raise ValueError("CIFAR-100 archive MD5 differs from the official dataset")
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 differs from the selected pretrained VGG16-BN")
    model = VGG16BN().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(args.device)
    train, _ = load_split(args.data_archive, "train")
    ids = list(range(len(train)))
    random.Random(args.seed).shuffle(ids)
    calibration_ids = ids[:args.calibration_samples]
    calibration = preprocess(train[calibration_ids], args.device)
    del train
    images, labels_np = load_split(args.data_archive, "test")
    count = min(args.max_eval_samples, len(images)) if args.max_eval_samples else len(images)
    evaluation = preprocess(images[:count], args.device)
    labels = torch.from_numpy(labels_np[:count].copy())
    evaluation_hash = hashlib.sha256(images[:count].tobytes() + labels_np[:count].tobytes()).hexdigest()
    del images
    print(f"VGG16-BN, CIFAR-100: {count} test images; {len(calibration_ids)} train calibration images", flush=True)
    start = time.perf_counter()
    checkpoint_logits = predict(model, evaluation, args.batch_size)
    checkpoint_seconds = time.perf_counter() - start
    print(f"Original accuracy: {float((checkpoint_logits.argmax(-1) == labels).float().mean()):.4f}", flush=True)
    weight_logits, weight_seconds = checkpoint_logits, checkpoint_seconds
    weight_metadata = dict(bits=args.weight_bits, scheme="none", exported=False)
    if args.weight_bits != 32:
        bundle = quantize_weights(model, args.weight_bits)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(dict(bundle, metadata=dict(architecture="cifar100_vgg16_bn",
                   source_url=CHECKPOINT_URL, source_sha256=checkpoint_hash, mean=MEAN, std=STD)), out / "weights.pt")
        weight_metadata.update(scheme="symmetric_per_output_channel", exported=True,
                               checkpoint="weights.pt", matrices=len(bundle["matrices"]))
        del bundle
        start = time.perf_counter()
        weight_logits = predict(model, evaluation, args.batch_size)
        weight_seconds = time.perf_counter() - start
        load_weights(model, out / "weights.pt")
        torch.testing.assert_close(predict(model, evaluation[:args.batch_size], args.batch_size),
                                   weight_logits[:args.batch_size], atol=0, rtol=0)
        weight_metadata["reload_logits_verified"] = True
        print(f"INT{args.weight_bits} weight-only accuracy: "
              f"{float((weight_logits.argmax(-1) == labels).float().mean()):.4f}", flush=True)
    quantizers = instrument_image(model)
    reference_logits = predict(model, evaluation, args.batch_size)
    torch.testing.assert_close(reference_logits, weight_logits, atol=2e-5, rtol=2e-5)
    for q in quantizers.values():
        q.sampling = args.calibration_sampling
        q.observing = True
    predict(model, calibration, args.calibration_batch_size)
    for q in quantizers.values():
        q.observing = False
    report = dict(arguments=vars(args), architecture="CIFAR VGG16-BN (13 conv + 3 linear)",
                  model_source=CHECKPOINT_URL, model_sha256=checkpoint_hash, model_code_commit=MODEL_COMMIT,
                  preprocessing_commit=PREPROCESS_COMMIT,
                  dataset_source=DATA_URL, dataset_download_url=args.dataset_download_url, dataset_md5=DATA_MD5,
                  preprocessing=dict(mean=MEAN, std=STD, resize=False, augmentation=False),
                  versions=dict(torch=torch.__version__), calibration_split="train", evaluation_split="test",
                  calibration_ids=calibration_ids, evaluation_ids=list(range(count)),
                  evaluation_sha256=evaluation_hash, weight_quantization=weight_metadata,
                  quantization_sites={name: dict(signed=q.signed) for name, q in quantizers.items()},
                  limitations="Post-training numerical quantization; zero training epochs. "
                              "Weights are stored as integers and decoded to FP32 for conventional kernels. "
                              "Biases, BatchNorm and accumulators stay FP32; ReLU/pooling are unchanged. "
                              "No neuron dynamics or hardware energy/timing measurement.", results=[])

    def record(name, logits, seconds, scheme="none", bits=32, config=None):
        row = dict(variant=name, scheme=scheme, bits=bits,
                   weight_bits=32 if name == "checkpoint" else args.weight_bits,
                   range_offset=0, arithmetic="dense", seconds=seconds,
                   **metrics(logits, labels, checkpoint_logits, reference_logits),
                   top5_accuracy=float((logits.topk(5, dim=-1).indices == labels[:, None]).any(1).double().mean()),
                   predictions=logits.argmax(-1).tolist())
        if config is not None:
            row.update(calibration=config, statistics={k: q.stats for k, q in quantizers.items()} if args.collect_stats else {})
        report["results"].append(row)
        save_report(out, report)
        print(f"{name}: accuracy={row['accuracy']:.4f}; loss={row['loss_vs_checkpoint_pp']:+.2f} pp", flush=True)

    record("checkpoint", checkpoint_logits, checkpoint_seconds)
    if args.weight_bits != 32:
        record(f"weights_int{args.weight_bits}", weight_logits, weight_seconds)
    for scheme in args.schemes:
        for bits in args.bits:
            config = {name: q.configure(scheme, bits, range(-6, 3)) for name, q in quantizers.items()}
            for q in quantizers.values():
                q.recording = args.collect_stats
            start = time.perf_counter()
            logits = predict(model, evaluation, args.batch_size)
            record(f"{scheme}_a{bits}", logits, time.perf_counter() - start, scheme, bits, config)
    return report


if __name__ == "__main__":
    main()
