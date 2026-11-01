"""Evaluate fixed-checkpoint BERT accuracy under alternative activation codes.

Run from the repository root: python -m logQuant.evaluate --help
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .modeling import instrument, set_arithmetic
from .weights import load_weights, quantize_weights, save_weights


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or not {"sentence", "label"} <= rows[0].keys():
        raise ValueError(f"{path} must have sentence and label columns (labeled SST-2 format)")
    result = [{"sentence": row["sentence"], "label": int(row["label"]), "idx": i}
              for i, row in enumerate(rows)]
    if any(row["label"] not in (0, 1) for row in result):
        raise ValueError("SST-2 labels must be 0 or 1")
    return result


def load_rows(args):
    if args.data_dir:
        train = read_tsv(Path(args.data_dir) / "train.tsv")
        evaluation = read_tsv(Path(args.data_dir) / "dev.tsv")
    else:
        from datasets import load_dataset
        dataset = load_dataset("glue", "sst2", cache_dir=str(Path(args.cache_dir) / "datasets"))
        train, evaluation = dataset["train"], dataset["validation"]
    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    calibration = [dict(train[i]) for i in indices[:args.calibration_samples]]
    count = min(args.max_eval_samples, len(evaluation)) if args.max_eval_samples else len(evaluation)
    evaluation = [dict(evaluation[i]) for i in range(count)]
    if not calibration or not evaluation:
        raise ValueError("Calibration and evaluation sets must both be nonempty")
    return calibration, evaluation


def batches(rows, tokenizer, args):
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        tensors = tokenizer([row["sentence"] for row in chunk], padding="max_length",
                            truncation=True, max_length=args.max_length, return_tensors="pt")
        yield {k: v.to(args.device) for k, v in tensors.items()}


@torch.inference_mode()
def predict(model, encoded_batches):
    outputs = []
    for batch in encoded_batches:
        logits = model(**batch).logits.float().cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model logits")
        outputs.append(logits)
    return torch.cat(outputs)


def metrics(logits, labels, checkpoint_logits, reference_logits):
    pred = logits.argmax(-1)
    accuracy = float((pred == labels).double().mean())
    baseline = float((reference_logits.argmax(-1) == labels).double().mean())
    checkpoint_accuracy = float((checkpoint_logits.argmax(-1) == labels).double().mean())
    return dict(accuracy=accuracy, correct=int((pred == labels).sum()), examples=len(labels),
                accuracy_loss_pp=100 * (baseline - accuracy),
                loss_vs_checkpoint_pp=100 * (checkpoint_accuracy - accuracy),
                agreement_with_reference=float((pred == reference_logits.argmax(-1)).double().mean()),
                logits_rmse=float((logits - reference_logits).square().mean().sqrt()))


def save_report(output_dir, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_dir / "results.json")
    fields = ["variant", "scheme", "bits", "weight_bits", "range_offset", "arithmetic", "accuracy", "top5_accuracy", "correct",
              "examples", "accuracy_loss_pp", "loss_vs_checkpoint_pp", "agreement_with_reference",
              "logits_rmse", "seconds"]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--revision", default="main", help="HF checkpoint revision; resolved commit is recorded")
    p.add_argument("--data-dir", help="Optional local SST-2 directory with train.tsv and dev.tsv")
    p.add_argument("--cache-dir", default="logQuant/.cache")
    p.add_argument("--output-dir", default="logQuant/results/sst2")
    p.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6], choices=range(2, 17))
    p.add_argument("--weight-bits", type=int, choices=[8, 16, 32], default=32,
                   help="Row-wise symmetric Linear/Embedding weights; 32 preserves original weights")
    p.add_argument("--export-weights", action="store_true",
                   help="Save integer weights/scales, tokenizer and config in output-dir/weights")
    p.add_argument("--schemes", nargs="+", default=["uniform", "log"], choices=["uniform", "log"])
    p.add_argument("--scope", choices=["linear", "attention", "all"], default="all")
    p.add_argument("--kv-mode", choices=["float", "binary"], default="float",
                   help="Binary K/V is a separate optional ablation; held fixed across quantizers")
    p.add_argument("--rounding", choices=["nearest", "floor"], default="nearest")
    p.add_argument("--calibration-offsets", type=int, nargs="+", default=list(range(-6, 3)))
    p.add_argument("--range-offsets", type=int, nargs="+", default=[0],
                   help="Shift fitted exponent ranges for sensitivity analysis; never fit on dev labels")
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--calibration-sampling", choices=["random_extrema", "legacy_linspace"],
                   default="random_extrema", help="Training observer; legacy repeats fixed positions")
    p.add_argument("--max-eval-samples", type=int, default=0, help="0 uses the full validation split")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arithmetic", choices=["dense", "shift"], default="dense",
                   help="Shift is a slow floating exponent-adjust/add reference for LOG variants only")
    p.add_argument("--collect-stats", action="store_true", help="Record evaluation quantization error/zero/clipping counts")
    p.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer from local cache only; use --data-dir too")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if any((output_dir / name).exists() for name in ("results.json", "summary.csv", "weights")):
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")
    if args.export_weights and args.weight_bits == 32:
        raise ValueError("--export-weights requires --weight-bits 8 or 16")
    if min(args.calibration_samples, args.batch_size, args.max_length, args.threads) < 1 or args.max_eval_samples < 0:
        raise ValueError("Counts/length/threads must be positive; max-eval-samples must be nonnegative")
    if args.local_files_only and not args.data_dir:
        raise ValueError("For offline evaluation provide --data-dir as well as --local-files-only")
    if args.arithmetic == "shift" and args.max_eval_samples == 0:
        raise ValueError("The shift reference is slow; set --max-eval-samples explicitly")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    from transformers import BertForSequenceClassification, BertTokenizer, __version__ as transformers_version
    options = dict(cache_dir=str(Path(args.cache_dir) / "models"), revision=args.revision,
                   local_files_only=args.local_files_only)
    print(f"Loading {args.model}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.model, **options)
    model, loading = BertForSequenceClassification.from_pretrained(
        args.model, attn_implementation="eager", output_loading_info=True, **options)
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError(f"Checkpoint did not load completely: {loading}")
    if model.config.num_labels != 2:
        raise ValueError("Use a BERT checkpoint already fine-tuned for SST-2 (two labels)")
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds checkpoint positional embeddings")
    model = model.to(args.device).eval()
    calibration, evaluation = load_rows(args)
    calibration_batches = list(batches(calibration, tokenizer, args))
    evaluation_batches = list(batches(evaluation, tokenizer, args))
    labels = torch.tensor([row["label"] for row in evaluation])
    print(f"Evaluating original checkpoint on {len(evaluation)} validation examples", flush=True)
    start = time.perf_counter()
    checkpoint_logits = predict(model, evaluation_batches)
    checkpoint_seconds = time.perf_counter() - start
    print(f"Checkpoint accuracy: {float((checkpoint_logits.argmax(-1) == labels).float().mean()):.4f} "
          f"on {len(evaluation)} examples ({checkpoint_seconds:.1f}s)", flush=True)
    weight_logits, weight_seconds = checkpoint_logits, checkpoint_seconds
    weight_metadata = dict(bits=args.weight_bits, scheme="none", exported=False)
    if args.weight_bits != 32:
        print(f"Quantizing Linear/Embedding weights to INT{args.weight_bits}", flush=True)
        bundle = quantize_weights(model, args.weight_bits)
        weight_metadata.update(scheme="symmetric_per_row", matrices=len(bundle["matrices"]),
                               integer_parameters=sum(e["values"].numel() for e in bundle["matrices"].values()),
                               retained_float_tensors=list(bundle["float_state"]))
        start = time.perf_counter()
        weight_logits = predict(model, evaluation_batches)
        weight_seconds = time.perf_counter() - start
        if args.export_weights:
            save_weights(bundle, output_dir / "weights", model, tokenizer,
                         dict(source_model=args.model, revision=args.revision,
                              model_commit=getattr(model.config, "_commit_hash", None)))
            # Evaluate the same values after loading the exported integers/scales.
            load_weights(model, output_dir / "weights" / "weights.pt")
            torch.testing.assert_close(predict(model, evaluation_batches[:1]),
                                       weight_logits[:len(evaluation_batches[0]["input_ids"])],
                                       atol=0, rtol=0, msg="Export/reload changed weight-only logits")
            weight_metadata.update(exported=True, checkpoint="weights/weights.pt",
                                   reload_logits_verified=True)
        del bundle
        print(f"INT{args.weight_bits} weight-only accuracy: "
              f"{float((weight_logits.argmax(-1) == labels).float().mean()):.4f}", flush=True)
    quantizers = instrument(model, args.scope, args.kv_mode)
    print("Checking instrumented model with activation quantization disabled", flush=True)
    start = time.perf_counter()
    reference_logits = predict(model, evaluation_batches)
    reference_seconds = time.perf_counter() - start
    if args.kv_mode == "float":
        torch.testing.assert_close(reference_logits, weight_logits, atol=2e-5, rtol=2e-5,
                                   msg="Pass-through instrumentation changed checkpoint outputs")
    for q in quantizers.values():
        q.sampling = args.calibration_sampling
        q.observing = True
    print(f"Collecting ranges at {len(quantizers)} sites from {len(calibration)} training examples", flush=True)
    predict(model, calibration_batches)
    for q in quantizers.values():
        q.observing = False
    report = dict(arguments=vars(args), model_commit=getattr(model.config, "_commit_hash", None),
                  label_mapping=model.config.id2label,
                  versions=dict(torch=torch.__version__, transformers=transformers_version),
                  calibration_split="train", evaluation_split="validation",
                  calibration_ids=[r["idx"] for r in calibration],
                  evaluation_ids=[r["idx"] for r in evaluation],
                  evaluation_sha256=hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest(),
                  quantization_sites=list(quantizers),
                  weight_quantization=weight_metadata,
                  limitations="Activation-value experiment; no neuron dynamics or measured hardware energy. "
                              "Quantized weights are decoded to FP32 for execution. Biases, LayerNorm, "
                              "accumulators and other unquantized operations remain floating point; "
                              "this is not an integer-kernel or fixed-point accumulator simulation.",
                  results=[])
    baselines = [("checkpoint", checkpoint_logits, checkpoint_seconds)]
    if args.weight_bits != 32:
        baselines.append((f"weights_int{args.weight_bits}", weight_logits, weight_seconds))
    baselines.append(("activation_fp", reference_logits, reference_seconds))
    for name, logits, seconds in baselines:
        report["results"].append(dict(variant=name, scheme="none", bits=32, range_offset=0,
                                      weight_bits=32 if name == "checkpoint" else args.weight_bits,
                                      arithmetic="dense", seconds=seconds,
                                      **metrics(logits, labels, checkpoint_logits, reference_logits)))
    save_report(output_dir, report)
    for scheme in args.schemes:
        for bits in args.bits:
            for offset in args.range_offsets:
                config = {name: q.configure(scheme, bits, args.calibration_offsets, args.rounding, offset)
                          for name, q in quantizers.items()}
                for q in quantizers.values():
                    q.recording = args.collect_stats
                set_arithmetic(model, args.arithmetic)
                print(f"Evaluating {scheme}, {bits} bits, range offset {offset}", flush=True)
                start = time.perf_counter()
                logits = predict(model, evaluation_batches)
                row = dict(variant=f"{scheme}_a{bits}_offset{offset}", scheme=scheme, bits=bits,
                           weight_bits=args.weight_bits,
                           range_offset=offset, arithmetic=args.arithmetic if scheme == "log" else "dense",
                           seconds=time.perf_counter() - start,
                           **metrics(logits, labels, checkpoint_logits, reference_logits),
                           calibration=config, statistics={k: q.stats for k, q in quantizers.items()} if args.collect_stats else {},
                           predictions=logits.argmax(-1).tolist())
                report["results"].append(row)
                save_report(output_dir, report)
                print(f"{row['variant']}: accuracy={row['accuracy']:.4f}, "
                      f"loss={row['accuracy_loss_pp']:+.2f} pp, time={row['seconds']:.1f}s", flush=True)
    print(f"Saved {output_dir / 'summary.csv'} and results.json", flush=True)
    return report


if __name__ == "__main__":
    main()
