"""Quantization-aware fine-tuning (QAT) of VGG16-BN on CIFAR-100.

Follows finetuning/README.md:
- the quantized student runs the exact quantized forward values;
- gradients reach the FP32 master weights through the straight-through
  surrogate (activation_ste / weight fake-quantization);
- a frozen FP32 teacher provides distillation targets;
- activation ranges come from training-only calibration and stay frozen;
- the main result is the final epoch; the 5,000-image development split
  (also part of the pretrained model's training data) is tracked per epoch
  and its best checkpoint is reported as a secondary number.

BatchNorm running statistics stay fixed (its affine parameters train);
dropout is off so train-time activations match the calibrated ranges.

Configurations (first site is the signed image input, later sites follow ReLU):
  hw_w4_log7      W4 (-7..7) per channel, 7 log levels per site: the ShiftNet hardware
  w16_log_a4      INT16 weights, log A4 (7 signed / 15 unsigned levels): logQuant PTQ 54.02%
  w16_uniform_a4  INT16 weights, uniform A4: the control (PTQ 64.01%)
  hw_w4_uniform_a4 W4 per channel, uniform A4 (15 unsigned levels): INT4 SparTen / linear TTFS T=15
  hw_w4_log3      W4 per channel, 3 log levels per site (unsigned 2-bit): log TTFS T=3
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
import types
from pathlib import Path

import torch
from torch.nn import functional as F

from ..image_models.data import CHECKPOINT_SHA256, DATA_MD5, digest, load_split
from ..image_models.model import VGG16BN, QuantizedConv2d, instrument_image
from ..modeling import QuantizedLinear
from ..weights import load_weights
from .prepare_data import stratified_cifar_split
from .qat_reference import activation_ste, distillation_loss

CONFIGS = {
    "hw_w4_log7": dict(weight_bits=4, scheme="log", bits=(4, 3)),
    "w16_log_a4": dict(weight_bits=16, scheme="log", bits=(4, 4)),
    "w16_uniform_a4": dict(weight_bits=16, scheme="uniform", bits=(4, 4)),
    "hw_w4_uniform_a4": dict(weight_bits=4, scheme="uniform", bits=(4, 4)),
    "hw_w4_log3": dict(weight_bits=4, scheme="log", bits=(3, 2)),
}
MEAN = (0.5070, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2761)


def fake_weight(weight, bits):
    """Symmetric per-output-channel integer weights; identity gradient to the masters."""
    qmax = (1 << (bits - 1)) - 1
    maximum = weight.detach().flatten(1).abs().amax(1).reshape((-1,) + (1,) * (weight.ndim - 1))
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / qmax)
    integer = (weight.detach() / scale).round().clamp(-qmax, qmax)
    return integer * scale + (weight - weight.detach()), integer, scale


def activation(q, x):
    """Exact quantized forward; clipped straight-through gradient.

    The plain identity surrogate (qat_reference.activation_ste) also passes
    gradient through saturated values, which let activations run away in the
    first VGG run (job 2549681.hn1 diverged at epoch 3). Here the gradient is
    zero where the input is beyond the code's top boundary, 2^(e_max + 1/2)
    for log codes and 2^e_max for uniform ones.
    """
    if q.observing or q.mode == "none":
        return q(x)
    top = 2.0 ** (q.e_max + (0.5 if q.mode == "log" else 0.0))
    inside = (x.abs() <= top) if q.signed else (x <= top)
    return activation_ste(x, q.mode, q.bits, q.e_max, q.signed).detach() + (x - x.detach()) * inside


def make_student(model, weight_bits):
    quantizers = instrument_image(model)
    for module in model.modules():
        if isinstance(module, QuantizedConv2d):
            def forward(self, x):
                w = fake_weight(self.weight, weight_bits)[0]
                return F.conv2d(activation(self.quantizer, x), w, self.bias, self.stride, self.padding,
                                self.dilation, self.groups)
            module.forward = types.MethodType(forward, module)
        elif isinstance(module, QuantizedLinear):
            def forward(self, x):
                return F.linear(activation(self.quantizer, x), fake_weight(self.weight, weight_bits)[0], self.bias)
            module.forward = types.MethodType(forward, module)
    return quantizers


def normalize(images_u8):
    x = images_u8.float() / 255
    return (x - x.new_tensor(MEAN).view(1, 3, 1, 1)) / x.new_tensor(STD).view(1, 3, 1, 1)


def augment(images_u8, generator):
    """Random 4-pixel-padded crop and horizontal flip, on the GPU."""
    b = images_u8.shape[0]
    padded = F.pad(images_u8, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (b,), device=images_u8.device, generator=generator)
    dx = torch.randint(0, 9, (b,), device=images_u8.device, generator=generator)
    base = torch.arange(32, device=images_u8.device)
    rows = (dy[:, None] + base)[:, None, :, None]
    cols = (dx[:, None] + base)[:, None, None, :]
    flip = torch.rand(b, device=images_u8.device, generator=generator) < 0.5
    cols = torch.where(flip[:, None, None, None], 31 - cols + 2 * dx[:, None, None, None], cols)
    return padded[torch.arange(b, device=images_u8.device)[:, None, None, None],
                  torch.arange(3, device=images_u8.device)[None, :, None, None], rows, cols]


@torch.no_grad()
def accuracy(model, images, labels, batch):
    correct = 0
    for s in range(0, len(images), batch):
        correct += int((model(normalize(images[s:s + batch])).argmax(-1) == labels[s:s + batch]).sum())
    return correct / len(images)


def export(model, quantizers, weight_bits):
    """Integer weights and scales, FP32 remainder, frozen activation configuration."""
    matrices = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            _, integer, scale = fake_weight(module.weight, weight_bits)
            matrices[name + ".weight"] = dict(values=integer.to(torch.int8 if weight_bits <= 8 else torch.int16).cpu(),
                                              scale=scale.cpu())
    sites = {name: dict(mode=q.mode, bits=q.bits, e_max=q.e_max, signed=q.signed, rounding=q.rounding)
             for name, q in quantizers.items()}
    return dict(format="logquant_qat_vgg16_v1", weight_bits=weight_bits, axis=0, matrices=matrices,
                master_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                activation_sites=sites)


def run(name, cfg, a, base_state, data, out_dir, ckpt_dir, log):
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    train_x, train_y, dev_x, dev_y, test_x, test_y, calib_x = data
    teacher = VGG16BN().eval().to(a.device)
    teacher.load_state_dict(base_state)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = VGG16BN().eval().to(a.device)
    student.load_state_dict(base_state)
    quantizers = make_student(student, cfg["weight_bits"])
    for q in quantizers.values():
        q.sampling, q.observing = "random_extrema", True
    with torch.no_grad():
        for s in range(0, len(calib_x), a.calibration_batch_size):
            student(normalize(calib_x[s:s + a.calibration_batch_size]))
    sites = {}
    for i, q in enumerate(quantizers.values()):
        q.observing = False
        sites[q.name] = q.configure(cfg["scheme"], cfg["bits"][0] if i == 0 else cfg["bits"][1], range(-6, 3))
        q.recording = False
    record = dict(config=cfg, sites=sites, epochs=[])
    record["ptq_dev"] = accuracy(student, dev_x, dev_y, a.eval_batch)
    record["ptq_test"] = accuracy(student, test_x, test_y, a.eval_batch)
    log(f"[{name}] PTQ start: dev {record['ptq_dev']:.4f} test {record['ptq_test']:.4f}")
    # BatchNorm running statistics stay fixed: the model stays in eval mode while training.
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.weight_decay)
    steps = a.epochs * math.ceil(len(train_x) / a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)
    best = (record["ptq_dev"], 0)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(export(student, quantizers, cfg["weight_bits"]), epoch=0, dev_accuracy=best[0]),
               ckpt_dir / "best.pt")
    for epoch in range(1, a.epochs + 1):
        started, total, seen = time.time(), 0.0, 0
        order = torch.randperm(len(train_x), device=a.device, generator=gen)
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            x = normalize(augment(train_x[idx], gen))
            with torch.no_grad():
                t_logits = teacher(x)
            loss = distillation_loss(student(x), t_logits, train_y[idx], a.alpha, a.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
            seen += len(idx)
        dev = accuracy(student, dev_x, dev_y, a.eval_batch)
        entry = dict(epoch=epoch, loss=total / seen, dev_accuracy=dev, lr=sched.get_last_lr()[0],
                     seconds=time.time() - started)
        record["epochs"].append(entry)
        log(f"[{name}] epoch {epoch}: loss {entry['loss']:.4f} dev {dev:.4f} ({entry['seconds']:.0f} s)")
        if dev > best[0]:
            best = (dev, epoch)
            torch.save(dict(export(student, quantizers, cfg["weight_bits"]), epoch=epoch, dev_accuracy=dev),
                       ckpt_dir / "best.pt")
        torch.save(dict(export(student, quantizers, cfg["weight_bits"]), epoch=epoch, dev_accuracy=dev),
                   ckpt_dir / "last.pt")
    # Main result: the final epoch (the schedule anneals to ~0). The development split was
    # part of the pretrained checkpoint's training data (the teacher scores ~100% on it), so
    # best-dev selection is reported only as a secondary number.
    record["last_epoch"] = a.epochs
    record["test_accuracy"] = accuracy(student, test_x, test_y, a.eval_batch)
    chosen = torch.load(ckpt_dir / "best.pt", map_location=a.device, weights_only=True)
    student.load_state_dict(chosen["master_state"])
    for mname, module in student.named_modules():
        if mname + ".weight" in chosen["matrices"]:
            m = chosen["matrices"][mname + ".weight"]
            assert torch.equal(fake_weight(module.weight, cfg["weight_bits"])[1].cpu(), m["values"].float().cpu())
    record["best_epoch"], record["best_dev"] = best[1], best[0]
    record["best_dev_test_accuracy"] = accuracy(student, test_x, test_y, a.eval_batch)
    for kind in ("last", "best"):
        record[f"{kind}_checkpoint"] = str(ckpt_dir / f"{kind}.pt")
        record[f"{kind}_sha256"] = hashlib.sha256((ckpt_dir / f"{kind}.pt").read_bytes()).hexdigest()
    log(f"[{name}] final epoch test {record['test_accuracy']:.4f}; best-dev epoch {best[1]} "
        f"(dev {best[0]:.4f}) test {record['best_dev_test_accuracy']:.4f}")
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="original FP32 VGG16-BN checkpoint (teacher and student start)")
    p.add_argument("--weights", help="or: a logQuant INT16 bundle (local smoke tests)")
    p.add_argument("--data-archive", required=True)
    p.add_argument("--manifest", help="data_manifest.json; its CIFAR split must match the recomputed one")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="where best.pt / last.pt go, per config")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip", type=float, default=2.0, help="gradient-norm clipping")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--calibration-batch-size", type=int, default=8)
    p.add_argument("--limit-train", type=int, default=0, help="smoke tests only")
    p.add_argument("--limit-eval", type=int, default=0, help="smoke tests only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    if a.checkpoint_dir.exists():
        raise FileExistsError(f"Choose a fresh checkpoint directory: {a.checkpoint_dir}")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    log_file = (a.output_dir / "train.log").open("w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    if digest(a.data_archive, "md5") != DATA_MD5:
        raise ValueError("CIFAR-100 archive MD5 differs from the official dataset")
    model = VGG16BN()
    if a.checkpoint:
        if digest(a.checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("unexpected VGG16-BN checkpoint")
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True), strict=True)
        source = dict(checkpoint=a.checkpoint, sha256=CHECKPOINT_SHA256)
    else:
        load_weights(model, a.weights)
        source = dict(weights=a.weights, sha256=digest(a.weights))
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    images, labels = load_split(a.data_archive, "train")
    train_ids, dev_ids = stratified_cifar_split(labels, a.seed)
    calib_ids = random.Random(a.seed).sample(train_ids, a.calibration_samples)
    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())["cifar100"]
        if m["train_ids"] != train_ids or m["development_ids"] != dev_ids or \
                (a.calibration_samples == 128 and m["calibration_ids"] != calib_ids):
            raise ValueError("recomputed split differs from the manifest")
    if a.limit_train:
        train_ids = train_ids[:a.limit_train]
    test_images, test_labels = load_split(a.data_archive, "test")
    if a.limit_eval:
        dev_ids, test_images, test_labels = dev_ids[:a.limit_eval], test_images[:a.limit_eval], test_labels[:a.limit_eval]
    to = lambda arr: torch.from_numpy(arr).to(a.device)  # noqa: E731
    data = (to(images[train_ids]), to(labels[train_ids]), to(images[dev_ids]), to(labels[dev_ids]),
            to(test_images), to(test_labels), to(images[calib_ids]))
    report = dict(arguments={k: str(v) for k, v in vars(a).items()}, source=source, torch=torch.__version__,
                  train_images=len(train_ids), dev_images=len(dev_ids), test_images=len(test_labels),
                  results={})
    teacher = VGG16BN().eval().to(a.device)
    teacher.load_state_dict(base_state)
    report["teacher_dev"] = accuracy(teacher, data[2], data[3], a.eval_batch)
    report["teacher_test"] = accuracy(teacher, data[4], data[5], a.eval_batch)
    log(f"teacher: dev {report['teacher_dev']:.4f} test {report['teacher_test']:.4f}")
    del teacher
    for name in a.configs:
        report["results"][name] = run(name, CONFIGS[name], a, base_state, data, a.output_dir,
                                      a.checkpoint_dir / name, log)
        (a.output_dir / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [f"Teacher (FP32): dev {100 * report['teacher_dev']:.2f}%, test {100 * report['teacher_test']:.2f}%", "",
             "Dev = 5,000 held-out training images, which the pretrained teacher saw in training.", "",
             "| Config | PTQ test | QAT test (final epoch) | QAT test (best-dev epoch) | PTQ dev | best dev (epoch) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in report["results"].items():
        lines.append(f"| {name} | {100 * r['ptq_test']:.2f}% | **{100 * r['test_accuracy']:.2f}%** | "
                     f"{100 * r['best_dev_test_accuracy']:.2f}% | {100 * r['ptq_dev']:.2f}% | "
                     f"{100 * r['best_dev']:.2f}% ({r['best_epoch']}) |")
    (a.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
