"""Export VGG16-BN conv layers of a QAT checkpoint as integer problems for the CNN engines.

    python hardware/sim_runs/export_conv.py --checkpoint last.pt --data-archive cifar.tar.gz \
        --output DIR [--images 0 1] [--device cuda]

Runs on Aqua (PyTorch). The checkpoint is a logQuant QAT export (train_vgg.py): integer W4
weights, master state and frozen activation sites. Every conv layer except the first (whose
input is the signed image) is written for each image as

    DIR/<layer>/meta.json      shapes, code, scales, per-image statistics
    DIR/<layer>/weights.bin    int8 [Cout, Cin, 3, 3], -7..7
    DIR/<layer>/thresholds.bin int32 [Cout, T]: output tick t fires when S reaches it
    DIR/<layer>/down.bin       uint8 [Cout]: 1 = the channel compares downwards (S <= threshold)
    DIR/<layer>/img<k>_ticks.bin    uint8 [Cin, H, W] input spike tick, T = no spike
    DIR/<layer>/img<k>_sums.bin     int32 [Cout, H, W] exact integer conv sums S
    DIR/<layer>/img<k>_out.bin      uint8 [Cout, H, W] reference output tick from S

Codes (T ticks; tick T means zero / no spike):
    uniform (15 levels): value k = T - t        (linear TTFS, and SparTen's INT4 value)
    log (L levels):      value 2^(T - 1 - t)    (log TTFS)
S = sum over the 3x3xCin window of weight * value, zero padding 1. The output tick of channel c
is the first t with S >= thresholds[c, t] (S <= for down channels), which is the next layer's
quantizer applied to ReLU(BN(conv)) before max-pooling (pooling commutes with the code).
"""
import argparse
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from logQuant.finetuning.train_vgg import accuracy, fake_weight, make_student  # noqa: E402
from logQuant.image_models.data import load_split  # noqa: E402
from logQuant.image_models.model import VGG16BN, QuantizedConv2d  # noqa: E402
from logQuant.quantization import encode_log, levels  # noqa: E402

INT32 = 2**31 - 1


def ticks_of(x, site):
    """Spike tick per element (T = silent) and the real value of one hardware unit."""
    n = levels(site["bits"], False)
    if site["mode"] == "uniform":
        step = math.ldexp(1.0, site["e_max"]) / n
        k = torch.floor(x.clamp_min(0) / step + 0.5).clamp(0, n).long()
        return n - k, n, step
    code = encode_log(x, site["bits"], site["e_max"], False, site["rounding"])
    t = torch.where(code.active, code.times().long(), torch.full_like(code.exponent, n).long())
    return t, n, math.ldexp(1.0, site["e_max"] - n + 1)


def values_of(t, mode, T):
    if mode == "uniform":
        return (T - t).double()
    return torch.where(t < T, torch.pow(2.0, (T - 1 - t).double()), torch.zeros_like(t, dtype=torch.float64))


def output_boundaries(site):
    """Real-valued threshold of output tick t = 0..T-1 under the next site's code."""
    n = levels(site["bits"], False)
    if site["mode"] == "uniform":
        step = math.ldexp(1.0, site["e_max"]) / n
        return [(n - t - 0.5) * step for t in range(n)]
    return [2.0 ** (site["e_max"] - t - 0.5) for t in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", type=int, nargs="+", default=[0, 1])
    p.add_argument("--device", default="cuda")
    p.add_argument("--weight-bits", type=int, default=0, help="override the checkpoint's (e.g. 4 for an INT16 checkpoint)")
    p.add_argument("--skip-accuracy", action="store_true", help="do not evaluate the 10,000 test images")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = VGG16BN().eval()
    wbits = a.weight_bits or ck["weight_bits"]
    quantizers = make_student(model, wbits)
    model.load_state_dict(ck["master_state"])
    for name, s in ck["activation_sites"].items():
        q = quantizers[name]
        q.mode, q.bits, q.e_max, q.signed, q.rounding = s["mode"], s["bits"], s["e_max"], s["signed"], s["rounding"]
        q.observing = q.recording = False
    model.to(a.device)
    test_x, test_y = load_split(str(a.data_archive), "test")
    from logQuant.finetuning.train_vgg import normalize
    images = torch.from_numpy(test_x).to(a.device)
    labels = torch.from_numpy(test_y).to(a.device)
    top1 = None if a.skip_accuracy else accuracy(model, images, labels, 500)
    print(f"checkpoint W{wbits} test accuracy {top1}", flush=True)

    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, QuantizedConv2d)]
    feats = list(model.features)
    index = {id(m): i for i, m in enumerate(feats)}
    seen = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, name=n: seen.__setitem__(name, inp[0].detach()))
             for n, m in convs]
    with torch.no_grad():
        model(normalize(images[a.images]))
    for h in hooks:
        h.remove()
    sites = ck["activation_sites"]
    next_site = {convs[i][0]: convs[i + 1][0] + ".input" for i in range(len(convs) - 1)}
    next_site[convs[-1][0]] = "classifier.0.input"
    summary = dict(checkpoint=str(a.checkpoint), weight_bits=wbits, test_accuracy=top1, images=a.images,
                   labels=[int(test_y[i]) for i in a.images], layers=[])
    for li, (name, conv) in enumerate(convs[1:], start=1):
        site, nsite = sites[name + ".input"], sites[next_site[name]]
        assert site["mode"] == nsite["mode"] and not site["signed"] and not nsite["signed"]
        x = seen[name].double()
        t, T, unit = ticks_of(x, site)
        assert levels(nsite["bits"], False) == T
        _, w_int, w_scale = fake_weight(conv.weight.detach(), wbits)
        w_int = w_int.double()
        w_scale = w_scale.double().flatten()
        sums = F.conv2d(values_of(t, site["mode"], T), w_int, None, 1, 1)
        assert sums.abs().max() < INT32 and torch.equal(sums, sums.round())
        bn = feats[index[id(conv)] + 1]
        g = (bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps))
        A = g * w_scale * unit
        B = g * (conv.bias.double() - bn.running_mean.double()) + bn.bias.double()
        bounds = torch.tensor(output_boundaries(nsite), dtype=torch.float64, device=A.device)
        down = A < 0
        raw = (bounds[None, :] - B[:, None]) / torch.where(A == 0, torch.ones_like(A), A)[:, None]
        theta = torch.where(down[:, None], torch.floor(raw), torch.ceil(raw))
        never = torch.where(down, -INT32, INT32)[:, None].expand_as(theta).double()
        zero_hit = (B[:, None] >= bounds[None, :])  # A == 0: constant output
        theta = torch.where((A == 0)[:, None], torch.where(zero_hit, -INT32 * 1.0, never), theta)
        theta = theta.clamp(-INT32, INT32).long()
        # reference output tick: first t whose threshold is reached
        S = sums.long()
        reached = torch.where(down[None, :, None, None, None],
                              S[..., None] <= theta[None, :, None, None, :],
                              S[..., None] >= theta[None, :, None, None, :])
        out = torch.where(reached.any(-1), reached.float().argmax(-1), torch.full_like(S, T))
        # the model's own next-layer code (pre-pool) for agreement
        y = torch.relu(A[None, :, None, None] * sums + B[None, :, None, None])
        model_t, _, _ = ticks_of(y, nsite)
        agree = float((model_t == out).double().mean())
        d = a.output / name
        d.mkdir()
        w_int.to(torch.int8).cpu().numpy().tofile(d / "weights.bin")
        theta.to(torch.int32).cpu().numpy().tofile(d / "thresholds.bin")
        down.to(torch.uint8).cpu().numpy().tofile(d / "down.bin")
        stats = []
        for k, img in enumerate(a.images):
            t[k].to(torch.uint8).cpu().numpy().tofile(d / f"img{k}_ticks.bin")
            S[k].to(torch.int32).cpu().numpy().tofile(d / f"img{k}_sums.bin")
            out[k].to(torch.uint8).cpu().numpy().tofile(d / f"img{k}_out.bin")
            stats.append(dict(image=img, input_density=float((t[k] < T).double().mean()),
                              output_density=float((out[k] < T).double().mean()),
                              tick_histogram=torch.bincount(t[k].flatten(), minlength=T + 1).tolist()))
        meta = dict(name=name, index=li, cin=int(w_int.shape[1]), cout=int(w_int.shape[0]),
                    h=int(x.shape[2]), w=int(x.shape[3]), T=T, mode=site["mode"], bits=site["bits"],
                    input_e_max=site["e_max"], output_e_max=nsite["e_max"],
                    weight_density=float((w_int != 0).double().mean()),
                    down_channels=int(down.sum()), model_agreement=agree, images=stats)
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        summary["layers"].append({k: meta[k] for k in ("name", "cin", "cout", "h", "w", "T", "weight_density",
                                                        "model_agreement")} |
                                 dict(input_density=[s["input_density"] for s in stats]))
        print(f"{name}: {meta['cin']}->{meta['cout']} @{meta['h']}x{meta['w']} T={T} "
              f"in-density {[round(s['input_density'], 3) for s in stats]} w-density {meta['weight_density']:.3f} "
              f"agree {agree:.5f}", flush=True)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
