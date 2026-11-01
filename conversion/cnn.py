"""LSQ W4A4 VGG -> TTFS graph (integer weights, per-tick thresholds), integer TTFS reference and checks."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ir import Graph, Node

T_CNN = 15


def _bn_affine(bn):
    a = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    return a, (bn.bias.detach().double() - a * bn.running_mean.double())


def quantize_classifier(model, bits=4):
    """Post-training per-row symmetric quantisation of the float classifier, in place,
    so the PyTorch QNN (the reference) and the TTFS graph share the same weights.
    4-bit: 91.35 % vs 91.39 % float on the CIFAR-10 test set."""
    W = model.fc.weight.data
    qp = 2 ** (bits - 1) - 1
    step = W.abs().amax(1, keepdim=True) / qp
    model.fc.weight.data = torch.round(W / step).clamp(-qp - 1, qp) * step
    model.fc.w_step, model.fc.w_bits = step.squeeze(1).double(), bits
    return model


def import_qcnn(model):
    g = Graph()
    g.add(Node("x", "input", attrs={"shape": (3, 32, 32)}))
    prev, li, pi = "x", 0, 0
    for blk in model.blocks:
        if isinstance(blk, nn.MaxPool2d):
            prev = g.add(Node(f"pool{pi}", "pool_max", [prev], {"k": 2})).name
            pi += 1
            continue
        conv, bn, aq = blk["conv"], blk["bn"], blk["aq"]
        step_w = conv.wq.step.detach().double()
        w_int = torch.clamp(
            torch.round(conv.weight.detach().double() / step_w.view(-1, 1, 1, 1)), conv.wq.qn, conv.wq.qp
        )
        a, b = _bn_affine(bn)
        g.add(
            Node(
                f"conv{li}",
                "conv",
                [prev],
                {"w_int": w_int, "step_w": step_w, "k": 3, "pad": 1},
                digital=(li == 0),
            )
        )
        g.add(Node(f"bn{li}", "bn", [f"conv{li}"], {"a": a, "b": b}))
        g.add(Node(f"relu{li}", "relu", [f"bn{li}"]))
        prev = g.add(
            Node(
                f"quant{li}",
                "quant",
                [f"relu{li}"],
                {"step": float(aq.step.detach()), "bits": 4, "signed": False},
            )
        ).name
        li += 1
    g.add(Node("flatten", "flatten", [prev]))
    fc = model.fc
    if hasattr(fc, "w_step"):  # quantised readout: spike inputs, scores from U
        w_int = torch.round(fc.weight.detach().double() / fc.w_step.view(-1, 1))
        g.add(
            Node(
                "fc",
                "linear",
                ["flatten"],
                {
                    "w_int": w_int,
                    "step_w": fc.w_step,
                    "bits": fc.w_bits,
                    "bias": fc.bias.detach().double(),
                    "readout": True,
                },
            )
        )
    else:
        g.add(
            Node(
                "fc",
                "linear",
                ["flatten"],
                {"w": fc.weight.detach().double(), "bias": fc.bias.detach().double()},
                digital=True,
            )
        )
    g.add(Node("out", "output", ["fc"]))
    return g


def convert(graph, T=T_CNN):
    """Fold bn / relu / quant into a ttfs_neuron after each conv."""
    out, s_in = Graph(), None
    rename = lambda names: [f"neuron{i[len('quant'):]}" if i.startswith("quant") else i for i in names]
    for n in graph.nodes:
        if n.op in ("bn", "relu"):
            continue
        if n.op == "quant":
            li = n.name[len("quant") :]
            conv, bn = graph[f"conv{li}"], graph[f"bn{li}"]
            a, b = bn.attrs["a"], bn.attrs["b"]
            ga = a * conv.attrs["step_w"]  # LSQ steps can be negative too
            sgn = torch.where(ga < 0, -1.0, 1.0).double()
            w = conv.attrs["w_int"] * sgn.view(-1, 1, 1, 1)
            assert n.attrs["step"] > 0 and (s_in is None or s_in > 0)
            scale = ga.abs() * (1.0 if conv.digital else s_in)
            v = torch.arange(1, T + 1, dtype=torch.float64)
            theta = ((v.view(1, -1) - 0.5) * n.attrs["step"] - b.view(-1, 1)) / scale.view(-1, 1)
            out.add(Node(conv.name, "conv", rename(conv.inputs), dict(conv.attrs, w_int=w), conv.digital))
            out.add(
                Node(
                    f"neuron{li}",
                    "ttfs_neuron",
                    [conv.name],
                    {
                        "T": T,
                        "signed": False,
                        "scale": scale,
                        "bias": b,
                        "step_out": n.attrs["step"],
                        "theta": theta,
                    },
                )
            )
            s_in = n.attrs["step"]
            continue
        if n.op == "conv":
            continue  # emitted with its neuron
        out.add(Node(n.name, n.op, rename(n.inputs), dict(n.attrs), n.digital))
    return out


def run_ttfs(graph, x):
    """Integer TTFS semantics.  Returns ({neuron name: output levels}, logits)."""
    vals, levels, step = {}, {}, None
    for n in graph.nodes:
        src = vals.get(n.inputs[0]) if n.inputs else None
        if n.op == "input":
            vals[n.name] = x.double()
        elif n.op == "conv":
            vals[n.name] = F.conv2d(src, n.attrs["w_int"], None, 1, n.attrs["pad"])
        elif n.op == "ttfs_neuron":
            U = src.unsqueeze(-1)
            L = (U >= n.attrs["theta"].view(1, -1, 1, 1, n.attrs["T"])).sum(-1)
            levels[n.name] = L
            vals[n.name] = L.double()  # next layer sees integer levels q
            step = n.attrs["step_out"]
        elif n.op == "pool_max":
            vals[n.name] = F.max_pool2d(src, n.attrs["k"])  # max level = earliest spike
        elif n.op == "flatten":
            vals[n.name] = torch.flatten(src, 1)
        elif n.op == "linear" and "w_int" in n.attrs:  # readout: integer U, no firing
            U = src @ n.attrs["w_int"].t()
            vals[n.name] = U * (n.attrs["step_w"] * step) + n.attrs["bias"]
        elif n.op == "linear":
            vals[n.name] = (src * step) @ n.attrs["w"].t() + n.attrs["bias"]
        elif n.op == "output":
            vals[n.name] = src
    return levels, vals["out"]


def qnn_levels(model, x):
    """Per-layer output levels (pre-pool) and logits of the PyTorch QNN."""
    levels = {}

    def hook(li, blk, h):
        y = blk["aq"](F.relu(blk["bn"](blk["conv"](h))))
        levels[f"neuron{li}"] = torch.round(y / blk["aq"].step.detach()).long()
        return y

    model.eval()
    with torch.no_grad():
        logits = model(x, hook=hook)
    return levels, logits


def verify(model, graph, x):
    """Mismatching output levels vs the PyTorch QNN, per layer (isolated) and through the network (chained)."""
    ref, ref_logits = qnn_levels(model, x)
    inputs = {}

    def hook(li, blk, h):
        inputs[f"neuron{li}"] = h.detach().double()
        return blk["aq"](F.relu(blk["bn"](blk["conv"](h))))

    with torch.no_grad():
        model(x, hook=hook)
    iso, dist, s_prev = {}, {}, None
    for n in graph.of("ttfs_neuron"):
        conv = graph[n.inputs[0]]
        h = inputs[n.name] if conv.digital else torch.round(inputs[n.name] / s_prev)
        U = F.conv2d(h, conv.attrs["w_int"], None, 1, conv.attrs["pad"])
        L = (U.unsqueeze(-1) >= n.attrs["theta"].view(1, -1, 1, 1, n.attrs["T"])).sum(-1)
        bad = L != ref[n.name]
        V = (n.attrs["scale"].view(1, -1, 1, 1) * U + n.attrs["bias"].view(1, -1, 1, 1)) / n.attrs["step_out"]
        frac = (V - 0.5) - torch.floor(V - 0.5)
        iso[n.name] = int(bad.sum())
        dist[n.name] = float(torch.minimum(frac, 1 - frac)[bad].max()) if bad.any() else 0.0
        s_prev = n.attrs["step_out"]
    got, logits = run_ttfs(graph, x)
    return {
        "isolated_mismatch": iso,
        "isolated_max_boundary_dist": dist,
        "total": {k: ref[k].numel() for k in ref},
        "chained_mismatch": {k: int((ref[k] != got[k]).sum()) for k in ref},
        "logit_maxdiff": float((ref_logits.double() - logits).abs().max()),
        "same_prediction": float((ref_logits.argmax(1) == logits.argmax(1)).float().mean()),
    }


def _spiking_inputs(graph, x):
    """{node name: (integer input levels, T)} for every node that receives spikes:
    non-digital convs and the quantised readout; plus the network's own values."""
    levels, _ = run_ttfs(graph, x)
    vals, out, T = {}, {}, None
    for n in graph.nodes:
        if n.op == "ttfs_neuron":
            vals[n.name], T = levels[n.name].double(), n.attrs["T"]
        elif n.op == "pool_max":
            vals[n.name] = F.max_pool2d(vals[n.inputs[0]], n.attrs["k"])
        elif n.op == "flatten":
            vals[n.name] = torch.flatten(vals[n.inputs[0]], 1)
        if (n.op == "conv" and not n.digital) or (n.op == "linear" and "w_int" in n.attrs):
            out[n.name] = (vals[n.inputs[0]].long(), T)
    return out


def _write_trace(path, q, T):
    s, idx = torch.nonzero(q.flatten(1) > 0, as_tuple=True)
    tick = (T + 1 - q.flatten(1)[s, idx]).to(torch.int8)
    order = np.lexsort((idx.numpy(), tick.numpy(), s.numpy()))
    np.savez_compressed(
        path,
        sample=s.numpy()[order].astype(np.int32),
        tick=tick.numpy()[order],
        index=idx.numpy()[order].astype(np.int32),
        sign=np.ones(len(order), np.int8),
        shape=np.array(q.shape[1:]),
        T=T,
    )


def export_traces(graph, x, out_dir):
    """Spike traces of the input of every spiking-input layer: convs after the first,
    and the classifier when it is a quantised readout."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name, (q, T) in _spiking_inputs(graph, x).items():
        path = os.path.join(out_dir, f"{name}_input.npz")
        _write_trace(path, q, T)
        files.append(path)
    return files


def decode_trace(path):
    """Trace file -> dense signed level tensor [samples, *shape]; checks the file on the way."""
    t = np.load(path)
    T, shape = int(t["T"]), tuple(int(s) for s in t["shape"])
    n = int(t["sample"].max()) + 1 if len(t["sample"]) else 0
    key = t["sample"].astype(np.int64) * (T + 1) + t["tick"]
    assert np.all(np.diff(key) >= 0), "not sorted by (sample, tick)"
    assert len(t["tick"]) == 0 or (t["tick"].min() >= 1 and t["tick"].max() <= T), "tick out of range"
    assert set(np.unique(t["sign"])) <= {-1, 1}, "sign must be +-1"
    flat = t["sample"].astype(np.int64) * int(np.prod(shape)) + t["index"]
    assert len(np.unique(flat)) == len(flat), "a neuron spikes more than once"
    q = np.zeros(n * int(np.prod(shape)), np.int64)
    q[flat] = t["sign"].astype(np.int64) * (T + 1 - t["tick"].astype(np.int64))
    return torch.from_numpy(q.reshape((n,) + shape))


def verify_traces(model, graph, x, out_dir):
    """Decode every exported trace and compare it with (a) the integer TTFS network's
    own layer inputs (must be identical) and (b) the PyTorch QNN's layer inputs (differ
    only by cascades of the float32 rounding ties reported by verify())."""
    files = export_traces(graph, x, out_dir)
    own = _spiking_inputs(graph, x)
    real, steps = {}, {}

    def hook(li, blk, h):
        real[f"conv{li}"] = h.detach().double()
        steps[f"conv{li + 1}"] = float(blk["aq"].step.detach())
        return blk["aq"](F.relu(blk["bn"](blk["conv"](h))))

    fc_hook = model.fc.register_forward_hook(lambda mod, a, o: real.__setitem__("fc", a[0].detach().double()))
    model.eval()
    with torch.no_grad():
        model(x, hook=hook)
    fc_hook.remove()
    steps["fc"] = steps[f"conv{len(graph.of('ttfs_neuron'))}"]
    report = {}
    for path in files:
        name = os.path.basename(path)[: -len("_input.npz")]
        ref = torch.round(real[name] / steps[name]).long()  # QNN levels entering this layer
        got = decode_trace(path).reshape(ref.shape)
        report[name] = {
            "spikes": int((got != 0).sum()),
            "mismatch": int((got != ref).sum()),
            "nonzero_ref": int((ref != 0).sum()),
            "mismatch_vs_ttfs": int((got != own[name][0].reshape(ref.shape)).sum()),
        }
    return report
