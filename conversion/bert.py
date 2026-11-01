"""Otters W1A4 BERT -> polarity TTFS per quantised linear (signed sites T = 8, unsigned FFN2 input T = 15)."""

import os
import sys
import numpy as np
import torch

from .ir import Graph, Node

OTTERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "otters")
T_POL = 8
SITES = {"query": "attention.self.query", "key": "attention.self.key", "value": "attention.self.value",
         "attn_out": "attention.output.dense", "ffn1": "intermediate.dense", "ffn2": "output.dense"}  # fmt: skip


def _otters_path():
    code = os.path.join(OTTERS, "code")
    if code not in sys.path:
        sys.path.insert(0, code)


def load():
    """Otters W1A4 BERT (RTE) with the exact QNN quantisers installed in place of the device response."""
    _otters_path()
    import transformer.ttfs_function as tf
    import transformer.utils_quant_ttfs as uq
    from transformer.configuration_bert import BertConfig
    from transformer.modeling_bert_quant_ttfs import BertForSequenceClassification

    path = os.path.join(OTTERS, "checkpoint")
    cfg = BertConfig.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path, config=cfg).eval()
    steps = 2**cfg.input_bits - 1
    original = tf.ttfs_spike_quantization_vectorized

    def exact_signed(
        x, alpha, t_, delta
    ):  # q = clamp(floor(x / alpha), -8, 7); x = alpha * (T s - (T + 1) / 2)
        if t_ != steps:
            return original(x, alpha, t_, delta)
        q = (x / alpha).floor().clamp(-(t_ // 2 + 1), t_ // 2)
        return (q + (t_ + 1) / 2.0) / t_

    def exact_unsigned(x, alpha, t_, delta):  # reconstruction is alpha * T * s
        return (x / alpha).floor().clamp(0, t_) / t_

    for module in (tf, uq):
        module.ttfs_spike_quantization_vectorized = exact_signed
        module.ttfs_spike_quantization_vectorized_nonbias = exact_unsigned
    return model, cfg, None


def rte_batch(n=8):
    """(input ids, segment ids, mask) of the first n RTE dev examples."""
    _otters_path()
    from transformer.tokenization import BertTokenizer
    from utils_glue import convert_examples_to_features, output_modes, processors

    tok = BertTokenizer.from_pretrained(os.path.join(OTTERS, "code", "models", "RTE"), do_lower_case=True)
    proc = processors["rte"]()
    examples = proc.get_dev_examples(os.path.join(OTTERS, "glue_data", "RTE"))[:n]
    feats = convert_examples_to_features(examples, proc.get_labels(), 128, tok, output_modes["rte"])
    return tuple(
        torch.tensor([getattr(f, k) for f in feats]) for k in ("input_ids", "segment_ids", "input_mask")
    )


def _alpha(m):
    from transformer.utils_quant_ttfs import find_closest_power_of_two

    return float(find_closest_power_of_two(max(float(m.input_clip_val.detach()), 1e-5)))


def import_bert(model):
    from transformer.utils_quant_ttfs import weight_quant_fn

    g = Graph()
    for li, layer in enumerate(model.bert.encoder.layer):
        for site, path in SITES.items():
            m = layer.get_submodule(path)
            with torch.no_grad():  # the model's own float32 binariser,
                Wq = weight_quant_fn(
                    m.weight,
                    m.weight_clip_val,
                    num_bits=m.weight_bits,  # so ties at
                    symmetric=m.symmetric,
                    quant_method=m.weight_quant_method,  # the mean
                    layerwise=m.weight_layerwise,
                ).double()  # match
            W = Wq.sign()
            assert m.weight_bits == 1 and m.input_bits == 4 and Wq.abs().max() > 0
            signed = bool(m.symmetric)
            name = f"L{li}.{site}"
            g.add(Node(f"{name}.in", "input", attrs={"features": W.shape[1]}))
            g.add(
                Node(
                    name,
                    "linear",
                    [f"{name}.in"],
                    {
                        "w_int": W,
                        "w_scale": float(Wq.abs().max()),
                        "bias": m.bias.detach().double(),
                        "alpha": _alpha(m),
                        "move": m.move.bias.detach().double(),
                        "signed": signed,
                        "qn": -8 if signed else 0,
                        "qp": 7 if signed else 15,
                        "T": T_POL if signed else 15,
                        "module": path,
                    },
                )
            )
    _add_site(g, "pooler", model.bert.pooler.dense, "bert.pooler.dense", weight_quant_fn)
    return g


def _add_site(g, name, m, path, weight_quant_fn):
    with torch.no_grad():
        Wq = weight_quant_fn(
            m.weight,
            m.weight_clip_val,
            num_bits=m.weight_bits,
            symmetric=m.symmetric,
            quant_method=m.weight_quant_method,
            layerwise=m.weight_layerwise,
        ).double()
    signed = bool(m.symmetric)
    g.add(Node(f"{name}.in", "input", attrs={"features": Wq.shape[1]}))
    g.add(
        Node(
            name,
            "linear",
            [f"{name}.in"],
            {
                "w_int": Wq.sign(),
                "w_scale": float(Wq.abs().max()),
                "bias": m.bias.detach().double(),
                "alpha": _alpha(m),
                "move": m.move.bias.detach().double(),
                "signed": signed,
                "qn": -8 if signed else 0,
                "qp": 7 if signed else 15,
                "T": T_POL if signed else 15,
                "module": path,
            },
        )
    )


LN_NOTE = ["shift-only: Σu, Σu² (bit-serial), 1/σ = 2 shifts", "γ, β folded into the next thresholds"]


def layer_graph(g, li):
    """One encoder layer as a connected graph: the six converted sites, the spiking
    attention matmuls, temporal softmax and the shift-only LayerNorm (bert_units)."""
    s = lambda site: g[f"L{li}.{site}"]
    lg = Graph()
    lg.add(Node("hidden", "input", attrs={"carries": "real values (LayerNorm output)"}))
    for site in ("query", "key", "value"):
        lg.add(Node(site, "linear", ["hidden"], dict(s(site).attrs, carries="real values")))
    lg.add(
        Node(
            "QK^T",
            "matmul",
            ["query", "key"],
            {
                "role": "weighted",
                "carries": "score spikes",
                "edge_labels": {"query": "Q spikes, polarity T=8", "key": "K stored (±1)"},
                "note": ["Q spikes (4-bit, T=8) into stored K (±1)", "S/U site; score tick = round(s / ln2)"],
            },
        )
    )
    lg.add(
        Node(
            "softmax",
            "softmax",
            ["QK^T"],
            {
                "role": "time",
                "carries": "P levels per tick bin",
                "note": [
                    "temporal: first spike = max, Z by doubling",
                    "one 4-bit level per tick bin (bert_units)",
                ],
            },
        )
    )
    lg.add(
        Node(
            "PV",
            "matmul",
            ["softmax", "value"],
            {
                "role": "weighted",
                "carries": "real values",
                "edge_labels": {"value": "V stored (±1)"},
                "note": ["sum of stored V rows (±1) per bin,", "times the bin's level (adds)"],
            },
        )
    )
    lg.add(Node("attn_out", "linear", ["PV"], dict(s("attn_out").attrs, carries="real values")))
    lg.add(Node("add1", "add", ["attn_out", "hidden"], {"carries": "real values", "note": ["residual"]}))
    lg.add(Node("LN1", "layernorm", ["add1"], {"carries": "real values", "note": LN_NOTE}))
    lg.add(Node("ffn1", "linear", ["LN1"], dict(s("ffn1").attrs, carries="real values")))
    lg.add(
        Node(
            "relu",
            "relu",
            ["ffn1"],
            {
                "role": "spiking",
                "carries": "real values",
                "note": [
                    "ReLU + ffn2's 4-bit unsigned quantiser",
                    "= neuron firing ffn1's outputs",
                    "(early decision acts here)",
                ],
            },
        )
    )
    lg.add(Node("ffn2", "linear", ["relu"], dict(s("ffn2").attrs, carries="real values")))
    lg.add(Node("add2", "add", ["ffn2", "LN1"], {"carries": "real values", "note": ["residual"]}))
    lg.add(Node("LN2", "layernorm", ["add2"], {"carries": "real values", "note": LN_NOTE}))
    lg.add(Node("next layer", "output", ["LN2"]))
    return lg


def capture(model, batch, ed=None):
    """{site name: (input, output)} for every quantised linear, one forward pass."""
    ids, seg, msk = batch
    rec, hooks = {}, []
    for li, layer in enumerate(model.bert.encoder.layer):
        for site, path in SITES.items():

            def hook(mod, args, out, key=f"L{li}.{site}"):
                rec[key] = (args[0].detach().double(), out.detach().double(), msk.bool())

            hooks.append(layer.get_submodule(path).register_forward_hook(hook))

    def pool_hook(mod, args, out):  # [CLS] only: make it a 1-token sequence
        x = args[0].detach().double().unsqueeze(1)
        rec["pooler"] = (x, out.detach().double().unsqueeze(1), torch.ones(x.shape[:2], dtype=torch.bool))

    hooks.append(model.bert.pooler.dense.register_forward_hook(pool_hook))
    if ed is not None:
        ed.STATE["mask"] = msk
    with torch.no_grad():
        logits = model(ids, seg, msk)
    for h in hooks:
        h.remove()
    rec["_mask"] = msk.bool()
    rec["_logits"] = logits[0] if isinstance(logits, tuple) else logits
    return rec


def levels(node, x):
    a = node.attrs
    return torch.clamp(torch.floor((x + a["move"]) / a["alpha"]), a["qn"], a["qp"])


def run_site(node, q):
    a = node.attrs
    U = q @ a["w_int"].t()  # exact integers
    return a["alpha"] * a["w_scale"] * U + a["bias"], U


def verify(graph, cap):
    """Per site: max |V_ttfs - y_qnn| relative to the output range, valid tokens only."""
    out = {}
    for n in graph.of("linear"):
        x, y, mask = cap[n.name]
        V, _ = run_site(n, levels(n, x))
        err = (V - y).abs()[mask]
        out[n.name] = float(err.max() / y[mask].abs().max())
    return out


def encode(q, T):
    """QNN level -> (tick, sign); tick 0 marks silence (q = 0)."""
    mag = q.abs()
    return torch.where(mag > 0, T + 1 - mag, torch.zeros_like(mag)).to(torch.int8), torch.sign(q).to(
        torch.int8
    )


def export_traces(graph, cap, out_dir, sites=None):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for n in graph.of("linear"):
        if sites and n.name.split(".", 1)[-1] not in sites:
            continue
        x, _, mask = cap[n.name]
        q = levels(n, x) * mask.unsqueeze(-1)  # padding tokens send nothing
        tick, sign = encode(q, n.attrs["T"])
        B, L, K = q.shape
        s, t, k = torch.nonzero(tick > 0, as_tuple=True)
        tk = tick[s, t, k]
        idx = (t * K + k).numpy()
        order = np.lexsort((idx, tk.numpy(), s.numpy()))
        path = os.path.join(out_dir, f"{n.name}_input.npz")
        np.savez_compressed(
            path,
            sample=s.numpy()[order].astype(np.int32),
            tick=tk.numpy()[order],
            index=idx[order].astype(np.int32),
            sign=sign[s, t, k].numpy()[order],
            shape=np.array([L, K]),
            T=n.attrs["T"],
            valid=mask.numpy(),
        )
        files.append(path)
    return files


def verify_traces(graph, cap, out_dir, sites=None):
    """Every exported trace, decoded, must equal the QNN's input levels (valid tokens)."""
    from .convert import decode_trace

    report = {}
    for path in export_traces(graph, cap, out_dir, sites):
        name = os.path.basename(path)[: -len("_input.npz")]
        n = graph[name]
        x, _, mask = cap[name]
        ref = (levels(n, x) * mask.unsqueeze(-1)).long()
        got = decode_trace(path)
        got = torch.cat([got, torch.zeros((ref.shape[0] - got.shape[0],) + got.shape[1:], dtype=got.dtype)])
        report[name] = {
            "spikes": int((got != 0).sum()),
            "mismatch": int((got != ref).sum()),
            "nonzero_ref": int((ref != 0).sum()),
        }
    return report
