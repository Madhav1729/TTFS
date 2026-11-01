"""Firing-tick histogram of every quantised activation of the Otters BERT (RTE dev) -> data/lm."""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

CURRENT = {"name": "?"}
HIST = collections.defaultdict(lambda: None)


def install_tick_logger(steps):
    import transformer.ttfs_function as tf
    import transformer.utils_quant_ttfs as uq

    original = tf.ttfs_spike_quantization_vectorized

    def logged(x, alpha, t_, delta):
        if t_ == steps:
            th = torch.tensor(
                [alpha * (t_ - k - (t_ + 1) / 2) for k in range(t_)], dtype=x.dtype, device=x.device
            )
            cleared = x.detach().reshape(-1, 1) >= th.reshape(1, -1)
            silent = torch.full((cleared.shape[0],), t_, dtype=torch.long, device=x.device)
            ticks = torch.where(cleared.any(1), cleared.float().argmax(1), silent)
            counts = torch.bincount(ticks, minlength=t_ + 1).cpu().numpy()
            name = CURRENT["name"]
            HIST[name] = counts if HIST[name] is None else HIST[name] + counts
        return original(x, alpha, t_, delta)

    tf.ttfs_spike_quantization_vectorized = logged
    uq.ttfs_spike_quantization_vectorized = logged


def load_model():
    from transformer.configuration_bert import BertConfig
    from transformer.modeling_bert_quant_ttfs import BertForSequenceClassification

    cfg = BertConfig.from_pretrained(str(HERE / "checkpoint"))
    model = BertForSequenceClassification.from_pretrained(str(HERE / "checkpoint"), config=cfg).eval()
    for name, module in model.named_modules():
        if type(module).__name__ == "QuantizeLinear":
            module.register_forward_pre_hook(lambda m, i, n=name: CURRENT.__setitem__("name", n))
    return model, 2**cfg.input_bits - 1


def dev_batches(count, size=8):
    from transformer.tokenization import BertTokenizer
    from utils_glue import convert_examples_to_features, output_modes, processors

    tok = BertTokenizer.from_pretrained(str(HERE / "code" / "models" / "RTE"), do_lower_case=True)
    proc = processors["rte"]()
    examples = proc.get_dev_examples(str(HERE / "glue_data" / "RTE"))[: count * size]
    feats = convert_examples_to_features(examples, proc.get_labels(), 128, tok, output_modes["rte"])
    for i in range(0, len(feats), size):
        chunk = feats[i : i + size]
        yield tuple(
            torch.tensor([getattr(f, k) for f in chunk]) for k in ("input_ids", "segment_ids", "input_mask")
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "data" / "lm" / "profile_otters.json")
    a = ap.parse_args()
    model, steps = load_model()
    install_tick_logger(steps)
    with torch.no_grad():
        for ids, seg, mask in dev_batches(a.batches):
            model(ids, seg, mask)
    hist = {k: v.tolist() for k, v in HIST.items()}
    a.out.write_text(json.dumps({"T": steps, "task": "RTE", "split": "dev", "hist": hist}))
    total = np.sum([np.array(v) for v in hist.values()], axis=0)
    print(f"{len(hist)} sites -> {a.out}; silent fraction {total[-1] / total.sum():.4f}")


if __name__ == "__main__":
    main()
