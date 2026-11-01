"""Symmetric row-wise integer weights with floating-point reference execution."""

from pathlib import Path

import torch
from torch import nn


@torch.no_grad()
def quantize_weights(model, bits):
    """Quantize Linear/Embedding/Conv2d weights; keep other state in FP32.

    Each output row has its own scale: decoded_weight = integer * scale.
    The model executes the decoded values, isolating weight rounding error
    without claiming an integer kernel or fixed-point accumulator simulation.
    """
    if bits not in (8, 16):
        raise ValueError("Weight quantization supports 8 or 16 bits")
    qmax = (1 << (bits - 1)) - 1
    dtype = torch.int8 if bits == 8 else torch.int16
    packed = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            continue
        weight = module.weight.detach().float()
        if not torch.isfinite(weight).all():
            raise ValueError(f"Non-finite weights in {name}")
        maximum = weight.flatten(1).abs().amax(dim=1)
        maximum = maximum.reshape((-1,) + (1,) * (weight.ndim - 1))
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / qmax)
        integer = (weight / scale).round().clamp(-qmax, qmax).to(dtype)
        decoded = integer.float() * scale
        packed[name + ".weight"] = dict(values=integer.cpu(), scale=scale.cpu())
        module.weight.copy_(decoded)
    if not packed:
        raise ValueError("No Linear, Embedding or Conv2d weights found")
    float_state = {name: value.detach().cpu().clone()
                   for name, value in model.state_dict().items() if name not in packed}
    return dict(format="logquant_rowwise_weights_v1", bits=bits, axis=0,
                matrices=packed, float_state=float_state)


def save_weights(bundle, directory, model, tokenizer, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(dict(bundle, metadata=metadata), directory / "weights.pt")
    model.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


@torch.no_grad()
def load_weights(model, path):
    """Restore a bundle into a matching ordinary, uninstrumented model."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "logquant_rowwise_weights_v1" or bundle.get("bits") not in (8, 16):
        raise ValueError("Unsupported integer-weight checkpoint")
    expected_dtype = torch.int8 if bundle["bits"] == 8 else torch.int16
    state = dict(bundle["float_state"])
    for name, entry in bundle["matrices"].items():
        values, scale = entry["values"], entry["scale"]
        if values.ndim not in (2, 4):
            raise ValueError(f"Invalid quantized matrix: {name}")
        expected_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        if values.dtype != expected_dtype or scale.shape != expected_shape:
            raise ValueError(f"Invalid quantized matrix: {name}")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"Invalid scale: {name}")
        if name in state:
            raise ValueError(f"Duplicate checkpoint tensor: {name}")
        state[name] = values.float() * scale
    model.load_state_dict(state, strict=True)
    return bundle["bits"]
