"""Finite power-of-two codebooks and numerical/bitwise arithmetic references.

Bit counts include a sign bit at signed sites and a reserved zero magnitude
code. Magnitude codes 1..L represent consecutive integer exponents. This is
an explicit signed/zero-aware adaptation of arXiv:1603.01025, not a verbatim
implementation of its unsigned CNN experiment.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


def levels(bits: int, signed: bool) -> int:
    if not 2 <= bits <= 16:
        raise ValueError("Use 2..16 total activation bits in this experiment")
    return (1 << (bits - int(signed))) - 1


@dataclass
class SpikeCode:
    exponent: torch.Tensor
    sign: torch.Tensor
    active: torch.Tensor
    e_max: int
    nonzero_levels: int

    def times(self) -> torch.Tensor:
        """Logical TTFS slots 0..L-1; L is the no-spike sentinel."""
        return torch.where(self.active, self.e_max - self.exponent,
                           self.nonzero_levels)

    def decode(self) -> torch.Tensor:
        magnitude = torch.ldexp(torch.ones_like(self.exponent, dtype=torch.float32),
                                self.exponent)
        value = torch.where(self.sign < 0, -magnitude, magnitude)
        return torch.where(self.active, value, 0.0)


def encode_log(x, bits, e_max, signed=True, rounding="nearest"):
    n = levels(bits, signed)
    if rounding not in ("nearest", "floor"):
        raise ValueError("rounding must be nearest or floor")
    e_min = e_max - n + 1
    if not -126 <= e_max <= 126:
        raise ValueError("Largest codebook magnitude must be a normal float32 value")
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite activation")
    magnitude = x.abs() if signed else x.clamp_min(0)
    # Wide log codebooks can extend below float32. Encode only
    # observed values; unused tiny levels need not be materialized. Preserve
    # representable subnormals instead of rounding them up to the normal range.
    smallest = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    log = torch.log2(magnitude.clamp_min(smallest))
    # Round in LOG space (boundary sqrt(2)), not to the closest linear value.
    exponent = torch.floor(log + 0.5) if rounding == "nearest" else torch.floor(log)
    active = (magnitude > 0) & (exponent >= e_min)
    exponent = exponent.clamp(e_min, e_max).to(torch.int32)
    return SpikeCode(exponent, x.sign().to(torch.int8), active, int(e_max), n)


def quantize(x, mode, bits, e_max, signed=True, rounding="nearest"):
    if mode == "none":
        return x
    if mode == "log":
        return encode_log(x, bits, e_max, signed, rounding).decode().to(x.dtype)
    if mode != "uniform":
        raise ValueError(f"Unknown quantization mode: {mode}")
    n = levels(bits, signed)
    magnitude = x.abs() if signed else x.clamp_min(0)
    step = math.ldexp(1.0, e_max) / n
    q = torch.floor(magnitude / step + 0.5).clamp(0, n) * step
    return torch.where(x < 0, -q, q) if signed else q


def pow2_matmul(left, right):
    """Reference sum of signed exponent adjustments, without torch.matmul.

    The left operand MUST contain only zero or signed powers of two. Right
    operands and the accumulator are floats: this validates mathematical
    shift equivalence, not fixed-point overflow/rounding or GPU speed.
    """
    if left.ndim < 2 or right.ndim < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError("Expected compatible matrix/batched-matrix operands")
    mantissa, exponent = torch.frexp(left.abs())
    if not torch.all((left == 0) | (mantissa == 0.5)):
        raise ValueError("Left operand must be zero or signed powers of two")
    shape = (*torch.broadcast_shapes(left.shape[:-2], right.shape[:-2]),
             left.shape[-2], right.shape[-1])
    out = torch.zeros(shape, device=left.device, dtype=torch.promote_types(left.dtype, right.dtype))
    for k in range(left.shape[-1]):
        value = left[..., :, k, None]
        shift = exponent[..., :, k, None] - 1
        term = torch.ldexp(right[..., k, :].unsqueeze(-2), shift)
        term = torch.where(value < 0, -term, term)
        out = out + torch.where(value != 0, term, 0.0)
    return out


def integer_shift_linear(code, weight_int, weight_fraction_bits=0, accumulator_bits=63):
    """Exact integer left-shift/add reference with a common output binary point.

    Weight value = weight_int * 2**(-weight_fraction_bits).
    Returns (integer accumulator, output exponent); real output is
    accumulator * 2**output_exponent. Alignment to the smallest active
    exponent avoids lossy per-product right shifts. A conservative bound
    rejects potential overflow BEFORE performing any shifts.
    """
    if weight_int.dtype != torch.int64 or weight_int.ndim != 2:
        raise ValueError("Weights must be an [out, in] int64 tensor")
    if code.exponent.shape[-1] != weight_int.shape[-1]:
        raise ValueError("Input/weight dimension mismatch")
    if not 2 <= accumulator_bits <= 63:
        raise ValueError("accumulator_bits must be 2..63")
    out = torch.zeros((*code.exponent.shape[:-1], weight_int.shape[0]),
                      dtype=torch.int64, device=weight_int.device)
    if not code.active.any():
        return out, -weight_fraction_bits
    e_min = int(code.exponent[code.active].min())
    e_max = int(code.exponent[code.active].max())
    # Python integers avoid overflowing while computing the overflow bound.
    max_weight = max(abs(int(weight_int.min())), abs(int(weight_int.max())))
    max_terms = int(code.active.sum(dim=-1).max())
    bound = (max_weight << (e_max - e_min)) * max_terms
    if e_max - e_min > 62 or bound > (1 << (accumulator_bits - 1)) - 1:
        raise OverflowError("Shift/accumulation may overflow; increase precision or narrow the exponent range")
    for k in range(weight_int.shape[-1]):
        active = code.active[..., k, None]
        shift = torch.where(active, code.exponent[..., k, None] - e_min, 0).long()
        term = torch.bitwise_left_shift(weight_int[:, k], shift)
        term = torch.where(code.sign[..., k, None] < 0, -term, term)
        out = out + torch.where(active, term, 0)
    return out, e_min - weight_fraction_bits


class ActivationQuantizer(nn.Module):
    def __init__(self, name, signed=True, sample_limit=8192):
        super().__init__()
        self.name, self.signed = name, signed
        self.sample_limit = sample_limit
        self.mode, self.bits, self.e_max, self.rounding = "none", 4, 0, "nearest"
        self.observing = False
        self.recording = False
        self.samples = []
        self.sample_count = 0
        self.sampling = "random_extrema"
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(elements=0, input_nonzero=0, output_zero=0,
                          underflow=0, above_range=0, squared_error=0.0)

    def forward(self, x):
        if self.observing:
            flat = x.detach().reshape(-1)
            count = min(512, flat.numel(), self.sample_limit - self.sample_count)
            if count > 0:
                if self.sampling == "legacy_linspace":
                    indices = torch.linspace(0, flat.numel() - 1, count, device=x.device).long()
                    sample = flat[indices]
                elif self.sampling == "random_extrema":
                    # Repeated flattened positions can miss fixed outlier channels
                    # forever. Refresh samples and retain observed batch extremes.
                    low, high = torch.aminmax(flat)
                    if count == 1:
                        sample = torch.where(low.abs() > high.abs(), low, high).reshape(1)
                    else:
                        indices = torch.randint(flat.numel(), (count - 2,), device=x.device)
                        sample = torch.cat((low.reshape(1), high.reshape(1), flat[indices]))
                else:
                    raise ValueError(f"Unknown calibration sampling: {self.sampling}")
                self.samples.append(sample.float().cpu())
                self.sample_count += count
            return x
        q = quantize(x, self.mode, self.bits, self.e_max, self.signed, self.rounding)
        if self.recording:
            s = self.stats
            s["elements"] += x.numel()
            s["input_nonzero"] += int(torch.count_nonzero(x))
            s["output_zero"] += int((q == 0).sum())
            s["underflow"] += int(((x != 0) & (q == 0)).sum())
            s["above_range"] += int((x.abs() > math.ldexp(1.0, self.e_max)).sum())
            s["squared_error"] += float((x.float() - q.float()).square().sum())
        return q

    def configure(self, mode, bits, offsets, rounding="nearest", range_offset=0):
        self.mode, self.bits, self.rounding = mode, bits, rounding
        self.reset_stats()
        self.recording = mode != "none"
        if mode == "none":
            return {}
        if not self.samples:
            raise RuntimeError(f"No calibration samples for {self.name}")
        sample = torch.cat(self.samples)
        if not torch.isfinite(sample).all():
            raise ValueError(f"Non-finite calibration sample at {self.name}")
        max_abs = float(sample.abs().max())
        anchor = math.ceil(math.log2(max_abs)) if max_abs else 0
        candidates = sorted(set(max(-32, min(32, anchor + d)) for d in offsets))
        losses = [float((quantize(sample, mode, bits, e, self.signed, rounding) - sample).square().mean())
                  for e in candidates]
        best = min(range(len(losses)), key=losses.__getitem__)
        self.e_max = candidates[best] + range_offset
        applied_mse = float((quantize(sample, mode, bits, self.e_max, self.signed, rounding) - sample).square().mean())
        return dict(signed=self.signed, total_bits=bits, nonzero_magnitude_levels=levels(bits, self.signed),
                    e_max=self.e_max, e_min_log=self.e_max - levels(bits, self.signed) + 1,
                    calibration_mse=applied_mse, fitted_calibration_mse=losses[best], calibration_samples=sample.numel(),
                    fitted_e_max=candidates[best], range_offset=range_offset)
