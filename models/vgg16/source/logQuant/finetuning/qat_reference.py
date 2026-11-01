"""Gradient-tested QAT building blocks; no model trainer is launched here."""

import torch
from torch.nn import functional as F

from ..quantization import quantize


def activation_ste(value, scheme, bits, e_max, signed=True):
    decoded = quantize(value, scheme, bits, e_max, signed)
    # Exact quantized forward values; the identity derivative is a surrogate.
    return decoded.detach() + (value - value.detach())


def weight_ste(weight, bits=16):
    if bits not in (8, 16) or weight.ndim not in (2, 4):
        raise ValueError("Use INT8/INT16 matrix or Conv2d weights")
    qmax = (1 << (bits - 1)) - 1
    maximum = weight.detach().flatten(1).abs().amax(1)
    maximum = maximum.reshape((-1,) + (1,) * (weight.ndim - 1))
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / qmax)
    decoded = (weight.detach() / scale).round().clamp(-qmax, qmax) * scale
    return decoded + (weight - weight.detach())


def distillation_loss(student_logits, teacher_logits, labels, alpha=.5, temperature=2.):
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError("alpha must be in [0,1] and temperature positive")
    supervised = F.cross_entropy(student_logits, labels)
    teacher = (teacher_logits.detach() / temperature).softmax(-1)
    student = (student_logits / temperature).log_softmax(-1)
    imitation = F.kl_div(student, teacher, reduction="batchmean") * temperature**2
    return (1 - alpha) * supervised + alpha * imitation
