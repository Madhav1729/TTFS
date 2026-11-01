"""BERT instrumentation: replace dot-product operands, preserve the rest.

The placement follows Otters: linear inputs, projected queries, and softmax
probabilities. Unlike Otters, standard checkpoint weights, GELU, and K/V are
preserved by default, isolating the effect of the activation codebook.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import ActivationQuantizer, pow2_matmul


class QuantizedLinear(nn.Module):
    def __init__(self, original, name):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.quantizer = ActivationQuantizer(name, signed=True)
        self.arithmetic = "dense"

    def forward(self, x):
        q = self.quantizer(x)
        if self.arithmetic == "shift" and self.quantizer.mode == "log":
            out = pow2_matmul(q, self.weight.T)
            return out if self.bias is None else out + self.bias
        return F.linear(q, self.weight, self.bias)


class QuantizedSelfAttention(nn.Module):
    """BERT eager encoder attention; no decoder cache or relative positions."""

    def __init__(self, original, name, kv_mode="float"):
        super().__init__()
        if original.is_decoder or original.position_embedding_type != "absolute":
            raise ValueError("Only BERT encoder attention with absolute positions is supported")
        self.query, self.key, self.value = original.query, original.key, original.value
        self.dropout = original.dropout
        self.num_attention_heads = original.num_attention_heads
        self.attention_head_size = original.attention_head_size
        self.all_head_size = original.all_head_size
        self.query_quantizer = ActivationQuantizer(name + ".query_operand", signed=True)
        self.prob_quantizer = ActivationQuantizer(name + ".probability_operand", signed=False)
        self.kv_mode, self.arithmetic = kv_mode, "dense"

    def transpose_for_scores(self, x):
        x = x.view(*x.shape[:-1], self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise ValueError("Cross attention / decoder caching is not supported")
        q = self.query_quantizer(self.transpose_for_scores(self.query(hidden_states)))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        if self.kv_mode == "binary":
            k, v = k.sign(), v.sign()
        score_op = pow2_matmul if self.arithmetic == "shift" and self.query_quantizer.mode == "log" else torch.matmul
        scores = score_op(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = self.dropout(F.softmax(scores, dim=-1))
        if head_mask is not None:
            probs = probs * head_mask
        probs = self.prob_quantizer(probs)
        context_op = pow2_matmul if self.arithmetic == "shift" and self.prob_quantizer.mode == "log" else torch.matmul
        context = context_op(probs, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.shape[:-2], self.all_head_size)
        return (context, probs) if output_attentions else (context,)


def instrument(model, scope="all", kv_mode="float"):
    """Mutate a Hugging Face BertForSequenceClassification in place.

    Includes pooler/classifier linear inputs. All linear inputs are signed,
    since standard BERT uses GELU and can produce negative FFN activations.
    Only probabilities use an unsigned codebook. No weight quantization.
    """
    if model.config.model_type != "bert":
        raise ValueError("This first experiment supports BERT sequence classifiers only")
    if scope not in ("linear", "attention", "all") or kv_mode not in ("float", "binary"):
        raise ValueError("Unsupported scope or K/V mode")
    if model.config.is_decoder or getattr(model.config, "position_embedding_type", "absolute") != "absolute":
        raise ValueError("Only absolute-position BERT encoders are supported")
    if any(isinstance(m, ActivationQuantizer) for m in model.modules()):
        raise ValueError("Model has already been instrumented")
    if scope in ("attention", "all") or kv_mode == "binary":
        for index, layer in enumerate(model.bert.encoder.layer):
            layer.attention.self = QuantizedSelfAttention(
                layer.attention.self, f"bert.encoder.layer.{index}.attention.self", kv_mode)
    if scope in ("linear", "all"):
        # Snapshot before replacing modules; avoid recursively wrapping wrappers.
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, QuantizedLinear(module, name + ".input"))
    quantizers = {m.name: m for m in model.modules() if isinstance(m, ActivationQuantizer)}
    if scope == "linear":
        # Binary K/V alone may have installed attention wrappers. Their
        # query/probability quantizers stay pass-through in this ablation.
        quantizers = {k: v for k, v in quantizers.items() if k.endswith(".input")}
    return quantizers


def set_arithmetic(model, arithmetic):
    if arithmetic not in ("dense", "shift"):
        raise ValueError("arithmetic must be dense or shift")
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedSelfAttention)):
            module.arithmetic = arithmetic
