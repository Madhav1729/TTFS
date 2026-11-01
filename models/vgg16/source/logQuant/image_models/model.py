"""CIFAR VGG16-BN, matching chenyaofo/pytorch-cifar-models.

Architecture adapted from upstream vgg.py at commit
786c16252c0fc58ee9adac063f8337cc4a7a497a. See UPSTREAM_LICENSE.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..modeling import QuantizedLinear
from ..quantization import ActivationQuantizer


class VGG16BN(nn.Module):
    def __init__(self):
        super().__init__()
        configuration = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                         512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        channels = 3
        for output in configuration:
            if output == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([nn.Conv2d(channels, output, 3, padding=1),
                               nn.BatchNorm2d(output), nn.ReLU(inplace=True)])
                channels = output
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
                                        nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
                                        nn.Linear(512, 100))

    def forward(self, images):
        return self.classifier(torch.flatten(self.features(images), 1))


class QuantizedConv2d(nn.Module):
    def __init__(self, original, name, signed):
        super().__init__()
        if original.padding_mode != "zeros":
            raise ValueError("Only zero-padded convolutions are supported")
        self.weight, self.bias = original.weight, original.bias
        self.stride, self.padding = original.stride, original.padding
        self.dilation, self.groups = original.dilation, original.groups
        self.quantizer = ActivationQuantizer(name, signed=signed)

    def forward(self, inputs):
        return F.conv2d(self.quantizer(inputs), self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def instrument_image(model):
    """Quantize Conv/Linear inputs; normalized RGB is signed, later inputs ReLU-positive."""
    if any(isinstance(m, ActivationQuantizer) for m in model.modules()):
        raise ValueError("Model has already been instrumented")
    first_convolution = True
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            replacement = QuantizedConv2d(module, name + ".input", signed=first_convolution)
            first_convolution = False
        elif isinstance(module, nn.Linear):
            replacement = QuantizedLinear(module, name + ".input")
            replacement.quantizer.signed = False
        else:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
    return {m.name: m for m in model.modules() if isinstance(m, ActivationQuantizer)}
