"""Architecture parity, convolution weight export and eight-bit codebook checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from ..quantization import encode_log, levels
from ..weights import load_weights, quantize_weights
from .data import MEAN, STD, preprocess
from .model import VGG16BN, instrument_image


class ImageTests(unittest.TestCase):
    def test_vgg_shape_instrumentation_parity_and_signedness(self):
        torch.manual_seed(3)
        model = VGG16BN().eval()
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 13)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.modules()), 3)
        inputs = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            original = model(inputs)
            quantizers = instrument_image(model)
            actual = model(inputs)
            self.assertEqual(actual.shape, (2, 100))
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
            self.assertEqual(len(quantizers), 16)
            self.assertTrue(quantizers["features.0.input"].signed)
            self.assertEqual(sum(q.signed for q in quantizers.values()), 1)
            for q in quantizers.values():
                q.observing = True
            model(inputs)
            for q in quantizers.values():
                q.observing = False
                q.configure("log", 8, range(-2, 2))
                q.recording = False
            self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_conv_weight_roundtrip_and_batchnorm_preservation(self):
        def make():
            return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).eval()
        model = make()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        bundle = quantize_weights(model, 16)
        entry = bundle["matrices"]["0.weight"]
        self.assertEqual(entry["values"].dtype, torch.int16)
        self.assertEqual(entry["scale"].shape, (4, 1, 1, 1))
        for name in before:
            if name != "0.weight":
                torch.testing.assert_close(model.state_dict()[name], before[name], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            torch.save(bundle, path)
            restored = make()
            load_weights(restored, path)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)

    def test_eight_bit_log_codes_preserve_representable_small_powers(self):
        x = torch.tensor([0., 2.**-140, 2.**-60, 2.**-20, .5, 1., 2.])
        code = encode_log(x, 8, 1, signed=False)
        self.assertEqual(levels(8, False), 255)
        self.assertEqual(levels(8, True), 127)
        torch.testing.assert_close(code.decode(), x, atol=0, rtol=0)
        self.assertEqual(int(code.times()[-1]), 0)
        self.assertEqual(int(code.times()[0]), 255)

    def test_preprocessing_uses_author_normalization(self):
        output = preprocess(np.zeros((1, 3, 32, 32), dtype=np.uint8), "cpu")
        expected = -torch.tensor(MEAN) / torch.tensor(STD)
        torch.testing.assert_close(output[0, :, 0, 0], expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
