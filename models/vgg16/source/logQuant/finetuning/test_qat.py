"""Verify the proposed surrogate and data split before writing a full trainer."""

import unittest

import numpy as np
import torch

from ..quantization import quantize
from ..weights import quantize_weights
from .prepare_data import stratified_cifar_split
from .qat_reference import activation_ste, distillation_loss, weight_ste


class QATTests(unittest.TestCase):
    def test_actual_quantized_forward_with_nonzero_surrogate_gradient(self):
        for scheme in ("uniform", "log"):
            value = torch.tensor([0., .13, 1.4, -2.1, 100.], requires_grad=True)
            output = activation_ste(value, scheme, 4, 2)
            torch.testing.assert_close(output, quantize(value, scheme, 4, 2), atol=0, rtol=0)
            output.sum().backward()
            torch.testing.assert_close(value.grad, torch.ones_like(value))

    def test_weight_surrogate_matches_export_values_and_keeps_master(self):
        layer = torch.nn.Conv2d(3, 4, 3)
        original = layer.weight.detach().clone()
        decoded = weight_ste(layer.weight)
        decoded.sum().backward()
        torch.testing.assert_close(layer.weight, original, atol=0, rtol=0)
        self.assertTrue((layer.weight.grad != 0).all())
        quantize_weights(layer, 16)
        torch.testing.assert_close(decoded, layer.weight, atol=0, rtol=0)

    def test_distillation_updates_student_only(self):
        student = torch.randn(3, 5, requires_grad=True)
        teacher = torch.randn(3, 5, requires_grad=True)
        loss = distillation_loss(student, teacher, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)

    def test_validation_is_stratified_and_excluded_from_training(self):
        labels = np.repeat(np.arange(100), 500)
        train, dev = stratified_cifar_split(labels)
        self.assertEqual((len(train), len(dev)), (45000, 5000))
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(train) | set(dev), set(range(50000)))
        self.assertTrue((np.bincount(labels[dev]) == 50).all())
        self.assertEqual((train, dev), stratified_cifar_split(labels))


if __name__ == "__main__":
    unittest.main()
