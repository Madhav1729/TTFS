"""Arithmetic, adapter parity, and an offline CLI integration check."""

from pathlib import Path
import tempfile
import unittest

import torch

from .quantization import (ActivationQuantizer, encode_log, integer_shift_linear,
                           levels, pow2_matmul, quantize)
from .weights import load_weights, quantize_weights


class WeightTests(unittest.TestCase):
    def test_row_scales_zero_rows_and_retained_float_parameters(self):
        for bits in (8, 16):
            model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
            with torch.no_grad():
                model[0].weight.copy_(torch.tensor([[0., 0., 0., 0.],
                                                    [-1., .17, .4, 1.],
                                                    [-.001, .00032, .0001, .001]]))
            original = {k: v.clone() for k, v in model.state_dict().items()}
            bundle = quantize_weights(model, bits)
            entry = bundle["matrices"]["0.weight"]
            self.assertEqual(entry["values"].dtype, torch.int8 if bits == 8 else torch.int16)
            self.assertEqual(entry["values"][0].count_nonzero(), 0)
            self.assertTrue(torch.isfinite(entry["scale"]).all())
            self.assertTrue(((original["0.weight"] - model[0].weight).abs()
                             <= entry["scale"] / 2 + 1e-7).all())
            for name in ("0.bias", "1.weight", "1.bias"):
                torch.testing.assert_close(model.state_dict()[name], original[name], atol=0, rtol=0)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "weights.pt"
                torch.save(bundle, path)
                restored = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
                self.assertEqual(load_weights(restored, path), bits)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


class ArithmeticTests(unittest.TestCase):
    def test_zero_sign_underflow_saturation_and_times(self):
        x = torch.tensor([0., .25, .36, .7, 1.4, 2.9, -1.4])
        code = encode_log(x, bits=3, e_max=1, signed=True)
        torch.testing.assert_close(code.decode(), torch.tensor([0., 0., .5, .5, 1., 2., -1.]))
        self.assertEqual(code.times().tolist(), [3, 3, 2, 2, 1, 0, 1])
        self.assertEqual(levels(4, True), 7)  # total bits INCLUDE sign
        self.assertEqual(levels(4, False), 15)
        unsigned = quantize(torch.tensor([-1., 0., 1.]), "log", 3, 0, signed=False)
        torch.testing.assert_close(unsigned, torch.tensor([0., 0., 1.]))

    def test_rounding_is_in_log_domain(self):
        x = torch.tensor([1.41, 1.42, 1.99, 2.0])
        torch.testing.assert_close(quantize(x, "log", 4, 2), torch.tensor([1., 2., 2., 2.]))
        torch.testing.assert_close(quantize(x, "log", 4, 2, rounding="floor"), torch.tensor([1., 1., 1., 2.]))

    def test_sixteen_bits_extend_log_range_but_refine_uniform_spacing(self):
        x = torch.tensor([0., 2.**-100, .17, .7, 1.4, 2.9])
        self.assertEqual(levels(16, False), 65535)
        self.assertEqual(levels(16, True), 32767)
        torch.testing.assert_close(quantize(x, "log", 16, 2, signed=False),
                                   quantize(x, "log", 8, 2, signed=False), atol=0, rtol=0)
        narrow = quantize(x, "uniform", 8, 2, signed=False)
        wide = quantize(x, "uniform", 16, 2, signed=False)
        self.assertLess(float((wide - x).abs().sum()), float((narrow - x).abs().sum()))
        self.assertTrue(torch.isfinite(wide).all())

    def test_uniform_equal_code_budget(self):
        x = torch.linspace(-3, 3, 4096)
        for bits in (2, 3, 4, 5, 6, 8, 16):
            for signed in (False, True):
                for mode in ("uniform", "log"):
                    q = quantize(x, mode, bits, 1, signed)
                    self.assertLessEqual(q.unique().numel(), 2 ** bits)
                    self.assertTrue(torch.isfinite(q).all())

    def test_float_shift_matches_dense_batched_and_linear(self):
        torch.manual_seed(7)
        left = quantize(torch.randn(2, 3, 5, 7), "log", 4, 2)
        right = torch.randn(2, 3, 7, 4)
        torch.testing.assert_close(pow2_matmul(left, right), left @ right, atol=1e-6, rtol=1e-6)
        weight = torch.randn(7, 4)
        torch.testing.assert_close(pow2_matmul(left, weight), left @ weight, atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            pow2_matmul(torch.full((2, 7), 1.5), weight)

    def test_integer_shift_exact_and_overflow_guard(self):
        x = torch.tensor([[0., -0.25, 1., 4.], [2., 0.5, -1., 0.]])
        code = encode_log(x, 4, 2)
        weight = torch.tensor([[2, -3, 4, 5], [-1, 7, 0, -2]], dtype=torch.int64)
        acc, exponent = integer_shift_linear(code, weight, weight_fraction_bits=3)
        actual = torch.ldexp(acc.double(), torch.tensor(exponent))
        expected = code.decode().double() @ (weight.double() / 8).T
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaises(OverflowError):
            integer_shift_linear(code, weight * 100000, accumulator_bits=16)
        zero_acc, _ = integer_shift_linear(encode_log(torch.zeros_like(x), 4, 2), weight)
        self.assertEqual(int(zero_acc.abs().sum()), 0)

    def test_calibration_is_frozen_during_evaluation(self):
        q = ActivationQuantizer("test")
        q.observing = True
        q(torch.linspace(-4, 4, 4096))
        q.observing = False
        fit = q.configure("log", 4, range(-3, 2))
        exponent = q.e_max
        q(torch.tensor([10000., 0., 1e-9]))
        self.assertEqual(q.e_max, exponent)
        self.assertEqual(q.stats["elements"], 3)
        self.assertEqual(q.stats["underflow"], 1)
        self.assertTrue(fit["signed"])

    def test_calibration_keeps_outliers_with_a_bounded_sample(self):
        q = ActivationQuantizer("outliers", sample_limit=16)
        value = torch.zeros(1, 1, 1024)
        value[..., 501] = -1000
        value[..., 503] = 2000
        q.observing = True
        q(value)
        q.observing = False
        sample = torch.cat(q.samples)
        self.assertEqual(sample.numel(), 16)
        self.assertEqual(float(sample.min()), -1000)
        self.assertEqual(float(sample.max()), 2000)
        q.configure("uniform", 16, range(-6, 3))
        self.assertGreaterEqual(q.e_max, 11)
        torch.testing.assert_close(q(value), value, atol=.04, rtol=0)


class BertTests(unittest.TestCase):
    @staticmethod
    def make_model():
        from transformers import BertConfig, BertForSequenceClassification
        torch.manual_seed(11)
        config = BertConfig(vocab_size=24, hidden_size=24, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=40,
                            max_position_embeddings=32, hidden_dropout_prob=0,
                            attention_probs_dropout_prob=0, num_labels=2)
        config._attn_implementation = "eager"
        return BertForSequenceClassification(config).eval()

    def test_passthrough_attention_mask_and_no_weight_changes(self):
        from .modeling import instrument
        batch = dict(input_ids=torch.tensor([[2, 4, 5, 3, 0], [2, 6, 3, 0, 0]]),
                     attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]))
        for scope in ("linear", "attention", "all"):
            model = self.make_model()
            before = [p.detach().clone() for p in model.parameters()]
            with torch.no_grad():
                reference = model(**batch, output_attentions=True)
                quantizers = instrument(model, scope)
                output = model(**batch, output_attentions=True)
            torch.testing.assert_close(reference.logits, output.logits, atol=1e-7, rtol=1e-6)
            for a, b in zip(reference.attentions, output.attentions):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
            self.assertTrue(quantizers)
            for a, b in zip(before, model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_whole_tiny_model_shift_parity(self):
        from .modeling import instrument, set_arithmetic
        model = self.make_model()
        batch = dict(input_ids=torch.tensor([[2, 4, 5, 3]]), attention_mask=torch.ones(1, 4))
        quantizers = instrument(model)
        for q in quantizers.values():
            q.observing = True
        with torch.no_grad():
            model(**batch)
            for q in quantizers.values():
                q.observing = False
                q.configure("log", 4, range(-3, 2))
            dense = model(**batch).logits
            set_arithmetic(model, "shift")
            shift = model(**batch).logits
        torch.testing.assert_close(dense, shift, atol=1e-6, rtol=1e-5)

    def test_offline_runner_reports_counts_and_separate_calibration(self):
        from transformers import BertTokenizer
        from .evaluate import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, data_dir = root / "model", root / "data"
            model_dir.mkdir()
            data_dir.mkdir()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "film", "great", "awful"]
            (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
            BertTokenizer(vocab_file=str(model_dir / "vocab.txt")).save_pretrained(model_dir)
            self.make_model().save_pretrained(model_dir)
            (data_dir / "train.tsv").write_text("sentence\tlabel\ngood film\t1\nbad film\t0\n")
            (data_dir / "dev.tsv").write_text("sentence\tlabel\ngreat film\t1\nawful film\t0\n")
            report = main(["--model", str(model_dir), "--data-dir", str(data_dir),
                           "--local-files-only", "--output-dir", str(root / "results"),
                           "--bits", "3", "--calibration-samples", "2", "--max-length", "8",
                           "--batch-size", "2", "--threads", "1", "--collect-stats"])
            self.assertEqual(len(report["results"]), 4)
            self.assertEqual(report["calibration_split"], "train")
            for result in report["results"]:
                self.assertEqual(result["examples"], 2)
            self.assertTrue((root / "results" / "summary.csv").is_file())
            weighted = main(["--model", str(model_dir), "--data-dir", str(data_dir),
                             "--local-files-only", "--output-dir", str(root / "weights_run"),
                             "--bits", "3", "--weight-bits", "16", "--export-weights",
                             "--calibration-samples", "2", "--max-length", "8",
                             "--batch-size", "2", "--threads", "1"])
            self.assertEqual(len(weighted["results"]), 5)
            self.assertTrue(weighted["weight_quantization"]["reload_logits_verified"])
            self.assertEqual(weighted["results"][0]["weight_bits"], 32)
            self.assertEqual(weighted["results"][1]["weight_bits"], 16)
            self.assertTrue((root / "weights_run" / "weights" / "weights.pt").is_file())
            with self.assertRaises(FileExistsError):
                main(["--output-dir", str(root / "weights_run")])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
