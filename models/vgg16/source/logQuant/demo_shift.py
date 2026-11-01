"""Small, actual integer shift-and-add example; no model download required."""

import torch

from .quantization import encode_log, integer_shift_linear


def main():
    activation = torch.tensor([[0., -0.3, 1.1, 3.8]])
    weights = torch.tensor([[2, -3, 4, 5], [-1, 7, 0, -2]], dtype=torch.int64)
    fractional_bits = 3  # Real weights are the stored integers divided by 8.
    code = encode_log(activation, bits=4, e_max=2, signed=True)
    accumulator, output_exponent = integer_shift_linear(code, weights, fractional_bits)
    output = torch.ldexp(accumulator.double(), torch.tensor(output_exponent))
    reference = code.decode().double() @ (weights.double() / (1 << fractional_bits)).T
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    print("Input:", activation.tolist())
    print("Log-quantized input:", code.decode().tolist())
    print("Logical spike slots:", code.times().tolist(), "(7 means no spike)")
    print("Integer accumulator:", accumulator.tolist())
    print("Output exponent:", output_exponent)
    print("Decoded shift/add output:", output.tolist())
    print("Exactly matches dense arithmetic on the same quantized inputs and fixed-point weights.")


if __name__ == "__main__":
    main()
