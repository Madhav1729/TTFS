Teacher (FP32): dev 99.96%, test 74.00%

Dev = 5,000 held-out training images, which the pretrained teacher saw in training.

| Config | PTQ test | QAT test (final epoch) | QAT test (best-dev epoch) | PTQ dev | best dev (epoch) |
|---|---:|---:|---:|---:|---:|
| hw_w4_uniform_a4 | 60.68% | **71.41%** | 71.47% | 83.80% | 99.38% (27) |
| hw_w4_log3 | 32.54% | **62.83%** | 62.83% | 40.64% | 87.34% (30) |
