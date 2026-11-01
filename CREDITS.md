# Credits and third-party material

## Models and data

| Material | Used for | Source | Terms |
|---|---|---|---|
| `cifar100_vgg16_bn` pretrained weights and VGG16-BN definition (`models/vgg16/source/logQuant/image_models/model.py`, adapted from upstream commit 786c162) | FP32 teacher and starting point of the VGG-16 W4A4 network | chenyaofo, [pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) | BSD-3-Clause, `models/vgg16/source/logQuant/image_models/UPSTREAM_LICENSE` |
| Otters code (`models/otters/code/`) and the W1A4 RTE checkpoint (`models/otters/checkpoint/`) | BERT workload, activation profile, polarity conversion | Z. Yan, J. Mao et al., "Otters: An Energy-Efficient Spiking Transformer via Optical Time-to-First-Spike Encoding", ICLR 2026, [arXiv:2509.18968](https://arxiv.org/abs/2509.18968) | No licence file in the code; check with the authors before redistributing |
| BERT-base-uncased; fine-tuned RTE teacher `textattack/bert-base-uncased-RTE` | Backbone of the Otters model | J. Devlin et al., NAACL 2019; [TextAttack](https://huggingface.co/textattack/bert-base-uncased-RTE) | As published on the Hugging Face Hub |
| CIFAR-10, CIFAR-100 | CNN training and test images | A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009 | Dataset terms |
| GLUE RTE (`models/otters/glue_data/RTE/`) | BERT dev examples | A. Wang et al., "GLUE", ICLR 2019 | Dataset terms |

## Methods and baselines

| Material | Used for | Reference |
|---|---|---|
| LSQ quantiser | 4-bit QAT (VGG-16-half), 4-bit BERT weights | S. K. Esser et al., "Learned Step Size Quantization", ICLR 2020 |
| SparTen | Sparse QNN baseline (`rtl/sparten.sv`, `rtl/spf_periphery.sv`) | A. Gondimalla et al., "SparTen: A Sparse Tensor Accelerator for Convolutional Neural Networks", MICRO 2019 |
| LoAS | TPPE array of the APEX baseline | R. Yin, Y. Kim, D. Wu, P. Panda, "LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks", MICRO 2024 |
| APEX | Rate-coded SNN baseline (`baselines/apex/`) | D. Bawa Venkatesh, S. Radhakrishnan, R. Rakshit, G. Srinivasan, "APEX: A Dual-Sparsity Accelerator for Precise and Efficient SNN Inference", arXiv:2608.19046, 2026 |
| Eyeriss v1 | Row-stationary baseline (`baselines/eyeriss/`) | Y.-H. Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks", IEEE JSSC 2017 |

The SparTen, APEX and Eyeriss RTL in this folder are re-implementations from the papers, not the authors' code.

## Tools and libraries

| Tool | Use |
|---|---|
| NanGate 45 nm Open Cell Library (typical corner, `data/lib/`) | Standard cells for synthesis and power |
| CACTI 7 (R. Balasubramonian et al., ACM TACO 2017) | SRAM energies and areas (`data/cacti/`) |
| Verilator | Cycle simulation and SAIF switching activity |
| Synopsys Design Compiler U-2022.12 | Synthesis and power analysis |
