# Mistletoe

Official implementation of **Mistletoe: Pruning-Robust Watermarking for Visual Models with Symbiotic Gradient Alignment**. 

This repository contains the core implementation of the watermark construction and embedding pipeline proposed in the paper.

---

# Repository Structure

```text id="1i79l9"
watermark_data_generation.py
watermark_alignment_training.py
```

---

# 1. Watermark Data Generation

`watermark_data_generation.py`

This script generates the watermark dataset used for watermark embedding.

It includes:

* watermark sample generation
* watermark label selection

The current implementation is built for **ResNet-50**.
For other architectures, only the model-related components need to be replaced.

## Main functionality

* Generate watermark samples from seed images
* Construct watermark behaviors from in-distribution data
* Automatically select suitable watermark labels
* Export the final watermark dataset

---

# 2. Watermark Alignment Training

`watermark_alignment_training.py`

This script performs watermark embedding using the proposed Symbiotic Alignment Training strategy. 

## Main functionality

* Jointly optimize task and watermark objectives
* Align watermark optimization with task optimization
* Improve watermark robustness under fine-pruning
* Support mixed-precision training (AMP)

---

# Disclaimer

This repository is provided for research purposes only.
