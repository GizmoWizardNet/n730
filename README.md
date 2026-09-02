<p align="center">
  <img src="git-assets/logo.png" width="220">
</p>

<h1 align="center">N730</h1>

<p align="center">
  Experimental streamed transformer inference for legacy GPUs.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GPU-GT730-green">
  <img src="https://img.shields.io/badge/CUDA-11.4-blue">
  <img src="https://img.shields.io/badge/Quantization-INT2%20%7C%20INT4%20%7C%20INT8-orange">
  <img src="https://img.shields.io/badge/License-MIT-red">
</p>

---

# What is N730?

N730 is an experimental AI inference runtime built to run modern
Large Language Models on extremely low-end hardware such as the
NVIDIA GT 730.

Instead of loading an entire model into VRAM at once, N730 streams
quantized transformer layers dynamically during inference, allowing
models far larger than GPU memory capacity to run on legacy hardware.

---
---

## Features

- Layer streaming runtime
- Dynamic mixed precision quantization
- Native AVX2/C++ backend
- CUDA acceleration for Kepler GPUs
- Runtime dequantization
- KV cache autoregressive inference
- HuggingFace conversion pipeline

---
---

## Status
<h2>WIP</h2>
 <p align="left">
  <img src="git-assets/screen0.png">
</p>

 <p align="right">
  <img src="git-assets/screen1.png">
</p>

---

## Capabilities

- Converting HuggingFace transformer models into the `.n730` format
- Streaming 198+ transformer layers from disk in real time
- Running autoregressive transformer inference
- Dynamically dequantizing INT2 / INT4 / INT8 / FP16 weights
- Executing inference on hardware as old as the GT 730

## Concept

Instead of loading an entire model into VRAM at once, N730 streams quantized transformer layers dynamically during inference, allowing models far larger than the GPU's memory capacity to run on legacy hardware.

---

## Supported Quantization

| Format | Status |
|---|---|
| FP16 | Working |
| INT8 | Working |
| INT4 | Experimental |
| INT2 | Experimental / cursed |

## Hardware Tested

| GPU | Status |
|---|---|
| GT 730 2GB DDR3 | Working |
| GTX 1650 | Untested |
| RTX Series | Needs re-done compiler with different instruction set |

---

## License

MIT 

## Disclaimer

N730 is an experimental research project haha.

Performance, correctness, and stability are not working send help pls

This project is not affiliated with NVIDIA, DeepSeek, HuggingFace, or any model provider. F*ck big model providers.

**EXTRA NOTES**
- You are absolutely not allowed to sell or make money from this software commercially, whether it be through access limits, usage or donations.
- If any modifications that you think would help this project further, please do a PR.
- Grammatical or style issues do not require a PR, do an issue instead. Such PRs will be closed without notice.
