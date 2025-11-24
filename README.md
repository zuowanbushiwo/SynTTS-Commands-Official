# Syntts-Commands-Official：On-Device KWS via Synthetic Speech

<!-- Badges -->
<div align="center">
  
  [![arXiv](https://img.shields.io/badge/arXiv-2511.07821-b31b1b.svg)](https://arxiv.org/abs/2511.07821)
  [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-ffd21e)](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Dataset)
  [![Benchmarks](https://img.shields.io/badge/🤗%20Hugging%20Face-Benchmarks-ffd21e)](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Benchmarks)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br>

<p align="center">
  <strong>Official Implementation of "SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech"</strong>
</p>

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-resources">Resources</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-benchmark-results">Benchmarks</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📖 Introduction

**SynTTS-Commands** is a large-scale, multilingual (English & Chinese) synthetic speech command dataset designed for **low-power Keyword Spotting (KWS)** tasks. Generated using state-of-the-art TTS technology (CosyVoice 2), it addresses the data scarcity bottleneck in TinyML and Edge AI.

This repository contains:
1.  **Data Generation Scripts**: Code used to generate high-quality synthetic speech.
2.  **Training Code**: Implementation of KWS models (MicroCNN, DS-CNN, MobileNet-V1, etc.).
3.  **Evaluation Scripts**: Tools to reproduce the benchmark results presented in the paper.

## 🔗 Resources

| Resource | Description | Link |
| :--- | :--- | :--- |
| **📄 Paper** | Full technical report and analysis | [arXiv:2511.07821](https://arxiv.org/abs/2511.07821) |
| **💾 Dataset** | **384k+** Audio samples (Wave files) | [🤗 HF Dataset](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Dataset) |
| **🧠 Models** | Pre-trained checkpoints for benchmarks | [🤗 HF Models](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Benchmarks) |




