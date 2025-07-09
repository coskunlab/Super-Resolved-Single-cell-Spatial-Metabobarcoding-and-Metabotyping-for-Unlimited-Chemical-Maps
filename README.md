# Guided Super-Resolution MSI Demo – Python Shiny App
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Self-supervised **Guided Super-Resolution (GSR)** merges low-resolution mass-spectrometry ion maps with a co-registered high-resolution structural image (e.g. fluorescence or immunohistochemistry) to recover single-cell molecular detail.  
This Shiny for Python app provides an interactive front-end to a dual-encoder U-Net implementation of GSR, allowing you to upload your own images, tune hyper-parameters, and visualize results in real time.

---

## ✨ Key Features
* **Drag-and-drop UI** – Upload a high-res guide image and a low-res MSI channel directly in the browser.  
* **Self-supervised training** – Combines MSE and SSIM losses; no paired ground truth is required.  
* **Deterministic results** – Uses the project-wide seed **6740** for reproducibility.  
* **GPU-optional** – Runs on CPU by default; automatically switches to CUDA if available.  
* **Progress bar & live previews** – Monitor training and instantly compare guide, input, and super-resolved output (with a perceptual *viridis* palette).

---

## 🔧 Installation
```bash
# 1. Clone the repo
git clone https://github.com/your-username/gsr-msi-shiny.git
cd gsr-msi-shiny

# 2. Create an environment (recommended)
conda create -n gsr_msi python=3.10
conda activate gsr_msi

# 3. Install dependencies
pip install -r requirements.txt
# or, interactively:
pip install shiny>=0.9 torch torchvision torchaudio \
            super-image pytorch-msssim opencv-python \
            matplotlib pillow scikit-image numpy
