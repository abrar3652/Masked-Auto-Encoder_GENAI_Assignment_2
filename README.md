# Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)

This repository contains a PyTorch implementation of a Masked Autoencoder (MAE) based on an asymmetric Vision Transformer (ViT) architecture. The model is trained to reconstruct images with 75% of their patches masked.

## Architecture
* **Encoder:** ViT-Base (12 layers, 768 dim, 12 heads) - Processes only the 25% visible patches.
* **Decoder:** ViT-Small (12 layers, 384 dim, 6 heads) - Reconstructs the full image from visible + mask tokens.

## Dataset & Training
* **Dataset:** TinyImageNet (Resized to 224x224)
* **Optimization:** AdamW optimizer, Cosine Annealing LR Scheduler, Mixed Precision (AMP).
* **Metrics Achieved (30 Epochs):** PSNR: 22.61 dB | SSIM: 0.7162

## Run the Streamlit App Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`