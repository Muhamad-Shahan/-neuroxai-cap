# -neuroxai-cap

# NeuroXAI-Caps

An explainable CNN–Capsule Network for early Alzheimer's diagnosis from T1-weighted brain MRI scans.

## What it does
Classifies an uploaded MRI image into one of four cognitive impairment stages:
- No Impairment
- Very Mild Impairment
- Mild Impairment
- Moderate Impairment

Alongside the prediction, it shows:
- **Grad-CAM++** heatmap — highlights the brain regions the CNN backbone focused on.
- **LIME** explanation — outlines the superpixel regions most influential to the prediction.
- Inference latency and a Latency-Aware Accuracy Index (LAAI) score.

## Model
A hybrid CNN + Capsule Network (dynamic routing, 3 iterations), ~4.08M trainable parameters, trained on the "Best Alzheimer MRI Dataset" (Kaggle).

## Disclaimer
This is a research/academic prototype (Final Year Project). It is **not a medical device** and must not be used for real clinical diagnosis or treatment decisions.
