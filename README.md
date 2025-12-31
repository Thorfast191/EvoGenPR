# EvoGenPR: Fusion-Evolving Generative Pattern Recognition for Chest X-ray Analysis

## Overview

EvoGenPR is a closed-loop generative–discriminative learning framework designed for robust and adaptive medical image classification.  
The system integrates a **Generative Pattern Recognizer (GPR)** with a **Self-Evolving Neural Pattern Recognizer (SENPR)** through a novel **Fusion-Evolving Generative Loop (FEGL)**.

The framework is evaluated on the NIH ChestX-ray14 dataset and addresses key challenges in medical imaging:

- Severe class imbalance
- Distribution drift
- Limited abnormal samples
- Continual adaptation without catastrophic forgetting

---

## Core Contributions

- Hybrid **Diffusion–GAN** generator for targeted medical image synthesis
- Dual-encoder SENPR (ResNet + Swin Transformer)
- Feature Grafting and Scale-Aware Pyramid Fusion
- Continual learning using Replay Buffer + Elastic Weight Consolidation (EWC)
- Closed-loop feedback between generator and classifier (FEGL)
- 5-Fold **multilabel stratified cross-validation**

---

## Architecture

Input X-ray
│
▼
┌──────────────┐
│ GPR │ ← Diffusion + GAN
│ (Generator) │
└──────┬───────┘
│ Synthetic images + latent maps
▼
┌────────────────────────────┐
│ SENPR │
│ ResNet + Swin Transformer │
│ Feature Grafting + SPFM │
└───────────┬────────────────┘
│ Performance feedback
▼
FEGL Closed Loop

---

## Dataset

- **NIH ChestX-ray14**
- 112,120 frontal chest X-ray images
- 14 disease labels (multilabel)
- Hosted and executed on **Kaggle** to avoid storage constraints

---

## Evaluation Metrics

### Classification

- Accuracy
- Precision (macro)
- AUC-ROC (macro, multilabel)
- Focal Loss

### Generative

- Fréchet Inception Distance (FID)
- Inception Score (IS)

---

## Cross-Validation Strategy

- **5-fold multilabel stratified cross-validation**
- Preserves class distribution across folds
- Each fold ≈ 80% training / 20% validation

---

## Project Structure

evogenpr/
├── data/
├── gpr/
├── senpr/
├── fegl/
├── metrics/
├── configs/
├── train_colab.ipynb
└── README.md

---

## Running the Project (Kaggle / Colab)

1. Open `train_colab.ipynb`
2. Ensure GPU is enabled
3. Dataset path: `/kaggle/input/nih-chest-xray-dataset`
4. Run all cells sequentially

---

## Reproducibility

All experiments are controlled via `configs/config.yaml`.  
Random seeds and hyperparameters are explicitly defined.

---

## License

Academic and research use only.
