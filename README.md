# Domain Shift Analysis in Bengali Handwritten Digit Recognition

## Overview

This project studies the impact of **domain shift** in Bengali handwritten digit recognition and investigates methods to improve robustness under unseen stylistic and structural distortions.

The work compares:

* **Feature-level domain generalization** using MixStyle
* **Input-level structural augmentation** using stroke-based corruption techniques

The key finding is that **style variation and structural variation behave differently**, and feature-level methods alone are insufficient for handling severe structural distortions.

---

# Problem Statement

Deep learning models trained on clean handwritten digit datasets often fail when exposed to unseen domains containing:

* style variations
* illumination changes
* noise
* broken or incomplete strokes

This project analyzes these failures and proposes a robust training strategy to improve domain generalization.

---

# Dataset

## Bengali Handwritten Digit Dataset

* Classes: 10 (0–9)
* Input size: `224 × 224`
* Grayscale images converted to 3-channel format

### Domains

| Domain   | Description                             |
| -------- | --------------------------------------- |
| Clean    | Standard clean handwritten digits       |
| Domain C | Style variations                        |
| Domain D | Illumination and noise distortions      |
| Domain E | Structural distortions / broken strokes |

### Folder Structure

```bash
datasets/
│
├── train_final/
├── test_clean/
├── val_train_c/
├── val_train_d/
└── val_train_e/
```

---

# Preprocessing

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
```

---

# Baseline Model

## Architecture

* ResNet-18 backbone
* Global Average Pooling
* Fully Connected layer (10 classes)

### Training Setup

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Training Domain: Clean only

---

# MixStyle

MixStyle is a **feature-level domain generalization technique** that mixes the feature statistics of different samples.

## Key Idea

It mixes:

* feature-map mean
* feature-map standard deviation

across shuffled samples to simulate new styles during training.

## MixStyle Placement

Applied after:

* Layer1
* Layer2

with:

```python
p = 0.2
```

## Observation

MixStyle improves robustness to:

* style changes
* illumination variations

However, it performs poorly under:

* structural distortions
* broken strokes
* incomplete digit structures

---

# Augmentation Experiments

## 1. Mild Stroke Augmentation

Includes:

* slight erosion
* dilation
* blur
* thickness variation

### Observation

Provided moderate improvement.

---

## 2. Strong Stroke Augmentation

Includes:

* aggressive erosion
* random pixel removal
* random line cuts
* broken strokes
* structural corruption

### Observation

Produced the best robustness under severe domain shift.

---

## 3. Hybrid Augmentation

Random combination of:

* mild augmentations
* strong augmentations

### Observation

Training became inconsistent and unstable.

---

# Final Model

## Final Configuration

```text
ResNet18 + MixStyle + Strong Stroke Augmentation
```

### Strategy

* MixStyle handles style variation
* Strong augmentation handles structural variation

This combination achieved the best balance between clean accuracy and robustness.

---

# Results

| Model                 | Clean | Domain C | Domain D | Domain E |
| --------------------- | ----- | -------- | -------- | -------- |
| MixStyle              | 99.29 | 10.58    | 47.80    | 23.27    |
| MixStyle + Strong Aug | 99.62 | 80.62    | 90.28    | 44.43    |
| Hybrid + MixStyle     | 99.49 | 31.65    | 71.04    | 25.77    |

---

# Key Insights

* Style variation and structural variation are fundamentally different.
* Feature-level regularization alone is insufficient.
* Structural augmentation is critical for robust domain generalization.
* Strong augmentation forces the network to learn more robust structural representations.

---

# Evaluation

The project was evaluated using:

* Accuracy
* Confusion matrices
* Domain-wise comparison
* Diagonal vs off-diagonal concentration analysis

---

# Visualizations

Generated visualizations include:

* Baseline confusion matrices
* Final model confusion matrices
* MixStyle feature maps
* Strong augmentation examples
* Domain comparison samples

---

# Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Baseline Training
   ↓
Domain Evaluation
   ↓
MixStyle
   ↓
Mild / Strong / Hybrid Augmentation
   ↓
Final Model
   ↓
Evaluation & Analysis
```

---

# Libraries Used

* PyTorch
* torchvision
* OpenCV
* NumPy
* Matplotlib
* seaborn
* PIL

---

# Project Structure

```bash
project/
│
├── datasets/
│   ├── train_final/
│   ├── test_clean/
│   ├── val_train_c/
│   ├── val_train_d/
│   └── val_train_e/
│
├── outputs/
│   ├── confusion_matrices/
│   ├── feature_maps/
│   └── augmentation_samples/
│
├── train_baseline.py
├── train_mixstyle.py
├── augmentations.py
├── evaluate.py
└── README.md
```

---

# Challenges

1. Baseline overfitted the clean domain.
2. MixStyle failed under structural distortion.
3. Domain E produced heavy misclassification.
4. Hybrid augmentation caused inconsistent learning.

---

# Solutions

* Introduced stroke-based structural augmentation
* Simulated realistic distortions during training
* Combined feature-level and input-level domain generalization

---

# Limitations

* Evaluated only on Bengali handwritten digits
* No transformer-based architectures explored
* No real-world deployment
* Structural augmentation handcrafted manually

---

# Future Work

* Vision Transformer (ViT)-based DG
* Learnable augmentation policies
* Real-world scanned handwritten datasets
* Advanced domain adaptation methods
* Multi-dataset evaluation

---

# Core Thesis Message

> MixStyle effectively handles style variation, but strong structural augmentation is necessary for robust domain generalization under structural distortions.
