# Feedforward Neural Networks: Optimization & Generalization Analysis

A structured experimental study of Feedforward Neural Networks (FNNs) on the MNIST dataset.
Rather than just training a classifier, this project investigates **how optimization strategies,
model capacity, and training configurations influence convergence behavior and generalization performance**.

---

## Objective

To move beyond implementation and develop intuition about:
- How optimizers affect convergence speed and stability
- How model capacity impacts learning and overfitting
- How learning rate controls training dynamics
- How normalization and regularization influence performance

---

## Model Architecture
```
Input (784) → Hidden Layers (variable) → ReLU → Output (10 classes)
```

- **Input:** Flattened 28×28 MNIST images
- **Hidden layers:** `[64]`, `[128]`, `[256, 128]`
- **Loss Function:** CrossEntropyLoss
- **Training Setup:** 10,000 train / 5,000 test samples — 5 epochs for fast experimentation

---

## Experimental Design

All experiments follow a **controlled setup** — only one variable is changed at a time.

| Experiment | Variants Tested |
|---|---|
| Optimizer | SGD, SGD + Momentum, Adam |
| Model Capacity | Small `[64]`, Medium `[128]`, Large `[256, 128]` |
| Learning Rate | 0.1, 0.01, 0.001 |
| Normalization | With / Without |
| Regularization | None / L2 |

---

## Results & Observations

### 1. Optimizer Comparison

| Optimizer | Final Accuracy |
|---|---|
| SGD | ~74.5% |
| SGD + Momentum | ~87.2% |
| Adam | **~89.9%** |

- Adam converges fastest and achieves the highest accuracy
- Momentum dramatically improves vanilla SGD, reducing slow convergence
- Vanilla SGD struggles with both speed and stability

---

### 2. Model Capacity

| Model | Final Accuracy |
|---|---|
| Small `[64]` | ~87.3% |
| Medium `[128]` | ~87.6% |
| Large `[256, 128]` | **~88.8%** |

- Increasing capacity improves performance, but with diminishing returns
- Larger models learn faster but risk overfitting over longer training runs

---

### 3. Learning Rate Sensitivity

| Learning Rate | Behavior |
|---|---|
| 0.1 | Unstable, poor accuracy (~33%) |
| 0.01 | Best performance (~86–89%) |
| 0.001 | Stable but slower convergence (~88%) |

- High learning rate causes instability and divergence
- `0.01` is near-optimal for this setup
- Lower LR converges reliably but slowly

---

### 4. Normalization

| Setting | Final Accuracy |
|---|---|
| With normalization | ~87.5% |
| Without normalization | **~89.3%** |

- Contrary to expectation, normalization did not improve performance here
- Likely due to MNIST's simplicity and small model size
- Highlights that normalization is **context-dependent**, not universally beneficial

---

### 5. Regularization (L2)

| Weight Decay | Final Accuracy |
|---|---|
| 0 | **~89.6%** |
| 1e-4 | ~89.0% |

- L2 regularization slightly reduces performance in this setup
- The model is not heavily overfitting at only 5 epochs
- Regularization becomes more impactful with larger models and longer training

---

## Key Learnings

- **Optimizer choice** has the largest impact on convergence speed
- **Model capacity** provides incremental gains, not guaranteed improvements
- **Learning rate** is critical for training stability
- Not all best practices (like normalization) always help — context matters
- **Regularization is only useful when overfitting is present**

> Deep learning performance is governed more by training dynamics and hyperparameters than by architecture alone.

---

## Tech Stack

- Python, PyTorch, Torchvision, Matplotlib

---

## Project Structure
```
FNN/
├── fnn_optimization_generalization.ipynb
└── README.md
```

---

## Getting Started
```bash
git clone <your-repo-link>
cd deep-learning-foundations/FNN

pip install -r requirements.txt

jupyter notebook
```

> For final evaluation, use the full dataset and increase epochs beyond 5.

---

## Limitations

- FNN only — no convolutional layers
- Limited to 5 epochs; no long-term convergence study
- Results are MNIST-specific

---

## Future Work

- Extend experiments to CNNs on CIFAR-10
- Add Dropout regularization
- Compare Batch Normalization
- Explore deeper and wider architectures

---

## Conclusion

This study confirms that understanding optimization dynamics and hyperparameter sensitivity
is just as important as model architecture when building reliable deep learning systems.