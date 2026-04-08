# RNN vs LSTM vs GRU — Can they keep going without real data?

Train three models on a sine-like wave. Then ask them to keep going on their own — no real data, just their own predictions fed back in. That's where the difference shows up.

---

## The Core Idea

Short-term prediction is easy. All three models learn the signal and track it well.

The real test is **recursive extrapolation** — give the model a seed window, then cut off real data completely. Every prediction becomes the next input. After ~20–40 steps, the RNN loses the wave. The LSTM and GRU hold it.

---

## Signal

```
y = sin(t + cos(t)) + noise (σ = 0.03),  t ∈ [0, 30π],  N = 500
```

---

## Model Setup

All three models are **identical in every way except the recurrent layer**:

| Setting | Value |
|---|---|
| Hidden size | 12 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 50 |
| Input window | 20 steps |

No optimizer differences. No hidden size differences. No bias.

---

## Results

### Plot 1 — Short-term prediction
All three models overlap with the true signal. Nothing separates them here.

### Plot 2 — Recursive extrapolation (100 steps)
After the dashed line, the model receives no real data — only its own outputs.

- **RNN** → drifts and flattens
- **LSTM** → holds the wave
- **GRU** → holds the wave

---

## Why

A plain RNN rewrites its hidden state at every step with no control over what gets kept or discarded. Running on its own outputs, small errors build up and the memory collapses to a fixed point.

LSTM and GRU have **gates** — learned switches that decide what to remember and what to forget. That gives them a way to hold the wave's structure even without real data coming in.

The gate is the only difference. Everything else was identical.

---

## Project Structure

```
rnn-lstm-gru/
├── rnn_lstm_gru.ipynb
└── README.md
```

## Getting Started

```bash
git clone <your-repo-link>
cd rnn-lstm-gru

pip install torch numpy matplotlib

jupyter notebook rnn_lstm_gru.ipynb
```

## Stack

Python · PyTorch · NumPy · Matplotlib
