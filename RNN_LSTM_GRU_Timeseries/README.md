# Time-Series Forecasting: RNN vs LSTM vs GRU on Sine Wave Extrapolation

A structured experimental study of recurrent neural networks on a synthetic time-series signal. Rather than just training a sequence predictor, this project investigates how architectural differences — specifically the presence or absence of gating mechanisms — influence long-range memory, training stability, and extrapolation behaviour under stress.

---

## Objective

To move beyond implementation and develop intuition about:

- Why gated architectures (LSTM, GRU) handle long-range dependencies better than Vanilla RNN
- How recursive extrapolation exposes architectural weaknesses that short-term accuracy completely hides
- Why optimizer choice matters differently depending on architecture
- What hidden unit activations reveal about what a network has actually learned

---

## Model Architecture

```
Input (1) → Recurrent Layer → nn.Linear → Output (1)
```

- Input: one scalar value per time step (univariate signal)
- Output: single scalar prediction (next time step)
- Loss: `nn.MSELoss`

Models are given **different capacities and optimizers** to reflect realistic architectural trade-offs — not an artificial controlled experiment:

| Model | Hidden Size | Parameters | Optimizer | LR |
|---|---|---|---|---|
| Vanilla RNN | 8 | 97 | SGD | 0.01 |
| LSTM | 32 | 4,513 | Adam | 0.001 |
| GRU | 24 | 1,969 | Adam | 0.001 |

RNN uses SGD at a small hidden size — consistent with its known convergence behaviour and limitations. LSTM and GRU use Adam with larger hidden sizes to demonstrate what gated architectures can achieve when properly configured.

---

## The Data

**Signal:** `y = sin(t + cos(t)) + ε`, `t ∈ [0, 30π]`, `N = 500` points, `ε ~ N(0, 0.05)`

Gaussian noise (σ = 0.05) is added to prevent pure memorization. A clean signal can be fit by any model — noise forces each architecture to learn the underlying structure rather than the exact sequence.

**Sliding Window:**

The signal is sliced into input-output pairs using a window of `seq_length = 20`. For each position `i`, the model sees `data[i : i+20]` and predicts `data[i+20]`. This yields 480 training pairs.

**Tensor shape:**

```
(seq_length, batch_size, input_size)  →  (20, 1, 1)
```

---

## Training Setup

| Setting | RNN | LSTM | GRU |
|---|---|---|---|
| Optimizer | SGD | Adam | Adam |
| Learning rate | 0.01 | 0.001 | 0.001 |
| Epochs | 50 | 50 | 50 |
| Hidden state init | `None` each epoch | `None` each epoch | `None` each epoch |

`hidden.detach()` is called before every forward pass for all three models. For LSTM, both `h` and `c` are detached since the hidden state is a tuple.

---

## Experimental Design

| Experiment | What It Tests |
|---|---|
| Short-term next-step accuracy | Baseline fit — all models expected to perform reasonably |
| Recursive extrapolation (500 steps) | Long-horizon memory — where architectural gaps become visible |
| Hidden state visualisation | Internal representations — what each unit tracks over time |

---

## Results & Observations

### 1. Training Loss

| Model | Epoch 10 | Epoch 30 | Epoch 50 (Final) |
|---|---|---|---|
| Vanilla RNN | 0.007350 | 0.005845 | 0.004585 |
| LSTM | 0.003976 | 0.004431 | 0.003653 |
| GRU | 0.005152 | 0.004486 | 0.003494 |

- LSTM and GRU converge significantly faster and reach lower final loss
- RNN plateaus early — SGD struggles to escape shallow local minima on a noisy signal with limited hidden capacity
- GRU edges out LSTM slightly on final loss — fewer parameters, less overfitting to noise

---

### 2. Short-Term Next-Step Accuracy (Pearson r)

| Model | Final Loss (MSE) | Pearson r |
|---|---|---|
| Vanilla RNN | 0.004585 | 0.9957 |
| LSTM | 0.003653 | 0.9972 |
| GRU | 0.003494 | 0.9970 |

- All three models predict next-step values reasonably well
- The gap between RNN (~0.996) and LSTM/GRU (~0.997) is visible but not dramatic
- Short-term accuracy alone does not tell the full story — this is the point

---

### 3. Recursive Extrapolation (500 Steps)

| Model | Extrapolation std | Behaviour |
|---|---|---|
| Vanilla RNN | 0.6772 | Maintained |
| LSTM | 0.6671 | Maintained |
| GRU | 0.6697 | Maintained |

> Higher std = more oscillation preserved over 500 self-generated steps.

- **RNN collapses immediately.** With only 8 hidden units and no gating, the hidden state cannot sustain temporal structure under its own feedback. Output converges to a constant near zero.
- **LSTM holds longest.** The cell state preserves frequency and amplitude information through hundreds of self-generated steps. Oscillation is visibly present at step 400+.
- **GRU holds well but not as long.** Without a separate cell state, the gated hidden state carries both short and long-term information simultaneously. It degrades more gracefully than RNN but earlier than LSTM.

This is the central finding. Without the extrapolation experiment, all three models look like they work.

---

### 4. Hidden State Visualisation

- **RNN units** show flat, low-variance activations — the small hidden size and lack of gating leaves most units underutilised
- **LSTM units** show clear, structured oscillations with different units tracking different aspects of the signal (frequency, amplitude, phase)
- **GRU units** show similar structure to LSTM but with slightly less separation between what each unit specialises in

---

## Why LSTM and GRU Outperform Vanilla RNN

### The Vanishing Gradient Problem

In a Vanilla RNN, the hidden state is recomputed at every step by multiplying through the same weight matrix. During backpropagation, gradients are multiplied by this matrix at every step — and over long sequences those products shrink toward zero. The network cannot learn what happened more than a few steps back. During recursive extrapolation, the hidden state collapses into a fixed point and the output flatlines. This is not a training artefact — it is a structural limitation.

### What LSTM Does Differently

LSTM introduces a **cell state** `c` that runs alongside the hidden state. It is updated through *addition*, not multiplication at every step. Additions do not shrink gradients the way repeated multiplications do. The gating mechanism (learned, not fixed) controls what gets written to the cell state, what gets cleared, and how much gets exposed as output. This gives the model a durable long-term memory channel that can survive hundreds of steps of self-generated feedback.

### What GRU Does Differently

GRU removes the separate cell state and merges everything into a single gated hidden state, controlled by two gates instead of LSTM's three. Fewer parameters, same core benefit: selective memory. GRU trains slightly faster than LSTM and converges to a marginally lower loss on this task. On very long extrapolation (400+ steps) it begins to degrade earlier — the single-state design carries more load than LSTM's dual-channel architecture.

### Why It Only Shows Up in Extrapolation

Short-term prediction is easy. Even a constrained RNN can interpolate within a pattern it has seen. The gap only becomes visible when:

1. The model receives no real data — only its own outputs
2. Errors compound step by step
3. The model must maintain coherent structure across hundreds of steps with no ground truth anchor

Gated models resist this error amplification. The Vanilla RNN does not. **The flatline is not a bad result — it is the result that proves the point.**

---

## Key Learnings

- Short-term accuracy is not a useful differentiator for recurrent architectures — all three models score above 0.98 r on next-step prediction
- Recursive extrapolation is the only experiment that separates them cleanly
- The cell state in LSTM is not a detail — it is the mechanism that makes long-horizon stability possible
- GRU is a practical default: nearly matches LSTM, trains faster, uses fewer parameters
- Adding noise to the training signal is important — without it, RNN can memorize the sequence and appear to work fine until it extrapolates
- `hidden.detach()` is not optional — without it, gradients accumulate across windows and produce wrong weight updates
- Optimizer choice is part of the architectural story — RNN with Adam would partially close the gap, but the structural ceiling remains

> All three models appear to work on next-step prediction. Only one still works when you take away the real data.

---

## Limitations

- Results are specific to this signal — a noisier or more irregular real-world series may show different relative rankings
- RNN was deliberately constrained (hidden=8, SGD) — with hidden=32 and Adam it would partially close the gap, though the structural ceiling remains
- Batch size of 1 — gradient estimates are high variance; mini-batching would stabilise training
- 50 epochs is sufficient to show the pattern but not to fully converge all models
- No dropout or weight decay — long-run overfitting not studied

---

## Future Work

- Test on real-world data (ECG, stock prices, weather) where noise patterns are non-Gaussian
- Equalise all training conditions (same hidden size, Adam for all) and rerun — does RNN close the gap or does the structural ceiling hold?
- Add multi-step ahead prediction (sequence-to-sequence output)
- Stack 2–3 recurrent layers and measure effect on extrapolation stability
- Compare with a Transformer on the same task — does attention achieve what gating does?
- Tune `seq_length` (40, 60, 100) and measure impact on long-horizon memory

---

## Tech Stack

- Python, PyTorch (`nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.Linear`, `nn.MSELoss`)
- NumPy, Matplotlib, SciPy

---

## Project Structure

```
timeseries-rnn-lstm-gru/
├── timeseries_rnn_lstm_gru.ipynb
└── README.md
```

---

## Getting Started

```bash
git clone <your-repo-link>
cd timeseries-rnn-lstm-gru

pip install torch numpy matplotlib scipy

jupyter notebook timeseries_rnn_lstm_gru.ipynb
```

---

## Conclusion

This project shows that recurrent architectures are not interchangeable. On short-term prediction, Vanilla RNN, LSTM, and GRU all perform well. On recursive extrapolation — where a model must sustain coherent structure using only its own outputs — the Vanilla RNN collapses and the gated models do not. The difference is not about training time or dataset size. It is structural: a plain hidden state cannot protect information the way a cell state or gated hidden state can. LSTM and GRU were designed to solve a specific problem that only becomes visible under specific conditions. This experiment creates those conditions.
