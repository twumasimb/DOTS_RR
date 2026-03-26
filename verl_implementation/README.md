# DOTS-RR Implementation with verl

This directory contains a reimplementation of the DOTS-RR (Difficulty-targeted Online Data Selection + Rollout Replay) algorithm using the [verl](https://github.com/verl-project/verl) framework.

## Overview

### What is verl?

**verl** (Volcano Engine Reinforcement Learning) is a flexible RL training framework for LLMs that provides:
- Distributed rollout generation (supports vLLM, SGLang backends)
- Modular trainer architecture (PPO, GRPO, etc.)
- `DataProto` - a unified data structure for passing batches between components
- Ray-based orchestration for multi-GPU/multi-node training

### Why verl instead of trl?

| Feature | trl (GRPOTrainer) | verl |
|---------|-------------------|------|
| Rollout control | Handled internally | Full control via `RolloutWorker` |
| Batch manipulation | Limited | `DataProto` with concat, slice, save/load |
| Replay buffer | Not supported | Easy to implement with `DataProto` |
| Distributed training | Basic | Ray-based, highly scalable |
| Off-policy correction | Not supported | Built-in `temp_log_prob` mechanism |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DOTS-RR Training Loop                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Dataset   │───▶│    DOTS     │───▶│  Selected   │                  │
│  │  (N items)  │    │  Sampling   │    │  Questions  │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                         │
│                           ┌───────────────────┴───────────────────┐     │
│                           ▼                                       ▼     │
│                    ┌─────────────┐                         ┌───────────┐│
│                    │   Fresh     │                         │  Replay   ││
│                    │  Rollouts   │                         │  Buffer   ││
│                    │  (δ·B)      │                         │ ((1-δ)·B) ││
│                    └──────┬──────┘                         └─────┬─────┘│
│                           │                                      │      │
│                           └──────────────┬───────────────────────┘      │
│                                          ▼                              │
│                                   ┌─────────────┐                       │
│                                   │  Combined   │                       │
│                                   │   Batch     │                       │
│                                   └──────┬──────┘                       │
│                                          │                              │
│                                          ▼                              │
│                                   ┌─────────────┐                       │
│                                   │    GRPO    │                       │
│                                   │   Update    │                       │
│                                   └──────┬──────┘                       │
│                                          │                              │
│                                          ▼                              │
│                                   ┌─────────────┐                       │
│                                   │   Store     │                       │
│                                   │ Informative │──────▶ Replay Buffer  │
│                                   │  Rollouts   │                       │
│                                   └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
verl_implementation/
├── README.md                 # This file
├── config.py                 # Hyperparameters and configuration
├── data_proto.py             # Simplified DataProto implementation
├── replay_buffer.py          # FIFO replay buffer with deduplication
├── difficulty_predictor.py   # DOTS adaptive difficulty prediction
├── reward_functions.py       # Reward computation (boxed answer matching)
├── rollout_generator.py      # Rollout generation utilities
├── grpo_loss.py              # GRPO loss computation
├── train.py                  # Main training script
└── utils.py                  # Shared utilities
```

## Key Concepts

### 1. DataProto

`DataProto` is verl's core data structure for passing batches between components:

```python
@dataclass
class DataProto:
    batch: TensorDict          # PyTorch tensors (input_ids, log_probs, etc.)
    non_tensor_batch: dict     # NumPy arrays (question text, solution, etc.)
    meta_info: dict            # Metadata
```

**Key Operations:**
```python
# Concatenate batches
combined = DataProto.concat([batch1, batch2])

# Index/slice
subset = data[indices]

# Save/load for checkpointing
data.save_to_disk("checkpoint.pkl")
data = DataProto.load_from_disk("checkpoint.pkl")
```

### 2. Rollout Generation

In verl, rollouts are generated explicitly and stored with:
- `input_ids`: Full sequence (prompt + response)
- `response_ids`: Generated tokens only
- `old_log_probs`: Log probabilities at generation time (needed for PPO/GRPO ratio)
- `rewards`: Computed reward values
- `advantages`: Normalized advantages for policy gradient

### 3. Replay Buffer

The replay buffer stores **complete rollout data** (not just questions):

```python
# What's stored per rollout:
{
    "input_ids": tensor,           # Full sequence
    "attention_mask": tensor,      # Attention mask
    "old_log_probs": tensor,       # π_old(a|s) at generation time
    "advantages": tensor,          # Computed advantages
    "rewards": tensor,             # Binary reward (0 or 1)
    "index": int,                  # Question ID (for deduplication)
    "question": str,               # Original question text
    "solution": str,               # Ground truth solution
}
```

**Buffer Rules:**
1. Only store "informative" rollouts: avg_reward ∈ (0, 1)
2. FIFO eviction when capacity exceeded
3. Deduplicate: keep only most recent occurrence per question

### 4. GRPO Loss

GRPO (Group Relative Policy Optimization) loss:

```python
# Advantage computation (no std normalization per Dr. GRPO)
advantages = rewards - rewards.mean()

# Policy ratio
ratio = exp(log_prob - old_log_prob)

# Clipped surrogate loss
loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

For **replayed samples**, the `old_log_prob` comes from the buffer (behavior policy), making the update slightly off-policy. PPO/GRPO clipping handles this naturally.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets accelerate

# For full verl features (optional):
pip install verl

# Run training
python train.py
```

## Hyperparameters

See `config.py` for all hyperparameters. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G` | 8 | Rollouts per question |
| `K` | 256 | Reference set size for DOTS |
| `ALPHA` | 0.5 | Target difficulty |
| `TAU` | 1e-3 | Sampling temperature |
| `DELTA` | 0.5 | Fresh rollout fraction |
| `BUFFER_CAPACITY` | 512 | Max rollouts in replay buffer |
| `B` | 8 | Training batch size (questions) |
| `T` | 60 | Total training steps |

## References

- [DOTS-RR Paper](https://arxiv.org/abs/2506.05316)
- [verl GitHub](https://github.com/verl-project/verl)
- [ASTRAL Implementation](https://github.com/ASTRAL-Group/data-efficient-llm-rl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
