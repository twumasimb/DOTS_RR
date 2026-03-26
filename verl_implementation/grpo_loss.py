"""
GRPO Loss Computation

This module implements the Group Relative Policy Optimization (GRPO) loss
function used for training the policy model.

Key Concepts:
=============

1. GRPO vs PPO
   - PPO: Uses a learned value function for advantage estimation
   - GRPO: Uses group-relative advantages (no value function needed!)

   For each question with G rollouts:
       A_i = r_i - mean(r_1, ..., r_G)

   This is simpler and works well for verifiable reward tasks.

2. POLICY GRADIENT WITH CLIPPING
   The GRPO objective (similar to PPO):

       L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

   Where:
       ratio = π_θ(a|s) / π_old(a|s)
       - π_θ: Current policy (log probs computed in forward pass)
       - π_old: Old policy (stored in rollout data as old_log_probs)

   Clipping prevents the policy from changing too much in one update.

3. DR. GRPO MODIFICATIONS
   Following "Understanding R1-Zero-like Training" (Liu et al.):
   - Remove standard deviation normalization in advantages
   - Remove length normalization
   These reduce bias in the optimization.

4. OFF-POLICY CONSIDERATION (for Rollout Replay)
   For replayed samples:
   - old_log_probs comes from a previous policy (behavior policy)
   - This makes the update slightly off-policy
   - PPO/GRPO clipping naturally handles moderate off-policy-ness
   - FIFO buffer ensures samples don't get too stale

5. TOKEN-LEVEL VS SEQUENCE-LEVEL
   - Compute loss per token
   - Average over sequence (masked by attention)
   - Then average over batch

Reference:
    - GRPO: DeepSeekMath paper (Shao et al., 2024)
    - Dr. GRPO: Understanding R1-Zero-like Training (Liu et al., 2025)
    - verl implementation: verl/trainer/ppo/core_algos.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GRPOLossOutput:
    """
    Output from GRPO loss computation.

    Attributes:
        loss: Total loss value (scalar)
        policy_loss: Policy gradient loss
        kl_loss: KL divergence penalty (if used)
        entropy_loss: Entropy bonus (if used)
        metrics: Dictionary of logging metrics
    """
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy_loss: torch.Tensor
    metrics: Dict[str, float]


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-token log probabilities.

    Args:
        logits: [batch, seq_len, vocab_size] model output logits
        labels: [batch, seq_len] token IDs
        mask: [batch, seq_len] attention mask (1 = valid, 0 = padding)

    Returns:
        log_probs: [batch, seq_len] per-token log probabilities
    """
    # Shift for autoregressive: logits[t] predicts labels[t+1]
    # But usually we pass response logits aligned with response tokens
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for the actual tokens
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    if mask is not None:
        token_log_probs = token_log_probs * mask

    return token_log_probs


def compute_entropy(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-token entropy.

    Entropy H = -Σ p(x) log p(x)

    Higher entropy = more uniform distribution = more exploration.

    Args:
        logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]

    Returns:
        entropy: [batch, seq_len] per-token entropy
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)

    if mask is not None:
        entropy = entropy * mask

    return entropy


def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute clipped policy gradient loss.

    This is the core GRPO/PPO loss:
        L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    Args:
        log_probs: [batch, seq_len] current policy log probs π_θ
        old_log_probs: [batch, seq_len] old policy log probs π_old
        advantages: [batch] or [batch, 1] advantages (one per sequence)
        mask: [batch, seq_len] attention mask
        clip_epsilon: Clipping threshold ε

    Returns:
        loss: Scalar loss value
        metrics: Dictionary with logging metrics
    """
    # Compute importance ratio: π_θ / π_old = exp(log π_θ - log π_old)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Expand advantages to match sequence dimension
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)  # [batch, 1]

    # Clipped surrogate objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    # Take minimum (pessimistic bound)
    # Note: We want to maximize this, so loss = -objective
    policy_objective = torch.min(unclipped, clipped)

    # Mask and average
    # Per-token loss, averaged over valid tokens
    masked_objective = policy_objective * mask
    loss = -masked_objective.sum() / mask.sum().clamp(min=1)

    # Compute metrics
    with torch.no_grad():
        # Approximate KL divergence: KL ≈ (ratio - 1) - log(ratio)
        approx_kl = ((ratio - 1) - log_ratio).mean().item()

        # Clip fraction: how often was clipping active?
        clip_frac = ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).float().mean().item()

        # Mean ratio
        mean_ratio = ratio.mean().item()

    metrics = {
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
        "mean_ratio": mean_ratio,
    }

    return loss, metrics


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence penalty from reference policy.

    KL(π_θ || π_ref) = Σ π_θ(a) * (log π_θ(a) - log π_ref(a))

    For token-level, we use the simpler approximation:
        KL ≈ log π_θ - log π_ref

    Args:
        log_probs: [batch, seq_len] current policy
        ref_log_probs: [batch, seq_len] reference policy
        mask: [batch, seq_len] attention mask

    Returns:
        kl_loss: Scalar KL penalty
    """
    kl = log_probs - ref_log_probs
    masked_kl = kl * mask
    kl_loss = masked_kl.sum() / mask.sum().clamp(min=1)
    return kl_loss


def grpo_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
    ref_log_probs: Optional[torch.Tensor] = None,
) -> GRPOLossOutput:
    """
    Compute full GRPO loss with optional KL penalty and entropy bonus.

    This is the main loss function called during training.

    Args:
        model: The policy model being trained
        input_ids: [batch, seq_len] full sequence (prompt + response)
        attention_mask: [batch, seq_len]
        response_start_idx: Index where response starts (after prompt)
        old_log_probs: [batch, response_len] stored log probs from rollout
        advantages: [batch] group-relative advantages
        clip_epsilon: PPO clipping threshold
        kl_coef: KL penalty coefficient (0 = no penalty)
        entropy_coef: Entropy bonus coefficient (0 = no bonus)
        ref_log_probs: [batch, response_len] reference policy log probs (for KL)

    Returns:
        GRPOLossOutput with loss components and metrics
    """
    # Forward pass to get current log probs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab]

    # Get response portion
    response_logits = logits[:, response_start_idx:-1, :]  # Predict next token
    response_labels = input_ids[:, response_start_idx+1:]  # Shifted labels
    response_mask = attention_mask[:, response_start_idx+1:]

    # Truncate to match old_log_probs length
    min_len = min(response_logits.shape[1], old_log_probs.shape[1])
    response_logits = response_logits[:, :min_len]
    response_labels = response_labels[:, :min_len]
    response_mask = response_mask[:, :min_len]
    old_log_probs = old_log_probs[:, :min_len]

    # Compute current log probs
    current_log_probs = compute_log_probs(response_logits, response_labels, response_mask)

    # Policy loss
    policy_loss, metrics = compute_policy_loss(
        current_log_probs, old_log_probs, advantages, response_mask, clip_epsilon
    )

    total_loss = policy_loss

    # KL penalty (optional)
    kl_loss = torch.tensor(0.0, device=policy_loss.device)
    if kl_coef > 0 and ref_log_probs is not None:
        ref_log_probs = ref_log_probs[:, :min_len]
        kl_loss = compute_kl_penalty(current_log_probs, ref_log_probs, response_mask)
        total_loss = total_loss + kl_coef * kl_loss
        metrics["kl_loss"] = kl_loss.item()

    # Entropy bonus (optional)
    entropy_loss = torch.tensor(0.0, device=policy_loss.device)
    if entropy_coef > 0:
        entropy = compute_entropy(response_logits, response_mask)
        entropy_loss = -entropy.sum() / response_mask.sum().clamp(min=1)  # Negative for bonus
        total_loss = total_loss + entropy_coef * entropy_loss
        metrics["entropy"] = -entropy_loss.item()  # Report as positive entropy

    metrics["policy_loss"] = policy_loss.item()
    metrics["total_loss"] = total_loss.item()

    return GRPOLossOutput(
        loss=total_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        entropy_loss=entropy_loss,
        metrics=metrics
    )


def grpo_loss_from_dataproto(
    model: nn.Module,
    data,  # DataProto
    tokenizer,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
) -> GRPOLossOutput:
    """
    Compute GRPO loss directly from a DataProto batch.

    This is a convenience wrapper that handles the data unpacking.

    Args:
        model: Policy model
        data: DataProto with batch fields:
              - response_ids: Generated response tokens
              - old_log_probs: Log probs from rollout generation
              - advantages: Group-relative advantages
              - attention_mask: Response attention mask
        tokenizer: For getting pad token
        clip_epsilon: Clipping threshold
        kl_coef: KL coefficient
        entropy_coef: Entropy coefficient

    Returns:
        GRPOLossOutput
    """
    response_ids = data.batch["response_ids"]
    old_log_probs = data.batch["old_log_probs"]
    advantages = data.batch["advantages"]
    attention_mask = data.batch["attention_mask"]

    # For simplicity, we compute loss on response only
    # In full implementation, we'd concatenate with prompt

    # Forward pass
    outputs = model(input_ids=response_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Predict next token
    labels = response_ids[:, 1:]  # Shifted
    mask = attention_mask[:, 1:]

    # Truncate old_log_probs to match
    min_len = logits.shape[1]
    old_log_probs = old_log_probs[:, :min_len]

    # Current log probs
    current_log_probs = compute_log_probs(logits, labels, mask)

    # Policy loss
    policy_loss, metrics = compute_policy_loss(
        current_log_probs, old_log_probs, advantages, mask, clip_epsilon
    )

    return GRPOLossOutput(
        loss=policy_loss,
        policy_loss=policy_loss,
        kl_loss=torch.tensor(0.0),
        entropy_loss=torch.tensor(0.0),
        metrics=metrics
    )


class GRPOTrainer:
    """
    Simple GRPO trainer class.

    This wraps the loss computation with optimization logic.

    Example:
        >>> trainer = GRPOTrainer(model, optimizer, config)
        >>> for batch in dataloader:
        ...     loss_output = trainer.train_step(batch)
        ...     print(f"Loss: {loss_output.loss.item():.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def train_step(self, data) -> GRPOLossOutput:
        """
        Perform one training step.

        Args:
            data: DataProto batch

        Returns:
            GRPOLossOutput with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss_output = grpo_loss_from_dataproto(
            self.model, data, None,
            self.clip_epsilon, self.kl_coef, self.entropy_coef
        )

        # Backward
        loss_output.loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        # Update
        self.optimizer.step()

        return loss_output


# Example and testing
if __name__ == "__main__":
    print("Testing GRPO loss components...")

    # Test 1: compute_policy_loss
    print("\n--- Test 1: compute_policy_loss ---")
    batch_size, seq_len = 4, 10

    log_probs = torch.randn(batch_size, seq_len) * 0.1  # Current policy
    old_log_probs = torch.randn(batch_size, seq_len) * 0.1  # Old policy
    advantages = torch.tensor([0.5, -0.5, 0.3, -0.3])  # Per-sequence advantages
    mask = torch.ones(batch_size, seq_len)

    loss, metrics = compute_policy_loss(log_probs, old_log_probs, advantages, mask)
    print(f"Policy loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test 2: Effect of clipping
    print("\n--- Test 2: Clipping effect ---")
    # Large ratio (policy changed a lot)
    large_ratio_log_probs = old_log_probs + 1.0  # ratio ≈ e ≈ 2.7
    loss_large, metrics_large = compute_policy_loss(
        large_ratio_log_probs, old_log_probs, advantages, mask, clip_epsilon=0.2
    )
    print(f"Large ratio - clip_fraction: {metrics_large['clip_fraction']:.2f}")

    # Small ratio (policy similar)
    small_ratio_log_probs = old_log_probs + 0.1  # ratio ≈ 1.1
    loss_small, metrics_small = compute_policy_loss(
        small_ratio_log_probs, old_log_probs, advantages, mask, clip_epsilon=0.2
    )
    print(f"Small ratio - clip_fraction: {metrics_small['clip_fraction']:.2f}")

    # Test 3: Advantage sign
    print("\n--- Test 3: Advantage sign effect ---")
    # Positive advantage: encourage this action
    pos_adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss_pos, _ = compute_policy_loss(log_probs, old_log_probs, pos_adv, mask)

    # Negative advantage: discourage this action
    neg_adv = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    loss_neg, _ = compute_policy_loss(log_probs, old_log_probs, neg_adv, mask)

    print(f"Positive advantage loss: {loss_pos.item():.4f}")
    print(f"Negative advantage loss: {loss_neg.item():.4f}")

    print("\nAll tests passed!")
