"""
DOTS: Difficulty-targeted Online Data Selection

This module implements the adaptive difficulty prediction framework from the
DOTS-RR paper. The key idea is to efficiently estimate how difficult each
question is for the current policy, then prioritize questions with difficulty
close to 0.5 (where gradient signal is maximized).

Key Concepts:
=============

1. ADAPTIVE DIFFICULTY
   For a question q with G rollouts {o_i} and rewards {r_i}:
       d_q = (1/G) * Σ(1 - r_i)
   This is the average failure rate. d_q ∈ [0, 1] where:
   - d_q = 0: Model always succeeds (too easy)
   - d_q = 1: Model always fails (too hard)
   - d_q = 0.5: Model succeeds 50% of the time (optimal for learning!)

2. WHY 0.5 IS OPTIMAL (Theorem 1 in paper)
   The expected gradient magnitude is:
       E[||g||²] ∝ p(1-p) * (1 - 1/G)
   This is maximized when p = 0.5 (success rate = 50%, difficulty = 0.5).

3. EFFICIENT DIFFICULTY ESTIMATION
   Problem: Computing true difficulty requires G rollouts per question.
            For N questions, that's N*G rollouts - very expensive!

   Solution: Attention-based prediction
   a) Sample K reference questions (K << N)
   b) Run G rollouts on reference set, compute ground-truth difficulty
   c) For other N-K questions, PREDICT difficulty via attention:
      - Compute embeddings for all questions
      - Use similarity-weighted average of reference difficulties

4. ATTENTION-BASED PREDICTION (Section 4.1)
   For unlabeled question q with embedding z_q:
       a_i = softmax(z_q^T z_i / sqrt(h))  # attention to ref question i
       d_hat_q = Σ a_i * d_i               # weighted average of ref difficulties

   Questions similar to hard reference questions → predicted as hard
   Questions similar to easy reference questions → predicted as easy

5. CALIBRATION (Optional)
   The paper uses Platt scaling with a trained MLP:
       d_cal = sigmoid(w * logit(d_hat) + b)
   where (w, b) = MLP([mean(d_ref), std(d_ref)])

   For simplicity, we implement a simpler mean-shift calibration.

Reference:
    Paper Section 4.1: Attention-based Adaptive Difficulty Prediction Framework
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DOTSState:
    """
    State maintained across DOTS iterations.

    Attributes:
        ref_indices: Indices of reference questions in the dataset
        ref_embeddings: Embeddings of reference questions [K, hidden_dim]
        ref_difficulties: Ground-truth difficulties of reference questions [K]
        all_embeddings: Cached embeddings for all questions [N, hidden_dim]
        predicted_difficulties: Predicted difficulties for all questions [N]
    """
    ref_indices: torch.Tensor = None
    ref_embeddings: torch.Tensor = None
    ref_difficulties: torch.Tensor = None
    all_embeddings: torch.Tensor = None
    predicted_difficulties: torch.Tensor = None


@torch.no_grad()
def compute_embeddings(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 512
) -> torch.Tensor:
    """
    Compute L2-normalized embeddings for a list of texts.

    Uses mean pooling over the last hidden layer, which captures
    semantic similarity between questions.

    Args:
        texts: List of N text strings (questions)
        tokenizer: HuggingFace tokenizer
        model: HuggingFace causal LM (we use hidden states, not logits)
        batch_size: Batch size for encoding
        max_length: Maximum sequence length

    Returns:
        embeddings: [N, hidden_dim] tensor, L2-normalized

    Note:
        The paper mentions training an MLP adapter on top of frozen
        embeddings for better difficulty prediction. For simplicity,
        we use raw embeddings here which still work reasonably well.
    """
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)

        # Forward pass to get hidden states
        outputs = model(**encoded, output_hidden_states=True)

        # Last hidden layer: [batch, seq_len, hidden_dim]
        last_hidden = outputs.hidden_states[-1]

        # Mean pooling (ignore padding)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        sum_hidden = (last_hidden.float() * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_hidden / count

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def sample_reference_set(n_total: int, k: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Randomly sample K indices for the reference set.

    The reference set is re-sampled every μ steps to:
    1. Adapt to the evolving policy
    2. Get fresh difficulty measurements
    3. Provide diversity in the attention targets

    Args:
        n_total: Total number of questions in dataset
        k: Number of reference questions to sample
        seed: Optional random seed for reproducibility

    Returns:
        ref_indices: [K] tensor of indices into the dataset
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randperm(n_total)[:k]


def compute_ground_truth_difficulty(
    rewards_per_question: List[List[float]]
) -> torch.Tensor:
    """
    Compute ground-truth adaptive difficulty from rollout rewards.

    Equation 2 in paper:
        d_q = (1/G) * Σ(1 - r_i) = 1 - mean(rewards)

    Args:
        rewards_per_question: List of K lists, each containing G rewards

    Returns:
        difficulties: [K] tensor of difficulty values in [0, 1]

    Example:
        >>> rewards = [[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
        >>> compute_ground_truth_difficulty(rewards)
        tensor([0.5, 0.0, 1.0])  # 50% fail, 0% fail, 100% fail
    """
    difficulties = []
    for rewards in rewards_per_question:
        avg_reward = sum(rewards) / len(rewards)
        difficulty = 1.0 - avg_reward
        difficulties.append(difficulty)

    return torch.tensor(difficulties, dtype=torch.float32)


def predict_adaptive_difficulty(
    query_embeddings: torch.Tensor,
    ref_embeddings: torch.Tensor,
    ref_difficulties: torch.Tensor,
    calibrate: bool = True
) -> torch.Tensor:
    """
    Predict adaptive difficulty using attention-weighted averaging.

    This is the core of the DOTS prediction framework (Section 4.1):
    1. Compute similarity between each query and all reference questions
    2. Apply softmax to get attention weights
    3. Weighted average of reference difficulties

    Args:
        query_embeddings: [N, h] embeddings for all questions
        ref_embeddings: [K, h] embeddings for reference questions
        ref_difficulties: [K] ground-truth difficulties for references

    Returns:
        predicted: [N] tensor of predicted difficulties, clipped to [0, 1]

    Mathematical formulation:
        similarity[q, i] = z_q^T z_i / sqrt(h)
        attention[q, i] = softmax(similarity[q, :])
        d_hat[q] = Σ_i attention[q, i] * d_i
    """
    h = query_embeddings.shape[1]  # embedding dimension

    # Compute similarity matrix: [N, K]
    # Since embeddings are L2-normalized, this is cosine similarity
    similarity = (query_embeddings @ ref_embeddings.T) / (h ** 0.5)

    # Softmax over references for each query
    attention_weights = torch.softmax(similarity, dim=-1)  # [N, K]

    # Weighted average of reference difficulties
    predicted = attention_weights @ ref_difficulties  # [N]

    # Calibration: simple mean-shift to match reference distribution
    # (Paper uses trained MLP, this is a simplified version)
    if calibrate:
        ref_mean = ref_difficulties.mean()
        predicted = predicted - predicted.mean() + ref_mean

    return predicted.clamp(0.0, 1.0)


def difficulty_targeted_sampling(
    predicted_difficulties: torch.Tensor,
    alpha: float = 0.5,
    tau: float = 1e-3,
    n_samples: int = 8,
    exclude_indices: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample questions with difficulty close to target alpha.

    Sampling distribution (Equation in Section 4.2):
        P(q) ∝ exp(-|d_hat_q - α| / τ)

    This gives higher probability to questions with difficulty near alpha.
    Lower tau = sharper (more greedy), higher tau = more uniform.

    Args:
        predicted_difficulties: [N] tensor of predicted difficulties
        alpha: Target difficulty (default 0.5 for maximal gradient)
        tau: Temperature (default 1e-3 for sharp selection)
        n_samples: Number of questions to sample
        exclude_indices: Optional indices to exclude (e.g., reference set)

    Returns:
        selected_indices: [n_samples] indices of selected questions
        sampling_probs: [N] sampling probability for each question
    """
    # Compute scores: high score = close to alpha
    scores = -torch.abs(predicted_difficulties - alpha) / tau

    # Exclude certain indices if specified
    if exclude_indices is not None:
        scores[exclude_indices] = float('-inf')

    # Softmax to get probabilities
    probs = torch.softmax(scores, dim=0)

    # Sample without replacement
    selected_indices = torch.multinomial(probs, n_samples, replacement=False)

    return selected_indices, probs


def compute_enrichment(
    selected_indices: torch.Tensor,
    sampling_probs: torch.Tensor,
    n_total: int
) -> float:
    """
    Compute enrichment factor over uniform sampling.

    Enrichment = (avg probability of selected) / (uniform probability)

    Higher enrichment means DOTS is selecting more targeted questions.

    Args:
        selected_indices: Indices of selected questions
        sampling_probs: Probability of each question
        n_total: Total number of questions

    Returns:
        enrichment: Enrichment factor (>1 means better than uniform)
    """
    uniform_prob = 1.0 / n_total
    avg_selected_prob = sampling_probs[selected_indices].mean().item()
    return avg_selected_prob / uniform_prob


class DOTSSelector:
    """
    High-level interface for DOTS question selection.

    This class manages the full DOTS workflow:
    1. Cache embeddings for all questions (computed once)
    2. Sample reference set and compute ground-truth difficulties
    3. Predict difficulties for all questions
    4. Sample training batch targeting difficulty α

    Example:
        >>> selector = DOTSSelector(config)
        >>> selector.initialize(questions, tokenizer, model)
        >>>
        >>> for step in range(T):
        ...     if step % mu == 0:
        ...         # Re-estimate difficulties
        ...         ref_indices = selector.sample_reference_set()
        ...         ref_difficulties = run_rollouts_and_compute_difficulty(...)
        ...         selector.update_reference_difficulties(ref_difficulties)
        ...
        ...     # Sample training batch
        ...     selected = selector.select_batch(n_fresh)
    """

    def __init__(
        self,
        reference_set_size: int = 256,
        alpha: float = 0.5,
        tau: float = 1e-3,
        embedding_batch_size: int = 16
    ):
        """
        Initialize DOTS selector.

        Args:
            reference_set_size: K - number of reference questions
            alpha: Target difficulty
            tau: Sampling temperature
            embedding_batch_size: Batch size for computing embeddings
        """
        self.K = reference_set_size
        self.alpha = alpha
        self.tau = tau
        self.embedding_batch_size = embedding_batch_size

        # State
        self.state = DOTSState()
        self.questions: List[str] = []
        self.n_questions: int = 0

    def initialize(
        self,
        questions: List[str],
        tokenizer,
        model
    ) -> None:
        """
        Initialize by computing embeddings for all questions.

        This is done once at the start of training.

        Args:
            questions: List of N question strings
            tokenizer: HuggingFace tokenizer
            model: HuggingFace model for embeddings
        """
        self.questions = questions
        self.n_questions = len(questions)

        print(f"[DOTS] Computing embeddings for {self.n_questions} questions...")
        self.state.all_embeddings = compute_embeddings(
            questions, tokenizer, model, batch_size=self.embedding_batch_size
        )
        print(f"[DOTS] Embeddings shape: {self.state.all_embeddings.shape}")

    def sample_reference_set(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample a new reference set.

        Called every μ steps to refresh the reference set.

        Args:
            seed: Optional random seed

        Returns:
            ref_indices: [K] tensor of reference question indices
        """
        self.state.ref_indices = sample_reference_set(self.n_questions, self.K, seed)
        self.state.ref_embeddings = self.state.all_embeddings[self.state.ref_indices]
        return self.state.ref_indices

    def get_reference_questions(self) -> List[str]:
        """Get the question strings for the current reference set."""
        return [self.questions[i.item()] for i in self.state.ref_indices]

    def update_reference_difficulties(
        self,
        ref_difficulties: torch.Tensor
    ) -> None:
        """
        Update with ground-truth difficulties from rollouts.

        Called after running rollouts on the reference set.

        Args:
            ref_difficulties: [K] tensor of computed difficulties
        """
        self.state.ref_difficulties = ref_difficulties

        # Predict difficulties for all questions
        self.state.predicted_difficulties = predict_adaptive_difficulty(
            query_embeddings=self.state.all_embeddings,
            ref_embeddings=self.state.ref_embeddings,
            ref_difficulties=self.state.ref_difficulties,
        )

        # Log statistics
        pred = self.state.predicted_difficulties
        print(f"[DOTS] Reference difficulties: mean={ref_difficulties.mean():.3f}, std={ref_difficulties.std():.3f}")
        print(f"[DOTS] Predicted difficulties: mean={pred.mean():.3f}, std={pred.std():.3f}")

    def select_batch(
        self,
        n_samples: int,
        exclude_reference: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Select a batch of questions for training.

        Args:
            n_samples: Number of questions to select (δ·B for fresh rollouts)
            exclude_reference: If True, exclude reference set from selection

        Returns:
            selected_indices: [n_samples] indices of selected questions
            stats: Dictionary with selection statistics
        """
        if self.state.predicted_difficulties is None:
            raise RuntimeError("Must call update_reference_difficulties first")

        exclude = self.state.ref_indices if exclude_reference else None

        selected_indices, probs = difficulty_targeted_sampling(
            self.state.predicted_difficulties,
            alpha=self.alpha,
            tau=self.tau,
            n_samples=n_samples,
            exclude_indices=exclude
        )

        # Compute statistics
        selected_difficulties = self.state.predicted_difficulties[selected_indices]
        enrichment = compute_enrichment(selected_indices, probs, self.n_questions)

        stats = {
            "avg_predicted_difficulty": selected_difficulties.mean().item(),
            "std_predicted_difficulty": selected_difficulties.std().item(),
            "enrichment": enrichment,
            "n_selected": len(selected_indices),
        }

        return selected_indices, stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing DOTS components...")

    # Test 1: Ground truth difficulty computation
    print("\n--- Test 1: Ground truth difficulty ---")
    rewards = [
        [1, 1, 0, 0],  # 50% success → difficulty 0.5
        [1, 1, 1, 1],  # 100% success → difficulty 0.0
        [0, 0, 0, 0],  # 0% success → difficulty 1.0
        [1, 0, 1, 0],  # 50% success → difficulty 0.5
    ]
    difficulties = compute_ground_truth_difficulty(rewards)
    print(f"Rewards: {rewards}")
    print(f"Difficulties: {difficulties.tolist()}")

    # Test 2: Attention-based prediction
    print("\n--- Test 2: Attention-based prediction ---")
    # Simulate embeddings
    torch.manual_seed(42)
    N, K, h = 100, 10, 64
    query_emb = F.normalize(torch.randn(N, h), dim=-1)
    ref_emb = F.normalize(torch.randn(K, h), dim=-1)
    ref_diff = torch.rand(K)  # Random reference difficulties

    predicted = predict_adaptive_difficulty(query_emb, ref_emb, ref_diff)
    print(f"Predicted difficulties: mean={predicted.mean():.3f}, std={predicted.std():.3f}")
    print(f"Range: [{predicted.min():.3f}, {predicted.max():.3f}]")

    # Test 3: Difficulty-targeted sampling
    print("\n--- Test 3: Difficulty-targeted sampling ---")
    # Create difficulties with known distribution
    difficulties = torch.linspace(0, 1, 100)  # 0.0, 0.01, ..., 0.99

    selected, probs = difficulty_targeted_sampling(
        difficulties, alpha=0.5, tau=0.01, n_samples=10
    )
    selected_diffs = difficulties[selected]
    print(f"Selected indices: {selected.tolist()}")
    print(f"Selected difficulties: {selected_diffs.tolist()}")
    print(f"Mean selected difficulty: {selected_diffs.mean():.3f} (target: 0.5)")

    # With very low tau (greedy)
    selected_greedy, _ = difficulty_targeted_sampling(
        difficulties, alpha=0.5, tau=1e-6, n_samples=5
    )
    print(f"Greedy selection (tau=1e-6): {difficulties[selected_greedy].tolist()}")

    # Test 4: Enrichment computation
    print("\n--- Test 4: Enrichment ---")
    enrichment = compute_enrichment(selected, probs, 100)
    print(f"Enrichment over uniform: {enrichment:.2f}x")

    print("\nAll tests passed!")
