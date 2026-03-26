"""
Configuration for DOTS-RR Training with verl

This module contains all hyperparameters organized into logical groups.
Each parameter includes its purpose and typical values from the paper.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model and tokenizer configuration."""

    # Model identifier (HuggingFace model ID or local path)
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    # Precision for model loading
    # Options: "float32", "float16", "bfloat16"
    torch_dtype: str = "float16"

    # Device mapping strategy
    # Options: "auto", "cuda:0", "cpu"
    device_map: str = "auto"

    # Maximum sequence length (prompt + response)
    # Paper uses 3072 for Qwen2.5-Math, 4096 for Qwen2.5-3B
    max_seq_length: int = 3072

    # Maximum prompt length
    max_prompt_length: int = 1024


@dataclass
class DOTSConfig:
    """
    Difficulty-targeted Online Data Selection (DOTS) configuration.

    DOTS selects training questions based on their adaptive difficulty,
    prioritizing questions with difficulty close to ALPHA (typically 0.5).

    Key insight: Questions where the model succeeds ~50% of the time
    provide the strongest gradient signal (see Theorem 1 in paper).
    """

    # Reference set size (K in the paper)
    # Paper uses 128-256. Larger K = better difficulty estimates but more rollout cost
    reference_set_size: int = 256

    # Target difficulty (α in the paper)
    # 0.5 = questions the model answers correctly ~50% of the time
    # These provide maximal gradient signal
    alpha: float = 0.5

    # Sampling temperature (τ in the paper)
    # Lower τ = sharper selection (more deterministic, closer to greedy)
    # Higher τ = softer selection (more uniform/random)
    # Paper uses 1e-3 (very sharp)
    tau: float = 1e-3

    # Re-estimate difficulties every μ steps
    # Paper uses 2. More frequent = better adaptation but more cost
    reestimate_every: int = 2

    # Embedding batch size for computing question embeddings
    embedding_batch_size: int = 8


@dataclass
class RolloutReplayConfig:
    """
    Rollout Replay (RR) configuration.

    RR reuses recent rollouts to reduce per-step computation cost.
    Instead of generating fresh rollouts for all B questions,
    we generate for δ·B and retrieve (1-δ)·B from a replay buffer.

    The buffer stores complete rollout data including:
    - Generated tokens
    - Log probabilities (behavior policy)
    - Computed advantages
    - Rewards

    This makes GRPO slightly off-policy, but PPO clipping handles it.
    """

    # Fresh rollout fraction (δ in the paper)
    # 0.5 = half fresh rollouts, half from replay buffer
    # Paper uses 0.5
    delta: float = 0.5

    # Replay buffer capacity (C in the paper)
    # Maximum number of question-rollout groups to store
    # Paper uses 256-512
    buffer_capacity: int = 512

    # Replay selection strategy
    # Options: "random", "fifo" (oldest first), "difficulty" (closest to alpha)
    replay_strategy: str = "random"

    # Only store rollouts where avg_reward is not 0 or 1
    # (i.e., informative samples with gradient signal)
    filter_informative: bool = True


@dataclass
class RolloutConfig:
    """
    Rollout generation configuration.

    For each question, we generate G independent completions.
    These are used to:
    1. Compute adaptive difficulty: d_q = (1/G) * Σ(1 - r_i)
    2. Train the policy via GRPO
    """

    # Number of rollouts per question (G in the paper)
    # Paper uses 8
    num_rollouts: int = 8

    # Maximum new tokens to generate per rollout
    # Paper uses variable lengths; we use 1024 for demo
    max_new_tokens: int = 1024

    # Sampling temperature for generation
    # Paper uses 0.6
    temperature: float = 0.6

    # Top-p (nucleus) sampling threshold
    # Paper uses 0.95
    top_p: float = 0.95

    # Whether to use sampling (True) or greedy decoding (False)
    # Must be True for diverse rollouts
    do_sample: bool = True


@dataclass
class GRPOConfig:
    """
    Group Relative Policy Optimization (GRPO) configuration.

    GRPO normalizes advantages within each question's rollout group,
    avoiding the need for a separate value network (unlike PPO).

    Key equations:
        Advantage: A_i = r_i - mean(r_1, ..., r_G)
        Ratio: ratio = π_θ(a|s) / π_old(a|s)
        Loss: -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    Note: We follow Dr. GRPO and remove std normalization (introduces bias).
    """

    # PPO clipping epsilon (ε)
    # Controls how much the policy can change per update
    clip_epsilon: float = 0.2

    # Learning rate
    # Paper uses 1e-6
    learning_rate: float = 1e-6

    # KL penalty coefficient (β)
    # Penalizes deviation from reference policy
    # Paper removes KL penalty (set to 0)
    kl_coef: float = 0.0

    # Entropy bonus coefficient
    # Encourages exploration
    entropy_coef: float = 0.0

    # Whether to normalize advantages by std within group
    # Dr. GRPO recommends False (std normalization introduces bias)
    normalize_advantage_by_std: bool = False

    # Gradient clipping max norm
    max_grad_norm: float = 1.0

    # Number of gradient accumulation steps
    gradient_accumulation_steps: int = 1


@dataclass
class TrainingConfig:
    """
    Overall training configuration.
    """

    # Training batch size (B in the paper)
    # Number of questions per training step
    # Paper uses 512 (we use smaller for demo)
    batch_size: int = 8

    # Total training steps (T in the paper)
    # Paper uses 60
    total_steps: int = 60

    # Logging frequency
    log_every: int = 1

    # Checkpoint saving frequency
    save_every: int = 10

    # Output directory for checkpoints
    output_dir: str = "./output_dots_rr_verl"

    # Random seed for reproducibility
    seed: int = 42

    # Whether to use mixed precision training
    use_amp: bool = True


@dataclass
class DataConfig:
    """
    Dataset configuration.
    """

    # Dataset name (HuggingFace dataset ID)
    dataset_name: str = "trl-lib/DeepMath-103K"

    # Dataset split to use
    split: str = "train"

    # Number of samples to use (None = all)
    # Paper uses 8K-10K subsets
    num_samples: Optional[int] = 512

    # Shuffle seed
    shuffle_seed: int = 42


@dataclass
class Config:
    """
    Master configuration combining all sub-configs.

    Usage:
        config = Config()
        print(config.dots.alpha)  # 0.5
        print(config.rollout.num_rollouts)  # 8
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    dots: DOTSConfig = field(default_factory=DOTSConfig)
    replay: RolloutReplayConfig = field(default_factory=RolloutReplayConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0.0 <= self.dots.alpha <= 1.0, "alpha must be in [0, 1]"
        assert 0.0 < self.replay.delta <= 1.0, "delta must be in (0, 1]"
        assert self.dots.tau > 0, "tau must be positive"
        assert self.rollout.num_rollouts >= 2, "need at least 2 rollouts for GRPO"

    def print_config(self):
        """Pretty print all configuration values."""
        print("=" * 60)
        print("DOTS-RR Configuration")
        print("=" * 60)

        print("\n[Model]")
        print(f"  model_name: {self.model.model_name}")
        print(f"  torch_dtype: {self.model.torch_dtype}")
        print(f"  max_seq_length: {self.model.max_seq_length}")

        print("\n[DOTS - Difficulty-targeted Online Data Selection]")
        print(f"  reference_set_size (K): {self.dots.reference_set_size}")
        print(f"  target_difficulty (α): {self.dots.alpha}")
        print(f"  sampling_temperature (τ): {self.dots.tau}")
        print(f"  reestimate_every (μ): {self.dots.reestimate_every}")

        print("\n[Rollout Replay]")
        print(f"  fresh_fraction (δ): {self.replay.delta}")
        print(f"  buffer_capacity (C): {self.replay.buffer_capacity}")
        print(f"  replay_strategy: {self.replay.replay_strategy}")
        print(f"  filter_informative: {self.replay.filter_informative}")

        print("\n[Rollout Generation]")
        print(f"  num_rollouts (G): {self.rollout.num_rollouts}")
        print(f"  max_new_tokens: {self.rollout.max_new_tokens}")
        print(f"  temperature: {self.rollout.temperature}")
        print(f"  top_p: {self.rollout.top_p}")

        print("\n[GRPO]")
        print(f"  clip_epsilon: {self.grpo.clip_epsilon}")
        print(f"  learning_rate: {self.grpo.learning_rate}")
        print(f"  kl_coef: {self.grpo.kl_coef}")
        print(f"  normalize_advantage_by_std: {self.grpo.normalize_advantage_by_std}")

        print("\n[Training]")
        print(f"  batch_size (B): {self.training.batch_size}")
        print(f"  total_steps (T): {self.training.total_steps}")
        print(f"  output_dir: {self.training.output_dir}")

        print("\n[Data]")
        print(f"  dataset_name: {self.data.dataset_name}")
        print(f"  num_samples: {self.data.num_samples}")

        print("=" * 60)


# Default configuration instance
def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


# Paper configuration (matches experimental setup)
def get_paper_config() -> Config:
    """
    Get configuration matching the paper's experimental setup.

    Note: This uses larger batch sizes and more samples.
    Requires significant GPU memory (8x L40S or 8x A100).
    """
    return Config(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
            max_seq_length=3072,
        ),
        dots=DOTSConfig(
            reference_set_size=256,
            alpha=0.5,
            tau=1e-3,
            reestimate_every=2,
        ),
        replay=RolloutReplayConfig(
            delta=0.5,
            buffer_capacity=512,
            replay_strategy="random",
        ),
        rollout=RolloutConfig(
            num_rollouts=8,
            max_new_tokens=3072,
            temperature=0.6,
            top_p=0.95,
        ),
        grpo=GRPOConfig(
            clip_epsilon=0.2,
            learning_rate=1e-6,
            kl_coef=0.0,
        ),
        training=TrainingConfig(
            batch_size=512,
            total_steps=60,
            output_dir="./output_dots_rr_paper",
        ),
        data=DataConfig(
            dataset_name="trl-lib/DeepMath-103K",
            num_samples=10240,
        ),
    )


if __name__ == "__main__":
    # Demo: print default configuration
    config = get_default_config()
    config.print_config()
