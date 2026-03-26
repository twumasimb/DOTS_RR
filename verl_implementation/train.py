"""
DOTS-RR Training Script

This is the main training script that implements the full DOTS-RR algorithm:
- Difficulty-targeted Online Data Selection (DOTS)
- Rollout Replay (RR)
- GRPO policy optimization

Training Loop Overview:
=======================

For each training step t = 1, ..., T:

1. [Every μ steps] REFERENCE SET UPDATE
   - Sample K reference questions
   - Generate G rollouts per reference question
   - Compute ground-truth difficulties d_i = 1 - mean(rewards)
   - Predict difficulties for all N questions via attention

2. DIFFICULTY-TARGETED SAMPLING
   - Sample δ·B questions with P(q) ∝ exp(-|d_hat_q - α| / τ)
   - Questions with difficulty ≈ 0.5 are prioritized

3. FRESH ROLLOUT GENERATION
   - Generate G rollouts for the δ·B selected questions
   - Compute rewards and advantages

4. ROLLOUT REPLAY
   - Sample (1-δ)·B questions from replay buffer
   - Combine fresh + replay into training batch

5. GRPO UPDATE
   - Compute GRPO loss on combined batch
   - Update policy parameters

6. BUFFER UPDATE
   - Add informative fresh rollouts to replay buffer
   - FIFO eviction if over capacity

Usage:
    python train.py

    # Or with custom config:
    python train.py --batch_size 16 --total_steps 100

Reference:
    Paper: "Improving Data Efficiency for LLM Reinforcement Fine-tuning
            Through Difficulty-targeted Online Data Selection and Rollout Replay"
    Code: https://github.com/ASTRAL-Group/data-efficient-llm-rl/
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
from config import Config, get_default_config
from data_proto import DataProto
from replay_buffer import ReplayBuffer, combine_fresh_and_replay
from difficulty_predictor import DOTSSelector, compute_ground_truth_difficulty
from rollout_generator import generate_rollout_batch, filter_informative_rollouts
from grpo_loss import GRPOTrainer
from utils import (
    set_seed, get_device, count_parameters, format_number,
    Logger, Checkpointer, Timer, print_banner, print_section
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DOTS-RR Training")

    # Model
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name or path")

    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size (questions)")
    parser.add_argument("--total_steps", type=int, default=None,
                       help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")

    # DOTS
    parser.add_argument("--reference_set_size", type=int, default=None,
                       help="Reference set size K")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Target difficulty")
    parser.add_argument("--tau", type=float, default=None,
                       help="Sampling temperature")

    # Rollout Replay
    parser.add_argument("--delta", type=float, default=None,
                       help="Fresh rollout fraction")
    parser.add_argument("--buffer_capacity", type=int, default=None,
                       help="Replay buffer capacity")

    # Data
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of training samples")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    return parser.parse_args()


def load_data(config: Config):
    """
    Load and prepare the training dataset.

    Returns:
        Tuple of (questions, solutions, dataset)
    """
    print(f"Loading dataset: {config.data.dataset_name}")

    dataset = load_dataset(config.data.dataset_name, split=config.data.split)

    # Subsample if specified
    if config.data.num_samples is not None:
        dataset = dataset.shuffle(seed=config.data.shuffle_seed)
        dataset = dataset.select(range(min(config.data.num_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")

    # Extract questions and solutions
    # Adjust field names based on your dataset format
    questions = []
    solutions = []

    for example in dataset:
        # Handle different dataset formats
        if "prompt" in example:
            # Format: {"prompt": [{"role": "user", "content": "..."}], "solution": "..."}
            if isinstance(example["prompt"], list):
                question = example["prompt"][0]["content"]
            else:
                question = example["prompt"]
        elif "question" in example:
            question = example["question"]
        elif "problem" in example:
            question = example["problem"]
        else:
            raise ValueError(f"Unknown dataset format: {example.keys()}")

        if "solution" in example:
            solution = example["solution"]
        elif "answer" in example:
            solution = example["answer"]
        else:
            raise ValueError(f"Unknown solution field: {example.keys()}")

        questions.append(question)
        solutions.append(solution)

    print(f"Loaded {len(questions)} questions")
    print(f"Sample question: {questions[0][:100]}...")
    print(f"Sample solution: {solutions[0]}")

    return questions, solutions, dataset


def load_model(config: Config, device: str):
    """
    Load model and tokenizer.

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch_dtype,
        device_map=config.model.device_map,
    )

    print(f"Model parameters: {format_number(count_parameters(model))}")
    print(f"Device: {next(model.parameters()).device}")

    return model, tokenizer


def training_step(
    step: int,
    config: Config,
    model,
    tokenizer,
    optimizer,
    questions: list,
    solutions: list,
    dots_selector: DOTSSelector,
    replay_buffer: ReplayBuffer,
    timer: Timer,
    logger: Logger,
) -> dict:
    """
    Execute one training step of DOTS-RR.

    This implements the full algorithm from the paper.
    """
    device = next(model.parameters()).device
    metrics = {"step": step}

    # =========================================================================
    # STEP 1: Reference Set Update (every μ steps)
    # =========================================================================
    if step == 1 or step % config.dots.reestimate_every == 0:
        print_section(f"Step {step}: Updating Reference Set")
        timer.start("reference_update")

        # Sample new reference set
        ref_indices = dots_selector.sample_reference_set(seed=step)
        ref_questions = [questions[i.item()] for i in ref_indices]
        ref_solutions = [solutions[i.item()] for i in ref_indices]

        print(f"  Reference set size: {len(ref_indices)}")

        # Generate rollouts for reference set
        timer.start("reference_rollouts")
        model.eval()

        ref_rewards_per_q = []
        for i, (q, sol) in enumerate(zip(ref_questions, ref_solutions)):
            # Generate rollouts (simplified - in practice batch this)
            completions, _, _, _ = generate_rollouts_for_question_simple(
                q, tokenizer, model, config.rollout.num_rollouts,
                config.rollout.max_new_tokens, config.rollout.temperature
            )

            # Compute rewards
            from reward_functions import compute_reward
            rewards = [compute_reward(c, sol) for c in completions]
            ref_rewards_per_q.append(rewards)

            if i < 2:  # Print first 2 examples
                difficulty = 1.0 - sum(rewards) / len(rewards)
                print(f"    Q{i}: difficulty={difficulty:.3f}, rewards={rewards}")

        timer.stop("reference_rollouts")

        # Compute ground-truth difficulties
        ref_difficulties = compute_ground_truth_difficulty(ref_rewards_per_q)
        dots_selector.update_reference_difficulties(ref_difficulties)

        timer.stop("reference_update")
        metrics["ref_difficulty_mean"] = ref_difficulties.mean().item()
        metrics["ref_difficulty_std"] = ref_difficulties.std().item()

    # =========================================================================
    # STEP 2: Difficulty-Targeted Sampling
    # =========================================================================
    print_section(f"Step {step}: Selecting Training Batch")

    n_fresh = int(config.replay.delta * config.training.batch_size)
    n_replay = config.training.batch_size - n_fresh

    timer.start("dots_sampling")
    selected_indices, selection_stats = dots_selector.select_batch(n_fresh)
    timer.stop("dots_sampling")

    selected_questions = [questions[i.item()] for i in selected_indices]
    selected_solutions = [solutions[i.item()] for i in selected_indices]

    print(f"  Selected {n_fresh} questions for fresh rollouts")
    print(f"  Avg predicted difficulty: {selection_stats['avg_predicted_difficulty']:.3f}")
    print(f"  Enrichment over uniform: {selection_stats['enrichment']:.2f}x")

    metrics.update({f"dots_{k}": v for k, v in selection_stats.items()})

    # =========================================================================
    # STEP 3: Generate Fresh Rollouts
    # =========================================================================
    print_section(f"Step {step}: Generating Fresh Rollouts")

    timer.start("fresh_rollouts")
    model.eval()

    fresh_batch = generate_rollout_batch(
        questions=selected_questions,
        solutions=selected_solutions,
        indices=selected_indices.tolist(),
        tokenizer=tokenizer,
        model=model,
        num_rollouts=config.rollout.num_rollouts,
        max_new_tokens=config.rollout.max_new_tokens,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )
    timer.stop("fresh_rollouts")

    fresh_rewards = fresh_batch.batch["rewards"]
    print(f"  Generated {len(fresh_batch)} rollouts")
    print(f"  Avg reward: {fresh_rewards.mean():.3f}")

    metrics["fresh_reward_mean"] = fresh_rewards.mean().item()

    # =========================================================================
    # STEP 4: Rollout Replay
    # =========================================================================
    print_section(f"Step {step}: Rollout Replay")

    timer.start("replay")
    combined_batch, used_replay = combine_fresh_and_replay(
        fresh_batch=fresh_batch,
        replay_buffer=replay_buffer,
        delta=config.replay.delta,
        batch_size=config.training.batch_size,
    )
    timer.stop("replay")

    if used_replay:
        print(f"  Combined {n_fresh} fresh + {n_replay} replay = {len(combined_batch) // config.rollout.num_rollouts} questions")
    else:
        print(f"  Buffer not ready, using {n_fresh} fresh rollouts only")

    metrics["used_replay"] = int(used_replay)
    metrics["batch_size_actual"] = len(combined_batch) // config.rollout.num_rollouts

    # =========================================================================
    # STEP 5: GRPO Update
    # =========================================================================
    print_section(f"Step {step}: GRPO Update")

    timer.start("grpo_update")
    model.train()

    # Mini-batch training
    total_loss = 0
    n_minibatches = 0

    for mini_batch in combined_batch.make_iterator(
        mini_batch_size=config.rollout.num_rollouts * 2,  # 2 questions per mini-batch
        epochs=1,
        shuffle=True
    ):
        optimizer.zero_grad()

        # Move to device
        mini_batch = mini_batch.to(device)

        # Simplified loss computation (in practice, use full grpo_loss)
        # For demo, we'll use a placeholder
        loss = compute_simple_grpo_loss(model, mini_batch, config.grpo.clip_epsilon)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grpo.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_minibatches += 1

    avg_loss = total_loss / max(n_minibatches, 1)
    timer.stop("grpo_update")

    print(f"  Avg loss: {avg_loss:.4f}")
    metrics["loss"] = avg_loss

    # =========================================================================
    # STEP 6: Update Replay Buffer
    # =========================================================================
    timer.start("buffer_update")

    # Filter for informative samples before adding
    if config.replay.filter_informative:
        informative_batch = filter_informative_rollouts(
            fresh_batch, config.rollout.num_rollouts
        )
        if len(informative_batch) > 0:
            replay_buffer.add(informative_batch)
            print(f"  Added {len(informative_batch) // config.rollout.num_rollouts} informative questions to buffer")
    else:
        replay_buffer.add(fresh_batch)

    timer.stop("buffer_update")

    buffer_stats = replay_buffer.get_stats()
    print(f"  Buffer: {buffer_stats['num_questions']}/{replay_buffer.capacity} questions")
    metrics.update({f"buffer_{k}": v for k, v in buffer_stats.items()})

    # Log metrics
    logger.log(metrics, step=step)

    return metrics


def generate_rollouts_for_question_simple(
    question: str,
    tokenizer,
    model,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
):
    """
    Simplified rollout generation for a single question.

    For demo purposes - in production use the full generate_rollouts_for_question.
    """
    from rollout_generator import apply_chat_template

    encoded = apply_chat_template(question, tokenizer)
    prompt_ids = encoded["input_ids"].to(model.device)

    # Repeat for G rollouts
    batch_input_ids = prompt_ids.repeat(num_rollouts, 1)

    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = prompt_ids.shape[1]
    completions = [
        tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        for i in range(num_rollouts)
    ]

    return completions, None, None, None


def compute_simple_grpo_loss(model, data, clip_epsilon):
    """
    Simplified GRPO loss for demo.

    In production, use the full grpo_loss function.
    """
    response_ids = data.batch["response_ids"]
    advantages = data.batch["advantages"]
    old_log_probs = data.batch["old_log_probs"]

    # Forward pass
    outputs = model(input_ids=response_ids)
    logits = outputs.logits[:, :-1]
    labels = response_ids[:, 1:]

    # Log probs
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Truncate to match
    min_len = min(token_log_probs.shape[1], old_log_probs.shape[1])
    token_log_probs = token_log_probs[:, :min_len]
    old_log_probs = old_log_probs[:, :min_len]

    # Policy loss
    ratio = torch.exp(token_log_probs - old_log_probs)
    advantages = advantages.unsqueeze(1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    loss = -torch.min(unclipped, clipped).mean()

    return loss


def main():
    """Main training function."""
    print_banner("DOTS-RR Training")

    # Parse arguments and build config
    args = parse_args()
    config = get_default_config()

    # Override config with command line arguments
    if args.model_name:
        config.model.model_name = args.model_name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.total_steps:
        config.training.total_steps = args.total_steps
    if args.learning_rate:
        config.grpo.learning_rate = args.learning_rate
    if args.reference_set_size:
        config.dots.reference_set_size = args.reference_set_size
    if args.alpha:
        config.dots.alpha = args.alpha
    if args.tau:
        config.dots.tau = args.tau
    if args.delta:
        config.replay.delta = args.delta
    if args.buffer_capacity:
        config.replay.buffer_capacity = args.buffer_capacity
    if args.num_samples:
        config.data.num_samples = args.num_samples
    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Print configuration
    config.print_config()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Initialize logging and checkpointing
    logger = Logger(config.training.output_dir)
    checkpointer = Checkpointer(config.training.output_dir)
    timer = Timer()

    # =========================================================================
    # Load Data
    # =========================================================================
    print_section("Loading Data")
    questions, solutions, dataset = load_data(config)

    # =========================================================================
    # Load Model
    # =========================================================================
    print_section("Loading Model")
    model, tokenizer = load_model(config, device)

    # =========================================================================
    # Initialize Components
    # =========================================================================
    print_section("Initializing Components")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.grpo.learning_rate,
    )
    print(f"Optimizer: AdamW, lr={config.grpo.learning_rate}")

    # DOTS Selector
    dots_selector = DOTSSelector(
        reference_set_size=config.dots.reference_set_size,
        alpha=config.dots.alpha,
        tau=config.dots.tau,
        embedding_batch_size=config.dots.embedding_batch_size,
    )
    dots_selector.initialize(questions, tokenizer, model)

    # Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=config.replay.buffer_capacity,
        group_size=config.rollout.num_rollouts,
        filter_informative=config.replay.filter_informative,
        replay_strategy=config.replay.replay_strategy,
    )
    print(f"Replay buffer: capacity={config.replay.buffer_capacity}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print_banner(f"Starting Training: {config.training.total_steps} steps")

    for step in range(1, config.training.total_steps + 1):
        timer.start("total_step")

        metrics = training_step(
            step=step,
            config=config,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            questions=questions,
            solutions=solutions,
            dots_selector=dots_selector,
            replay_buffer=replay_buffer,
            timer=timer,
            logger=logger,
        )

        step_time = timer.stop("total_step")
        print(f"\n  Step {step} completed in {step_time:.2f}s")

        # Checkpoint
        if step % config.training.save_every == 0:
            checkpointer.save(model, tokenizer, step, metrics)

    # =========================================================================
    # Final Save
    # =========================================================================
    print_banner("Training Complete")

    # Save final model
    final_dir = os.path.join(config.training.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    # Save replay buffer
    replay_buffer.save(os.path.join(config.training.output_dir, "replay_buffer.pkl"))

    # Save logs
    logger.save()

    # Print timing report
    timer.report()

    print(f"\nCompleted {config.training.total_steps} DOTS-RR training steps.")


if __name__ == "__main__":
    main()
