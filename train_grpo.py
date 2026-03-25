import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

from utils import (
    format_prompt,
    compute_embeddings,
    sample_reference_set,
    extract_boxed_answer,
    normalize_answer,
    compute_reward,
    generate_rollouts,
    compute_difficulty,
    predict_adaptive_difficulty,
    sample_rollout_batch,
    boxed_reward_func,
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 0: Hyperparameters")
print("="*80)

G     = 8      # Rollouts per question (paper: 8) — how many times we sample the model per question
K     = 16     # Reference set size (paper: 128–256) — small here so the demo runs fast
ALPHA = 0.5    # Target difficulty — we want questions the model answers correctly ~50% of the time
TAU   = 1e-3   # Sampling temperature for difficulty-targeted selection (low = sharp / near-greedy)
DELTA = 0.5    # Fraction of batch that gets FRESH rollouts (the other 0.5 comes from the replay buffer)
B     = 8      # Training batch size (paper: 512)
T     = 60     # Total training steps (paper: 60)
MU    = 2      # Re-estimate difficulties every μ steps (paper: 2)
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

print(f"Config: G={G}, K={K}, α={ALPHA}, τ={TAU}, δ={DELTA}, B={B}, T={T}, μ={MU}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1: Load Dataset")
print("="*80)

raw_dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# Work with a manageable subset for this demo
N = 512
dataset = raw_dataset.shuffle(seed=42).select(range(N))

print(f"Dataset size (demo subset): {len(dataset)}")
print("\nFirst example:")
print("  Prompt  :", dataset[0]['prompt'][0]['content'][:120], "...")
print("  Solution:", dataset[0]['solution'])

dataset = dataset.map(format_prompt)

# Verify the instruction was appended correctly
print("\nFormatted prompt (first 200 chars):")
print(dataset[0]['prompt'][0]['content'][:200])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2: Load Model")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)

print(f"Model: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Compute Embeddings (once, static for all steps)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3: Compute Embeddings")
print("="*80)

# Extract the formatted question text from every example
questions = [ex['prompt'][0]['content'] for ex in dataset]

model.eval()
print(f"Computing embeddings for {len(questions)} questions...")
all_embeddings = compute_embeddings(questions, tokenizer, model, batch_size=8)
print(f"Done. Embeddings shape: {all_embeddings.shape}")  # (N, hidden_dim)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP: Steps 4-9 repeated for T iterations
# ══════════════════════════════════════════════════════════════════════════════
# Key insight: Each iteration fine-tunes on the model from the previous iteration.
# - Step 1: base model → model_1
# - Step 2: model_1 → model_2
# - Step N: model_{N-1} → model_N
#
# This allows the difficulty estimates to adapt as the policy improves:
# questions that were hard become easier, so we sample new challenging ones.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print(f"STARTING TRAINING LOOP: {T} steps")
print("="*80)

n_fresh = int(DELTA * B)  # number of questions that get fresh rollouts

# Initialize variables that persist across steps
ref_difficulties = None
ref_embeddings = None
predicted_difficulties = None

for step in range(T):
    print("\n" + "─"*80)
    print(f"STEP {step+1}/{T}")
    print("─"*80)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Sample Reference Set (every μ steps)
    # ──────────────────────────────────────────────────────────────────────────
    if step % MU == 0:
        print(f"\n[Step {step+1}] Sampling new reference set (every μ={MU} steps)...")

        ref_indices = sample_reference_set(len(dataset), K)
        ref_questions  = [dataset[i.item()]['prompt'][0]['content'] for i in ref_indices]
        ref_solutions  = [dataset[i.item()]['solution']             for i in ref_indices]
        ref_embeddings = all_embeddings[ref_indices]  # (K, hidden_dim)

        print(f"  Reference set size: {K}")
        print(f"  Indices (first 5): {ref_indices[:5].tolist()}")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 5: Run Rollouts and Compute Difficulties
        # ──────────────────────────────────────────────────────────────────────
        print(f"\n[Step {step+1}] Running G={G} rollouts on K={K} reference questions...")

        model.eval()
        ref_difficulties = []

        for idx, (q, sol) in enumerate(zip(ref_questions, ref_solutions)):
            completions = generate_rollouts(q, tokenizer, model, G=G, max_new_tokens=1024)
            difficulty, rewards = compute_difficulty(completions, sol, G)
            ref_difficulties.append(difficulty)

            if idx < 2:  # Print first 2 examples
                print(f"    [Q{idx}] d={difficulty:.3f} | rewards={rewards}")

        ref_difficulties = torch.tensor(ref_difficulties, dtype=torch.float32)
        print(f"  Difficulty stats: mean={ref_difficulties.mean():.3f}, std={ref_difficulties.std():.3f}")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 6: Predict Adaptive Difficulty
        # ──────────────────────────────────────────────────────────────────────
        print(f"\n[Step {step+1}] Predicting difficulty for all {len(dataset)} questions...")

        predicted_difficulties = predict_adaptive_difficulty(
            query_embeddings = all_embeddings,
            ref_embeddings   = ref_embeddings,
            ref_difficulties = ref_difficulties,
        )
        print(f"  Predicted stats: mean={predicted_difficulties.mean():.3f}, std={predicted_difficulties.std():.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7: Sample Rollout Batch
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[Step {step+1}] Sampling {n_fresh} questions for training batch...")

    selected_indices, sampling_probs = sample_rollout_batch(
        predicted_difficulties, alpha=ALPHA, tau=TAU, n_samples=n_fresh
    )

    # Calculate enrichment
    uniform_prob = 1.0 / len(dataset)
    avg_selected_prob = sampling_probs[selected_indices].mean().item()
    enrichment = avg_selected_prob / uniform_prob

    print(f"  Selected indices: {selected_indices.tolist()}")
    print(f"  Avg predicted difficulty: {predicted_difficulties[selected_indices].mean():.3f}")
    print(f"  Enrichment over uniform: {enrichment:.1f}×")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 8: Build DOTS-Selected Training Split
    # ──────────────────────────────────────────────────────────────────────────
    selected_data = dataset.select([i.item() for i in selected_indices])
    print(f"\n[Step {step+1}] Training on {len(selected_data)} difficulty-selected questions")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 9: Run GRPO Training Step
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[Step {step+1}] Running GRPO training...")

    checkpoint_dir = f"./output_dots/step_{step+1}"

    training_args = GRPOConfig(
        output_dir            = checkpoint_dir,
        per_device_train_batch_size = len(selected_data),
        num_train_epochs      = 1,
        num_generations       = G,
        max_completion_length = 256,
        learning_rate         = 1e-6,
        report_to             = "none",
        use_vllm              = False,
        logging_steps         = 1,
    )

    # Pass the current model to continue training from previous iteration's weights
    trainer = GRPOTrainer(
        model        = model,
        reward_funcs = [boxed_reward_func],
        train_dataset= selected_data,
        args         = training_args,
    )

    trainer.train()

    # Save checkpoint for this step
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"  Checkpoint saved to {checkpoint_dir}")

    # Update model reference to the trained model for next iteration
    # This ensures rollouts in step N+1 use the model trained in step N
    model = trainer.model

    print(f"[Step {step+1}] Training step complete.")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Completed {T} DOTS training steps.")

# Save final model
final_dir = "./output_dots/final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Final model saved to {final_dir}")
