import re
import torch
import torch.nn.functional as F


INSTRUCTION = r"\nLet's think step by step and output the final answer within \boxed{}. For example: \boxed{42} or \boxed{(0,\infty)} or \boxed{\dfrac{1}{4}} or \boxed{Yes}"

def format_prompt(example):
    """
    Appends the CoT instruction to the user message in each example.
    `example['prompt']` is a list: [{'role': 'user', 'content': '...'}]
    We modify the content field in-place and return the full example.
    """
    example['prompt'][0]['content'] += INSTRUCTION
    return example


@torch.no_grad()
def compute_embeddings(texts, tokenizer, model, batch_size=16):
    """
    Encodes a list of strings into fixed-size, L2-normalized embedding vectors.

    Args:
        texts      : list of N strings (the raw question text)
        tokenizer  : HuggingFace tokenizer
        model      : causal LM — we use its hidden states, not its output logits
        batch_size : questions processed at once (tune down if you hit OOM)

    Returns:
        embeddings : (N, hidden_dim) float32 tensor, L2-normalized
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenize — pad short sequences, truncate long ones
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        # Forward pass — request ALL hidden layer outputs
        outputs = model(**encoded, output_hidden_states=True)

        # Last hidden layer: shape (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1]

        # Mean pool: ignore padding tokens by masking them out before averaging
        mask       = encoded["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
        sum_hidden = (last_hidden.float() * mask).sum(dim=1)           # (batch, hidden_dim)
        count      = mask.sum(dim=1).clamp(min=1e-9)                   # (batch, 1)
        embeddings = sum_hidden / count                                # (batch, hidden_dim)

        # L2-normalize: makes cosine similarity = dot product (used in Step 4)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)


def sample_reference_set(n_total, K):
    """
    Randomly sample K indices (without replacement) from the full dataset.

    In the full training loop this is called at every DOTS step so the
    reference set rotates and adapts to the evolving policy.

    Returns:
        ref_indices : 1-D LongTensor of K indices into the dataset
    """
    return torch.randperm(n_total)[:K]


def extract_boxed_answer(text):
    """
    Extract the LAST \\boxed{...} handling nested braces.
    """
    pattern = r"\\boxed\{"
    results = []

    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1].strip())

    return results[-1] if results else None


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = ans.strip()
    # Strip surrounding $ or $$ delimiters
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    # Strip LaTeX text wrappers: \text{Yes} -> Yes
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    # Remove all whitespace
    ans = re.sub(r"\s+", "", ans)
    # Lowercase
    ans = ans.lower()
    return ans


def answers_match(extracted: str, solution: str) -> bool:
    return normalize_answer(extracted) == normalize_answer(solution)


def compute_reward(completion, solution):
    """
    Binary reward: 1.0 if the extracted answer matches the solution, else 0.0.
    Uses extract_boxed_answer, normalize_answer, and answers_match.
    """
    predicted = extract_boxed_answer(completion)
    if predicted is None:
        return 0.0
    return 1.0 if answers_match(predicted, solution) else 0.0


@torch.no_grad()
def generate_rollouts(prompt, tokenizer, model, G=8, max_new_tokens=256):
    """
    Generate G independent completions for a single question.

    We batch all G generations together (repeat the input G times) for efficiency.
    do_sample=True ensures each rollout is different — if we used greedy decoding,
    all G outputs would be identical and difficulty would always be 0 or 1.

    Args:
        prompt         : formatted question string (already has the CoT instruction)
        G              : number of rollouts
        max_new_tokens : max tokens per completion (keep small for demo speed)

    Returns:
        completions : list of G decoded strings (new tokens only, no prompt)
    """
    messages  = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    # Repeat the prompt G times so we generate all rollouts in a single forward pass
    input_ids = input_ids.repeat(G, 1)  # (G, seq_len)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,       # stochastic → diverse rollouts across the G copies
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the NEW tokens (strip the prompt prefix)
    prompt_len  = input_ids.shape[1]
    completions = [
        tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        for i in range(G)
    ]
    return completions


def compute_difficulty(completions, solution, G):
    """
    Compute adaptive difficulty from G rollout completions.

    d_q = (1/G) * Σ(1 - r_i)   [Equation 2 in the paper]

    This is simply the average failure rate — the fraction of rollouts where
    the model got the wrong answer.

    Returns:
        difficulty : float in [0, 1]
        rewards    : list of G binary reward values
    """
    rewards    = [compute_reward(c, solution) for c in completions]
    difficulty = 1.0 - (sum(rewards) / G)
    return difficulty, rewards


def predict_adaptive_difficulty(query_embeddings, ref_embeddings, ref_difficulties):
    """
    Predict adaptive difficulty for every question using attention-weighted averaging.

    Implements Section 4.1 equations:
        a_i     = softmax( z_q^T z_i / sqrt(h) )     — attention weights over K refs
        d_hat_q = sum_i  a_i * d_i                   — predicted difficulty

    Args:
        query_embeddings : (N, h) — embeddings for ALL N questions (including refs)
        ref_embeddings   : (K, h) — embeddings for the K reference questions
        ref_difficulties : (K,)   — ground-truth difficulties for the K references

    Returns:
        predicted : (N,) tensor of difficulty predictions, clipped to [0, 1]
    """
    h = query_embeddings.shape[1]  # embedding dimension

    # Similarity matrix: (N, K)
    # Because embeddings are L2-normalized, z_q^T z_i = cosine similarity
    similarity = (query_embeddings @ ref_embeddings.T) / (h ** 0.5)  # (N, K)

    # Attention weights: softmax over the K references for each query
    attention_weights = torch.softmax(similarity, dim=-1)  # (N, K)

    # Predicted difficulty: weighted sum of reference difficulties
    predicted = attention_weights @ ref_difficulties  # (N,)

    # ── Calibration (simplified Platt scaling) ───────────────────────────────
    # The paper trains a 2-layer MLP on [mean, std] of reference difficulties.
    # Here we just shift the prediction distribution to match the reference mean,
    # which is the dominant effect of the full calibration.
    ref_mean = ref_difficulties.mean()
    predicted = predicted - predicted.mean() + ref_mean

    return predicted.clamp(0.0, 1.0)


def sample_rollout_batch(predicted_difficulties, alpha, tau, n_samples):
    """
    Sample n_samples question indices using the difficulty-targeting distribution.

    P(q) ∝ exp( -|d_hat_q - alpha| / tau )

    Low tau  → near-deterministic: almost always picks questions nearest to alpha
    High tau → approaches uniform sampling (equivalent to standard GRPO)

    Args:
        predicted_difficulties : (N,) tensor of predicted difficulty scores
        alpha                  : target difficulty (0.5)
        tau                    : sampling temperature
        n_samples              : how many questions to select (= δ·B)

    Returns:
        selected_indices : (n_samples,) LongTensor — which questions to roll out
        probs            : (N,) tensor — sampling probability for each question
    """
    scores = -(predicted_difficulties - alpha).abs() / tau  # high score = close to 0.5
    probs  = torch.softmax(scores, dim=0)                   # normalize to a distribution
    # Sample without replacement so we don't repeat questions in the same batch
    selected_indices = torch.multinomial(probs, n_samples, replacement=False)
    return selected_indices, probs


def boxed_reward_func(completions, solution, **kwargs):
    """
    Binary reward: 1.0 if the model's \\boxed{} answer matches the solution, else 0.0.

    GRPOTrainer passes:
      completions : list of strings — the model's generated text for each prompt
      solution    : list of strings — the ground-truth answer for each prompt
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        # completions may be a list of messages (role/content dicts) or plain strings
        text = completion[0]['content'] if isinstance(completion, list) else completion
        rewards.append(compute_reward(text, sol))
    return rewards
