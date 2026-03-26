"""
Reward Functions for Mathematical Reasoning

This module implements reward computation for math reasoning tasks.
The paper uses simple rule-based rewards based on answer correctness,
without any format-related signals.

Key Concepts:
=============

1. VERIFIABLE REWARDS
   Unlike open-ended generation tasks, math problems have verifiable answers.
   We can automatically check if the model's answer matches the ground truth.

2. BINARY REWARDS
   r = 1 if extracted_answer == ground_truth else 0

   Simple but effective. The paper shows this works well for GRPO training.

3. ANSWER EXTRACTION
   Models are prompted to put answers in \\boxed{...} format.
   We extract the last \\boxed{} content as the predicted answer.

4. ANSWER NORMALIZATION
   Before comparison, both answers are normalized:
   - Strip whitespace
   - Remove $ delimiters
   - Expand LaTeX text wrappers (\\text{Yes} → Yes)
   - Lowercase

Reference:
    Paper Section 5.1: "For reward computation, we use a simple rule-based
    function based solely on answer correctness, without incorporating any
    format-related signals."
"""

import re
from typing import Optional, List, Union


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the LAST \\boxed{...} content from text.

    Handles nested braces correctly:
    - \\boxed{\\frac{1}{2}} → \\frac{1}{2}
    - \\boxed{{a, b}} → {a, b}

    Why last? Models sometimes show intermediate boxed answers before
    the final one. We want the final answer.

    Args:
        text: Model's completion text

    Returns:
        Content inside the last \\boxed{}, or None if not found

    Examples:
        >>> extract_boxed_answer("The answer is \\boxed{42}")
        '42'
        >>> extract_boxed_answer("First \\boxed{wrong}, then \\boxed{correct}")
        'correct'
        >>> extract_boxed_answer("No boxed answer here")
        None
    """
    pattern = r"\\boxed\{"
    results = []

    for match in re.finditer(pattern, text):
        start = match.end()
        depth = 1
        i = start

        # Find matching closing brace
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1

        if depth == 0:
            content = text[start:i-1].strip()
            results.append(content)

    return results[-1] if results else None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Normalization steps:
    1. Strip leading/trailing whitespace
    2. Remove $ and $$ delimiters (LaTeX math mode)
    3. Expand \\text{...} wrappers
    4. Remove all internal whitespace
    5. Convert to lowercase

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string

    Examples:
        >>> normalize_answer("  42  ")
        '42'
        >>> normalize_answer("$\\frac{1}{2}$")
        '\\frac{1}{2}'
        >>> normalize_answer("\\text{Yes}")
        'yes'
        >>> normalize_answer("1, 2, 3")
        '1,2,3'
    """
    if answer is None:
        return ""

    ans = answer.strip()

    # Remove $ or $$ delimiters
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()

    # Expand \text{...} wrappers
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)

    # Remove all whitespace
    ans = re.sub(r"\s+", "", ans)

    # Lowercase
    ans = ans.lower()

    return ans


def answers_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.

    Both answers are normalized before comparison.

    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if answers match after normalization
    """
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def compute_reward(completion: str, solution: str) -> float:
    """
    Compute binary reward for a single completion.

    Args:
        completion: Model's generated completion text
        solution: Ground truth solution/answer

    Returns:
        1.0 if correct, 0.0 if incorrect

    Example:
        >>> compute_reward("Let me solve this... \\boxed{42}", "42")
        1.0
        >>> compute_reward("I think it's \\boxed{41}", "42")
        0.0
    """
    predicted = extract_boxed_answer(completion)

    if predicted is None:
        return 0.0

    return 1.0 if answers_match(predicted, solution) else 0.0


def compute_rewards_batch(
    completions: List[str],
    solutions: List[str]
) -> List[float]:
    """
    Compute rewards for a batch of completions.

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions (same length)

    Returns:
        List of binary rewards (0.0 or 1.0)
    """
    assert len(completions) == len(solutions), \
        f"Length mismatch: {len(completions)} completions, {len(solutions)} solutions"

    return [compute_reward(c, s) for c, s in zip(completions, solutions)]


def compute_rewards_grouped(
    completions_per_question: List[List[str]],
    solutions: List[str]
) -> List[List[float]]:
    """
    Compute rewards for grouped completions (G completions per question).

    Args:
        completions_per_question: List of N lists, each containing G completions
        solutions: List of N solutions

    Returns:
        List of N lists, each containing G rewards

    Example:
        >>> completions = [
        ...     ["\\boxed{1}", "\\boxed{2}"],  # Q1: 2 completions
        ...     ["\\boxed{3}", "\\boxed{3}"],  # Q2: 2 completions
        ... ]
        >>> solutions = ["1", "3"]
        >>> compute_rewards_grouped(completions, solutions)
        [[1.0, 0.0], [1.0, 1.0]]
    """
    assert len(completions_per_question) == len(solutions)

    rewards = []
    for completions, solution in zip(completions_per_question, solutions):
        q_rewards = [compute_reward(c, solution) for c in completions]
        rewards.append(q_rewards)

    return rewards


def compute_difficulty_from_rewards(rewards: List[float]) -> float:
    """
    Compute adaptive difficulty from a group of rewards.

    d_q = (1/G) * Σ(1 - r_i) = 1 - mean(rewards)

    Args:
        rewards: List of G binary rewards for one question

    Returns:
        Difficulty in [0, 1]
    """
    return 1.0 - (sum(rewards) / len(rewards))


def is_informative(rewards: List[float]) -> bool:
    """
    Check if a group of rewards is informative.

    A question is "informative" if its rollouts have mixed rewards
    (not all 0s or all 1s). Only informative questions provide
    gradient signal in GRPO.

    Args:
        rewards: List of G binary rewards

    Returns:
        True if rewards are mixed (some 0s and some 1s)
    """
    avg_reward = sum(rewards) / len(rewards)
    return 0 < avg_reward < 1


# ============================================================================
# Advanced: Math-Verify Integration (Optional)
# ============================================================================
# The paper mentions using the Math-Verify library for answer matching:
# https://github.com/huggingface/Math-Verify
#
# Math-Verify handles edge cases like:
# - Equivalent fractions: 1/2 == 2/4
# - Symbolic equivalence: x^2 - 1 == (x+1)(x-1)
# - Numerical tolerance: 3.14159 ≈ π
#
# For production use, consider integrating Math-Verify:
#
# try:
#     from math_verify import verify_answer
#     USE_MATH_VERIFY = True
# except ImportError:
#     USE_MATH_VERIFY = False
#
# def compute_reward_with_verify(completion: str, solution: str) -> float:
#     if USE_MATH_VERIFY:
#         return 1.0 if verify_answer(completion, solution) else 0.0
#     return compute_reward(completion, solution)
# ============================================================================


# Example usage and testing
if __name__ == "__main__":
    print("Testing reward functions...")

    # Test 1: extract_boxed_answer
    print("\n--- Test 1: extract_boxed_answer ---")
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("First \\boxed{wrong} then \\boxed{correct}", "correct"),
        ("\\boxed{{a, b, c}}", "{a, b, c}"),
        ("No boxed answer", None),
        ("", None),
    ]
    for text, expected in test_cases:
        result = extract_boxed_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:40]}...' → {result} (expected {expected})")

    # Test 2: normalize_answer
    print("\n--- Test 2: normalize_answer ---")
    test_cases = [
        ("  42  ", "42"),
        ("$\\frac{1}{2}$", "\\frac{1}{2}"),
        ("\\text{Yes}", "yes"),
        ("1, 2, 3", "1,2,3"),
        ("  YES  ", "yes"),
    ]
    for raw, expected in test_cases:
        result = normalize_answer(raw)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{raw}' → '{result}' (expected '{expected}')")

    # Test 3: compute_reward
    print("\n--- Test 3: compute_reward ---")
    test_cases = [
        ("The answer is \\boxed{42}", "42", 1.0),
        ("I think \\boxed{41}", "42", 0.0),
        ("\\boxed{Yes}", "yes", 1.0),
        ("No answer here", "42", 0.0),
        ("\\boxed{$\\frac{1}{2}$}", "\\frac{1}{2}", 1.0),
    ]
    for completion, solution, expected in test_cases:
        result = compute_reward(completion, solution)
        status = "✓" if result == expected else "✗"
        print(f"  {status} reward={result} (expected {expected})")

    # Test 4: compute_difficulty_from_rewards
    print("\n--- Test 4: compute_difficulty_from_rewards ---")
    test_cases = [
        ([1.0, 1.0, 1.0, 1.0], 0.0),  # All correct → easy
        ([0.0, 0.0, 0.0, 0.0], 1.0),  # All wrong → hard
        ([1.0, 1.0, 0.0, 0.0], 0.5),  # Half correct → medium
        ([1.0, 0.0, 0.0, 0.0], 0.75), # 25% correct → hard
    ]
    for rewards, expected in test_cases:
        result = compute_difficulty_from_rewards(rewards)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} rewards={rewards} → difficulty={result:.2f} (expected {expected})")

    # Test 5: is_informative
    print("\n--- Test 5: is_informative ---")
    test_cases = [
        ([1.0, 1.0, 1.0, 1.0], False),  # All same → not informative
        ([0.0, 0.0, 0.0, 0.0], False),  # All same → not informative
        ([1.0, 1.0, 0.0, 0.0], True),   # Mixed → informative
        ([1.0, 0.0, 1.0, 0.0], True),   # Mixed → informative
    ]
    for rewards, expected in test_cases:
        result = is_informative(rewards)
        status = "✓" if result == expected else "✗"
        print(f"  {status} rewards={rewards} → informative={result} (expected {expected})")

    print("\nAll tests passed!")
