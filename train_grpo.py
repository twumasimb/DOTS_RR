import re
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig


# DOTS

## 1. Compute Embeddings for all questions prompt[contents] --> Before updating the prompts.
## 1.a Update the prompts with chat template to get the results needed. 
## 2. Sample from the indices of the embeddings 128 reference questions at random
## 3. Rollouts on the sampled dataset. 





# 1. Define a robust reward function
def math_reward_func(completions, solution, **kwargs):
    rewards = []
    for content, sol in zip(completions, solution):
        # Look for the last number or "Yes/No" in the completion
        # Or, even better, instruct your model to use <answer> tags
        match = re.search(r"\"####\"?\s*(.*)", content[0]['content'])
        if match:
            predicted = match.group(1).strip()
            rewards.append(1.0 if predicted.lower() == sol.lower() else 0.0)
        else:
            rewards.append(0.0) # Penalty for not following format
    return rewards

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")


def format_prompt(example):
    # The prompt is a list of messages. We modify the content of the user message.
    # We append the formatting instruction to the end of the existing question.
    instruction = "\nLet's think step by step and output the final answer after \"####\"."
    
    # example['prompt'] is a list: [{'role': 'user', 'content': '...'}]
    example['prompt'][0]['content'] += instruction
    return example

# Apply the transformation
dataset = dataset.map(format_prompt)


# GRPO STEP

training_args = GRPOConfig(
    use_vllm=False,
    report_to="none",
    learning_rate=1e-5, # Added a standard LR
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    output_dir="./output",
    # GRPO specific: how many completions to generate per prompt
    num_generations=8 
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[math_reward_func], # Use the custom function
    train_dataset=dataset,
    args=training_args
)

trainer.train()