"""
Utility Functions for DOTS-RR Training

This module contains shared utilities used across the training pipeline.
"""

import os
import random
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Get the appropriate device for training.

    Returns:
        Device string ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """
    Format a large number with K/M/B suffixes.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., "1.5B", "125M", "10K")
    """
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


class Logger:
    """
    Simple logger for training metrics.

    Logs to both console and a JSON file for later analysis.

    Example:
        >>> logger = Logger("output/logs")
        >>> logger.log({"step": 1, "loss": 0.5, "accuracy": 0.8})
        >>> logger.save()
    """

    def __init__(self, output_dir: str, name: str = "training"):
        """
        Initialize logger.

        Args:
            output_dir: Directory to save log files
            name: Name prefix for log file
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"{name}_{timestamp}.json")
        self.logs: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (added to metrics if provided)
        """
        entry = metrics.copy()
        if step is not None:
            entry["step"] = step
        entry["timestamp"] = datetime.now().isoformat()

        self.logs.append(entry)

        # Print to console
        log_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in metrics.items())
        if step is not None:
            log_str = f"[Step {step}] {log_str}"
        print(log_str)

    def save(self) -> None:
        """Save logs to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to {self.log_file}")


class Checkpointer:
    """
    Handles model checkpointing.

    Example:
        >>> ckpt = Checkpointer("output/checkpoints")
        >>> ckpt.save(model, tokenizer, step=10, metrics={"loss": 0.5})
        >>> model, tokenizer, metadata = ckpt.load("output/checkpoints/step_10")
    """

    def __init__(self, output_dir: str):
        """
        Initialize checkpointer.

        Args:
            output_dir: Directory to save checkpoints
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        model,
        tokenizer,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            step: Training step number
            metrics: Optional metrics to save
            optimizer: Optional optimizer state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = os.path.join(self.output_dir, f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        if metrics:
            metadata["metrics"] = metrics

        with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save optimizer state
        if optimizer is not None:
            torch.save(optimizer.state_dict(),
                      os.path.join(checkpoint_dir, "optimizer.pt"))

        print(f"Checkpoint saved to {checkpoint_dir}")
        return checkpoint_dir

    def load(self, checkpoint_dir: str):
        """
        Load a checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory

        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return model, tokenizer, metadata


def print_banner(text: str, width: int = 80) -> None:
    """
    Print a banner with the given text.

    Args:
        text: Text to display in banner
        width: Width of banner
    """
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def print_section(text: str, width: int = 80) -> None:
    """
    Print a section header.

    Args:
        text: Section title
        width: Width of line
    """
    print("\n" + "-" * width)
    print(text)
    print("-" * width)


def compute_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, std, min, max
    """
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class Timer:
    """
    Simple timer for measuring execution time.

    Example:
        >>> timer = Timer()
        >>> timer.start("rollout")
        >>> # ... do rollout ...
        >>> elapsed = timer.stop("rollout")
        >>> print(f"Rollout took {elapsed:.2f}s")
    """

    def __init__(self):
        self.starts: Dict[str, float] = {}
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def start(self, name: str) -> None:
        """Start timing an operation."""
        import time
        self.starts[name] = time.time()

    def stop(self, name: str) -> float:
        """
        Stop timing and return elapsed time.

        Args:
            name: Name of operation

        Returns:
            Elapsed time in seconds
        """
        import time
        elapsed = time.time() - self.starts.get(name, time.time())

        if name not in self.totals:
            self.totals[name] = 0
            self.counts[name] = 0

        self.totals[name] += elapsed
        self.counts[name] += 1

        return elapsed

    def get_average(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self.counts or self.counts[name] == 0:
            return 0
        return self.totals[name] / self.counts[name]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        stats = {}
        for name in self.totals:
            stats[name] = {
                "total": self.totals[name],
                "count": self.counts[name],
                "average": self.get_average(name),
            }
        return stats

    def report(self) -> None:
        """Print timing report."""
        print("\n--- Timing Report ---")
        for name, stats in self.get_stats().items():
            print(f"  {name}: {stats['total']:.2f}s total, "
                  f"{stats['average']:.2f}s avg ({stats['count']} calls)")


def move_to_device(data: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in a dictionary to a device.

    Args:
        data: Dictionary of tensors
        device: Target device

    Returns:
        Dictionary with tensors on target device
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}


# Example usage
if __name__ == "__main__":
    print("Testing utilities...")

    # Test Logger
    print("\n--- Test Logger ---")
    logger = Logger("/tmp/test_logs")
    logger.log({"loss": 0.5, "accuracy": 0.8}, step=1)
    logger.log({"loss": 0.3, "accuracy": 0.9}, step=2)

    # Test Timer
    print("\n--- Test Timer ---")
    import time
    timer = Timer()

    timer.start("test_op")
    time.sleep(0.1)
    elapsed = timer.stop("test_op")
    print(f"Elapsed: {elapsed:.3f}s")

    timer.start("test_op")
    time.sleep(0.05)
    timer.stop("test_op")

    timer.report()

    # Test format_number
    print("\n--- Test format_number ---")
    for n in [100, 1500, 1500000, 1500000000]:
        print(f"  {n} -> {format_number(n)}")

    print("\nAll tests passed!")
