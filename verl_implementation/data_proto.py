"""
DataProto: Core Data Structure for verl-style Training

This module implements a simplified version of verl's DataProto class,
which is the primary data structure for passing batches between components
(rollout generation, reward computation, policy update, replay buffer).

Key Features:
- Stores both tensor data (input_ids, log_probs) and non-tensor data (text, metadata)
- Supports concatenation, indexing, slicing
- Supports save/load for checkpointing
- Supports iteration for mini-batch training

verl's Original DataProto:
    https://github.com/verl-project/verl/blob/main/verl/protocol.py

This is a simplified standalone version that doesn't require verl installation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Iterator, Any
import torch
import numpy as np
import pickle
import os


@dataclass
class DataProto:
    """
    A unified data container for RL training batches.

    This class holds both PyTorch tensors and non-tensor data (strings, numpy arrays)
    in a structured way that supports common batch operations.

    Attributes:
        batch: Dictionary of PyTorch tensors (e.g., input_ids, attention_mask, log_probs)
        non_tensor_batch: Dictionary of non-tensor data (e.g., question text, solutions)
        meta_info: Dictionary of metadata (e.g., batch size, step number)

    Example:
        >>> data = DataProto(
        ...     batch={"input_ids": torch.tensor([[1, 2, 3]]), "log_probs": torch.tensor([[-0.5]])},
        ...     non_tensor_batch={"question": ["What is 2+2?"], "solution": ["4"]},
        ...     meta_info={"step": 1}
        ... )
        >>> print(len(data))  # 1
        >>> subset = data[0]  # Get first item
    """

    batch: Dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_batch: Dict[str, Union[List, np.ndarray]] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """
        Return the batch size (number of samples).

        Inferred from the first tensor in batch, or first list in non_tensor_batch.
        """
        if self.batch:
            first_tensor = next(iter(self.batch.values()))
            return first_tensor.shape[0]
        elif self.non_tensor_batch:
            first_list = next(iter(self.non_tensor_batch.values()))
            return len(first_list)
        return 0

    def __getitem__(self, indices: Union[int, List[int], slice, torch.Tensor, np.ndarray]) -> "DataProto":
        """
        Index or slice the DataProto.

        Args:
            indices: Can be:
                - int: Single index
                - List[int]: List of indices
                - slice: Slice object
                - torch.Tensor: Boolean mask or index tensor
                - np.ndarray: Boolean mask or index array

        Returns:
            New DataProto containing only the selected samples.

        Example:
            >>> data = DataProto(batch={"x": torch.arange(10).unsqueeze(1)})
            >>> subset = data[0]           # Single item
            >>> subset = data[[0, 2, 4]]   # Multiple items
            >>> subset = data[:5]          # First 5 items
            >>> subset = data[torch.tensor([True, False, True, ...])]  # Boolean mask
        """
        # Convert various index types to a list of indices
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(len(self))))
        elif isinstance(indices, torch.Tensor):
            if indices.dtype == torch.bool:
                indices = indices.nonzero(as_tuple=True)[0].tolist()
            else:
                indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            if indices.dtype == bool:
                indices = np.where(indices)[0].tolist()
            else:
                indices = indices.tolist()

        # Slice tensors
        new_batch = {}
        for key, tensor in self.batch.items():
            new_batch[key] = tensor[indices]

        # Slice non-tensor data
        new_non_tensor = {}
        for key, data in self.non_tensor_batch.items():
            if isinstance(data, np.ndarray):
                new_non_tensor[key] = data[indices]
            else:  # List
                new_non_tensor[key] = [data[i] for i in indices]

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor,
            meta_info=self.meta_info.copy()
        )

    @staticmethod
    def concat(data_list: List["DataProto"]) -> "DataProto":
        """
        Concatenate multiple DataProto objects along the batch dimension.

        This is the primary method for combining fresh rollouts with replay samples.

        Args:
            data_list: List of DataProto objects to concatenate.

        Returns:
            New DataProto containing all samples from all inputs.

        Example:
            >>> fresh = DataProto(batch={"x": torch.tensor([[1], [2]])})
            >>> replay = DataProto(batch={"x": torch.tensor([[3], [4]])})
            >>> combined = DataProto.concat([fresh, replay])
            >>> print(len(combined))  # 4
        """
        if not data_list:
            return DataProto()

        if len(data_list) == 1:
            return data_list[0]

        # Get keys from first DataProto
        first = data_list[0]
        tensor_keys = set(first.batch.keys())
        non_tensor_keys = set(first.non_tensor_batch.keys())

        # Concatenate tensors
        new_batch = {}
        for key in tensor_keys:
            tensors = [d.batch[key] for d in data_list if key in d.batch]
            if tensors:
                new_batch[key] = torch.cat(tensors, dim=0)

        # Concatenate non-tensor data
        new_non_tensor = {}
        for key in non_tensor_keys:
            combined = []
            for d in data_list:
                if key in d.non_tensor_batch:
                    data = d.non_tensor_batch[key]
                    if isinstance(data, np.ndarray):
                        combined.append(data)
                    else:
                        combined.extend(data)

            if combined:
                if isinstance(data_list[0].non_tensor_batch.get(key), np.ndarray):
                    new_non_tensor[key] = np.concatenate(combined, axis=0)
                else:
                    new_non_tensor[key] = combined

        # Merge meta_info (use first's meta_info as base)
        new_meta = first.meta_info.copy()

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor,
            meta_info=new_meta
        )

    def to(self, device: Union[str, torch.device]) -> "DataProto":
        """
        Move all tensors to the specified device.

        Args:
            device: Target device ("cuda", "cpu", or torch.device)

        Returns:
            New DataProto with tensors on the target device.
        """
        new_batch = {key: tensor.to(device) for key, tensor in self.batch.items()}
        return DataProto(
            batch=new_batch,
            non_tensor_batch=self.non_tensor_batch.copy(),
            meta_info=self.meta_info.copy()
        )

    def save_to_disk(self, path: str) -> None:
        """
        Save DataProto to disk using pickle.

        Useful for checkpointing the replay buffer.

        Args:
            path: File path to save to (e.g., "replay_buffer.pkl")
        """
        # Convert tensors to CPU for saving
        cpu_batch = {key: tensor.cpu() for key, tensor in self.batch.items()}

        save_dict = {
            "batch": cpu_batch,
            "non_tensor_batch": self.non_tensor_batch,
            "meta_info": self.meta_info
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load_from_disk(path: str, device: str = "cpu") -> "DataProto":
        """
        Load DataProto from disk.

        Args:
            path: File path to load from.
            device: Device to load tensors to.

        Returns:
            Loaded DataProto object.
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        batch = {key: tensor.to(device) for key, tensor in save_dict["batch"].items()}

        return DataProto(
            batch=batch,
            non_tensor_batch=save_dict["non_tensor_batch"],
            meta_info=save_dict["meta_info"]
        )

    def make_iterator(
        self,
        mini_batch_size: int,
        epochs: int = 1,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> Iterator["DataProto"]:
        """
        Create an iterator that yields mini-batches.

        Used for multiple gradient steps per training step (like PPO's mini-batch updates).

        Args:
            mini_batch_size: Size of each mini-batch.
            epochs: Number of epochs (passes through the data).
            shuffle: Whether to shuffle before each epoch.
            drop_last: Whether to drop the last incomplete batch.

        Yields:
            DataProto mini-batches.

        Example:
            >>> data = DataProto(batch={"x": torch.arange(100).unsqueeze(1)})
            >>> for mini_batch in data.make_iterator(mini_batch_size=16, epochs=2):
            ...     # Process mini_batch
            ...     pass
        """
        n = len(self)

        for epoch in range(epochs):
            if shuffle:
                indices = torch.randperm(n).tolist()
            else:
                indices = list(range(n))

            for start in range(0, n, mini_batch_size):
                end = start + mini_batch_size
                if end > n and drop_last:
                    break
                batch_indices = indices[start:min(end, n)]
                yield self[batch_indices]

    def union(self, other: "DataProto") -> "DataProto":
        """
        Merge another DataProto's fields into this one (in-place style, returns new).

        Useful for adding computed fields (e.g., adding log_probs to a batch).

        Args:
            other: DataProto whose fields to add.

        Returns:
            New DataProto with combined fields.
        """
        new_batch = {**self.batch, **other.batch}
        new_non_tensor = {**self.non_tensor_batch, **other.non_tensor_batch}
        new_meta = {**self.meta_info, **other.meta_info}

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor,
            meta_info=new_meta
        )

    def keys(self) -> List[str]:
        """Return all keys (tensor and non-tensor)."""
        return list(self.batch.keys()) + list(self.non_tensor_batch.keys())

    def __repr__(self) -> str:
        """String representation showing structure."""
        tensor_info = {k: tuple(v.shape) for k, v in self.batch.items()}
        non_tensor_info = {k: len(v) for k, v in self.non_tensor_batch.items()}
        return (
            f"DataProto(\n"
            f"  batch={tensor_info},\n"
            f"  non_tensor_batch={non_tensor_info},\n"
            f"  meta_info={self.meta_info}\n"
            f")"
        )


def create_rollout_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    indices: List[int],
    questions: List[str],
    solutions: List[str],
    **kwargs
) -> DataProto:
    """
    Factory function to create a DataProto for rollout data.

    This is the standard format for storing rollouts in the replay buffer.

    Args:
        input_ids: Full sequence token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        response_ids: Response-only token IDs [batch, response_len]
        old_log_probs: Log probabilities at generation time [batch, response_len]
        rewards: Reward values [batch] or [batch, 1]
        advantages: Computed advantages [batch] or [batch, 1]
        indices: Question indices (for deduplication)
        questions: Question text strings
        solutions: Ground truth solutions
        **kwargs: Additional tensor fields

    Returns:
        DataProto containing all rollout data.
    """
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_ids": response_ids,
        "old_log_probs": old_log_probs,
        "rewards": rewards.view(-1) if rewards.dim() > 1 else rewards,
        "advantages": advantages.view(-1) if advantages.dim() > 1 else advantages,
        **kwargs
    }

    non_tensor_batch = {
        "index": indices if isinstance(indices, list) else indices.tolist(),
        "question": questions,
        "solution": solutions,
    }

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# Example usage and testing
if __name__ == "__main__":
    print("Testing DataProto...")

    # Create sample data
    batch_size = 4
    seq_len = 10

    data = DataProto(
        batch={
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "log_probs": torch.randn(batch_size, seq_len),
            "rewards": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        },
        non_tensor_batch={
            "index": [0, 1, 2, 3],
            "question": ["Q1", "Q2", "Q3", "Q4"],
            "solution": ["A1", "A2", "A3", "A4"],
        },
        meta_info={"step": 1}
    )

    print(f"Original: {data}")
    print(f"Length: {len(data)}")

    # Test indexing
    subset = data[[0, 2]]
    print(f"\nSubset [0, 2]: {subset}")
    print(f"Questions: {subset.non_tensor_batch['question']}")

    # Test concatenation
    data2 = DataProto(
        batch={
            "input_ids": torch.randint(0, 1000, (2, seq_len)),
            "log_probs": torch.randn(2, seq_len),
            "rewards": torch.tensor([0.5, 0.5]),
        },
        non_tensor_batch={
            "index": [4, 5],
            "question": ["Q5", "Q6"],
            "solution": ["A5", "A6"],
        }
    )

    combined = DataProto.concat([data, data2])
    print(f"\nCombined: {combined}")

    # Test save/load
    combined.save_to_disk("/tmp/test_data_proto.pkl")
    loaded = DataProto.load_from_disk("/tmp/test_data_proto.pkl")
    print(f"\nLoaded from disk: {loaded}")

    # Test iterator
    print("\nIterating with mini_batch_size=2:")
    for i, mini_batch in enumerate(combined.make_iterator(mini_batch_size=2, epochs=1)):
        print(f"  Batch {i}: {len(mini_batch)} samples, questions={mini_batch.non_tensor_batch['question']}")

    print("\nAll tests passed!")
