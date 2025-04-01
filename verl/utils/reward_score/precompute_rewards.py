from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
import os

from .finemath import compute_score
from ..data_proto import DataProto

class RewardMemMapDataset:
    """
    A dataset class for storing pre-computed rewards using memory-mapped arrays.
    Each instance contains a unique identifier and its corresponding reward score.
    """
    
    def __init__(
        self,
        reward_path: str,
        dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.float32,
        create_if_not_exists: bool = False
    ):
        """
        Initialize the reward memory-mapped dataset.
        
        Args:
            reward_path: Path to the memory-mapped array file
            dtype: Data type for the memory-mapped array
            create_if_not_exists: Whether to create the file if it doesn't exist
        """
        self.reward_path = Path(reward_path)
        self.dtype = dtype
        self._rewards = None
        self._indices = {}  # Maps unique identifiers to array indices
        
        if create_if_not_exists and not self.reward_path.exists():
            # Create an empty memory-mapped array
            self._rewards = np.memmap(
                self.reward_path,
                dtype=self.dtype,
                mode='w+',
                shape=(0,)
            )
        else:
            # Load existing memory-mapped array
            self._rewards = np.memmap(
                self.reward_path,
                dtype=self.dtype,
                mode='r+'
            )
    
    def add_reward(self, identifier: str, reward: float) -> None:
        """
        Add a new reward score for a given identifier.
        
        Args:
            identifier: Unique identifier for the training example
            reward: The computed reward score
        """
        if identifier in self._indices:
            # Update existing reward
            idx = self._indices[identifier]
            self._rewards[idx] = reward
        else:
            # Add new reward
            idx = len(self._rewards)
            self._rewards = np.memmap(
                self.reward_path,
                dtype=self.dtype,
                mode='r+',
                shape=(idx + 1,)
            )
            self._rewards[idx] = reward
            self._indices[identifier] = idx
    
    def get_reward(self, identifier: str) -> Optional[float]:
        """
        Get the reward score for a given identifier.
        
        Args:
            identifier: Unique identifier for the training example
            
        Returns:
            The reward score if found, None otherwise
        """
        if identifier in self._indices:
            return float(self._rewards[self._indices[identifier]])
        return None
    
    def precompute_rewards(self, data: DataProto) -> None:
        """
        Pre-compute rewards for a batch of data using the finemath scoring system.
        
        Args:
            data: DataProto containing the training examples
        """
        for i in range(len(data)):
            data_item = data[i]
            
            # Get the unique identifier
            identifier = data_item.non_tensor_batch.get('uid', f'example_{i}')
            
            # Get the response and ground truth
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][-response_ids.shape[-1]:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode the response
            response_str = data_item.non_tensor_batch.get('tokenizer', None).decode(
                valid_response_ids, skip_special_tokens=True
            )
            
            # Get ground truth
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
            # Compute reward score
            reward = compute_score(response_str, ground_truth)
            
            # Store the reward
            self.add_reward(identifier, reward)
    
    def __len__(self) -> int:
        """Get the total number of stored rewards."""
        return len(self._rewards)
    
    def close(self) -> None:
        """Close the memory-mapped array."""
        if self._rewards is not None:
            self._rewards._mmap.close()
            self._rewards = None 