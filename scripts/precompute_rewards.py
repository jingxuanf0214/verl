from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.reward_score.precompute_rewards import RewardMemMapDataset
from verl.utils import hf_tokenizer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.reward_manager.prime import PrimeRewardManager

def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute rewards for a dataset')
    parser.add_argument('--parquet_files', type=str, nargs='+', required=True,
                      help='Path(s) to parquet file(s) containing the dataset')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use for tokenization')
    parser.add_argument('--actor_model_path', type=str, required=True,
                      help='Path to the actor model for generating responses')
    parser.add_argument('--reward_path', type=str, required=True,
                      help='Path to save the memory-mapped rewards file')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--max_prompt_length', type=int, default=1024,
                      help='Maximum length of prompts')
    parser.add_argument('--prompt_key', type=str, default='prompt',
                      help='Key in the parquet file containing prompts')
    parser.add_argument('--data_source', type=str, default='openai/gsm8k',
                      help='Data source identifier')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the training config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = hf_tokenizer(args.model_name)
    
    # Create dataset
    dataset = RLHFDataset(
        parquet_files=args.parquet_files,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        max_prompt_length=args.max_prompt_length,
        filter_prompts=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    # Initialize reward dataset
    reward_dataset = RewardMemMapDataset(
        reward_path=args.reward_path,
        dtype=torch.float32,
        create_if_not_exists=True
    )
    
    # Initialize trainer for rollout (without gradient updates)
    trainer = RayPPOTrainer(
        config_path=args.config_path,
        actor_model_path=args.actor_model_path,
        use_rm=False,  # We'll use rule-based rewards only
        num_examine=0  # Don't print examples
    )
    
    # Process batches
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        # Convert batch to DataProto format
        tensors = {}
        non_tensors = {}
        
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            else:
                non_tensors[key] = val
        
        # Add data source and reward model info
        non_tensors['data_source'] = args.data_source
        non_tensors['reward_model'] = {
            'style': 'rule',
            'ground_truth': [item.get('ground_truth', '') for item in batch]
        }
        
        # Create DataProto
        data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        
        # Generate responses using actor (without gradient updates)
        with torch.no_grad():
            responses = trainer.actor_wg.generate(data)
            data.batch['responses'] = responses
            
            # Compute rewards using the same mechanism as training
            reward_tensor = trainer.reward_fn(data)
            data.batch['token_level_scores'] = reward_tensor
        
        # Store rewards
        for i in range(len(data)):
            # Get the unique identifier
            identifier = data.non_tensor_batch.get('uid', f'example_{i}')
            
            # Get the reward at the last valid token
            valid_response_length = data.batch['attention_mask'][i, -data.batch['responses'].shape[-1]:].sum()
            reward = data.batch['token_level_scores'][i, valid_response_length - 1].item()
            
            # Store the reward
            reward_dataset.add_reward(identifier, reward)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Processed {batch_idx + 1}/{total_batches} batches')
    
    # Close reward dataset
    reward_dataset.close()
    print(f'Finished pre-computing rewards. Saved to {args.reward_path}')

if __name__ == '__main__':
    main() 