# dataloaders.py
import math
import os
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset

from preprocessing import tokenize, group_texts, padding_collate_fn


def load_lang_datasets(dataset_path: str,
                       language: str,
                       hf_token: Optional[str] = None,
                       max_tokens: Optional[int] = None,
                       validation_split_ratio: float = 0.25) -> DatasetDict:
    """
    Load a language dataset from the BabyLM Hugging Face hub
    
    Parameters:
    dataset_path: str
        Hugging Face dataset path, e.g., "BabyLM-community/babylm-eng"
    language: str
        Universal code for the corresponding language
    hf_token : str, optional
        Hugging Face token for private datasets. Use HF_TOKEN env var, if None.
    max_tokens : int, optional
        Maximum number of tokens to sample (for Tier 1 languages with 100M tokens,
        sample 10M = 10% as per methodology). Use all data, if None.
    validation_split_ratio : float
        Ratio of data to use for validation if no validation split exists. 
        Most BabyLM datasets only have 'train' split.
        
    Returns
    DatasetDict with 'train' and 'validation' splits
    """

    token = hf_token or os.getenv("HF_TOKEN")
    dataset = load_dataset(dataset_path, token=token)

    if max_tokens is not None:
        dataset = sample_to_max_tokens(dataset, max_tokens, validation_split_ratio)

    # If only train split exists:
    if "validation" not in dataset and "train" in dataset:
        train_dataset = dataset["train"]
        # Split train into train/validation
        split_dataset = train_dataset.train_test_split(
            test_size=validation_split_ratio, seed=42)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    elif "validation" not in dataset:
        raise ValueError(
            f"No 'train' or 'validation' split found in {dataset_path}. Dataset is:\n{dataset}"
        )

    return dataset



def sample_to_max_tokens(
    dataset: DatasetDict,
    max_tokens: int,
    validation_split_ratio: float = 0.25,
) -> DatasetDict:
    """
    Sample dataset to contain at most max_tokens total (across train + validation).

    Uses the 'num-tokens' field to efficiently sample documents.
    """
    train_ds = dataset["train"] if "train" in dataset else None
    valid_ds = dataset["validation"] if "validation" in dataset else None

    # Calculate total tokens
    train_tokens = sum(train_ds["num-tokens"]) if train_ds else 0
    valid_tokens = sum(valid_ds["num-tokens"]) if valid_ds else 0
    total_tokens = train_tokens + valid_tokens

    if total_tokens <= max_tokens:
        return dataset  # No sampling needed
    if not valid_ds:
        sampled_train = sample_dataset_by_tokens(train_ds, max_tokens)
        sampled_valid = None
        return DatasetDict({
            "train": sampled_train,
        })
    else:
        # Calculate target tokens for train and validation
        target_train_tokens = int(max_tokens * (1 - validation_split_ratio))
        target_valid_tokens = max_tokens - target_train_tokens

        # Sample train split
        sampled_train = sample_dataset_by_tokens(train_ds, target_train_tokens)
        # Sample validation split
        sampled_valid = sample_dataset_by_tokens(valid_ds, target_valid_tokens)
        return DatasetDict({
            "train": sampled_train,
            "validation": sampled_valid
        })


def sample_dataset_by_tokens(dataset: Dataset, target_tokens: int) -> Dataset:
    """
    Sample documents from dataset until we reach approximately target_tokens.
    """
    indices = []
    current_tokens = 0

    # Shuffle indices to get random sample
    all_indices = list(range(len(dataset)))
    random.Random(42).shuffle(all_indices)  # Deterministic shuffle

    for idx in all_indices:
        if current_tokens >= target_tokens:
            break
        indices.append(idx)
        current_tokens += dataset[idx]["num-tokens"]

    return dataset.select(indices)

def tokenize_and_group(
    dataset: DatasetDict,
    tokenizer,
    max_seq_len: int = 256,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize the raw dataset with the provided tokenizer and group
    it into fixed-length chunks usable for masked language modeling.
    """
    tokenized = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "input_field": "text"},
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    grouped = tokenized.map(
        group_texts,
        batched=True,
        fn_kwargs={"max_len": max_seq_len},
        num_proc=num_proc,
        desc="Grouping into fixed-length sequences",
    )

    return grouped

def create_dataloaders(
    dataset_path: str,
    tokenizer,
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_proc: int = 4,
    num_workers: int = 4,
    hf_token: Optional[str] = None,
    max_tokens: Optional[int] = None,
    validation_split_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    High-level helper that:
      1. loads a language dataset from Hugging Face hub,
      2. optionally samples to max_tokens (for Tier 1 languages),
      3. tokenizes and groups them,
      4. returns PyTorch dataloaders for training and validation.

    Parameters
    ----------
    dataset_path : str
        Hugging Face dataset path, e.g., "BabyLM-community/babylm-eng"
    tokenizer
        Tokenizer to use for tokenization
    max_seq_len : int
        Maximum sequence length for grouping
    batch_size : int
        Batch size for dataloaders
    num_proc : int
        Number of processes for dataset processing
    num_workers : int
        Number of workers for dataloaders
    hf_token : str, optional
        Hugging Face token for private datasets
    max_tokens : int, optional
        Maximum tokens to sample (10M for Tier 1 languages). If None, uses all data.
    validation_split_ratio : float
        Ratio for validation split if not present in dataset

    Returns
    -------
    train_dataloader, eval_dataloader, steps_per_epoch
    """
    dataset = load_language_datasets(
        dataset_path,
        hf_token=hf_token,
        max_tokens=max_tokens,
        validation_split_ratio=validation_split_ratio,
    )
    grouped = tokenize_and_group(dataset, tokenizer, max_seq_len=max_seq_len, num_proc=num_proc)

    train_dataloader = DataLoader(
        grouped["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=padding_collate_fn,
    )

    eval_dataloader = DataLoader(
        grouped["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=padding_collate_fn,
    )

    steps_per_epoch = math.ceil(len(grouped["train"]) / batch_size)

    return train_dataloader, eval_dataloader, steps_per_epoch


