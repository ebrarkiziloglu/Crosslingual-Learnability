# dataloaders.py
import logging
import math
import os
import sys
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset

from preprocessing import tokenize, group_texts, padding_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def load_lang_datasets(dataset_path: str,
                       language: str = None,
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
    logger.info(
        "Loading dataset '%s' (language=%s, max_tokens=%s, validation_split_ratio=%.2f)",
        dataset_path,
        language,
        str(max_tokens),
        validation_split_ratio,
    )
    dataset = load_dataset(dataset_path, token=token)

    logger.info(
        "Loaded dataset '%s' with splits: %s",
        dataset_path,
        list(dataset.keys()),
    )

    if max_tokens is not None:
        dataset = sample_to_max_tokens(dataset, max_tokens, validation_split_ratio)

    # If only train split exists:
    if "validation" not in dataset and "train" in dataset:
        train_dataset = dataset["train"]
        logger.info(
            "Dataset '%s' has only 'train' split; splitting into train/validation "
            "with validation ratio %.2f",
            dataset_path,
            validation_split_ratio,
        )
        # Split train into train/validation
        split_dataset = train_dataset.train_test_split(
            test_size=validation_split_ratio, seed=42)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    elif "validation" not in dataset:
        logger.error(
            "No 'train' or 'validation' split found in %s. Dataset is: %s",
            dataset_path,
            dataset,
        )
        raise ValueError(
            f"No 'train' or 'validation' split found in {dataset_path}. Dataset is:\n{dataset}"
        )

    logger.info(
        "Final dataset splits: %s",
        {k: len(v) for k, v in dataset.items()},
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
    logger.info(
        "[Sampling] Starting token-based sampling for dataset to fit max_tokens=%d. "
        "Current total tokens: %d (train=%d, validation=%d)",
        max_tokens,
        total_tokens,
        train_tokens,
        valid_tokens,
    )
    if total_tokens > max_tokens:
        logger.info(
            "[Sampling] Will sample down to %d tokens (%.2f%% of original data).",
            max_tokens,
            100.0 * max_tokens / total_tokens,
        )
    else:
        logger.info(
            "[Sampling] No sampling needed: dataset is already under the limit."
        )
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

        logger.info(
            "[Sampling] Token budget split: train=%d, validation=%d (ratio=%.2f)",
            target_train_tokens,
            target_valid_tokens,
            validation_split_ratio,
        )

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

    logger.info(
        "[Sampling] Sampling dataset of %d documents to approximately %d tokens.",
        len(dataset),
        target_tokens,
    )

    # Shuffle indices to get random sample
    all_indices = list(range(len(dataset)))
    random.Random(42).shuffle(all_indices)  # Deterministic shuffle

    for idx in all_indices:
        if current_tokens >= target_tokens:
            break
        indices.append(idx)
        current_tokens += dataset[idx]["num-tokens"]

    logger.info(
        "[Sampling] Finished sampling: selected %d documents, ~%d tokens.",
        len(indices),
        current_tokens,
    )

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
    logger.info(
        "Starting tokenization and grouping with max_seq_len=%d, num_proc=%d",
        max_seq_len,
        num_proc,
    )

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

    logger.info(
        "Completed tokenization and grouping. Split sizes: %s",
        {k: len(v) for k, v in grouped.items()},
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
    logger.info(
        "Creating dataloaders from dataset '%s' with max_seq_len=%d, batch_size=%d, "
        "num_proc=%d, num_workers=%d, max_tokens=%s, validation_split_ratio=%.2f",
        dataset_path,
        max_seq_len,
        batch_size,
        num_proc,
        num_workers,
        str(max_tokens),
        validation_split_ratio,
    )

    dataset = load_lang_datasets(
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

    logger.info(
        "Created dataloaders: train_batches_per_epoch=%d (train_samples=%d), "
        "eval_samples=%d",
        steps_per_epoch,
        len(grouped["train"]),
        len(grouped["validation"]),
    )

    return train_dataloader, eval_dataloader, steps_per_epoch
