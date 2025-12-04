"""
Evaluate saved (DeBERTa-style) masked language models on the MultiBLiMP benchmark.

This code is adapted from the lm_eval_example.ipynb notebook:
https://github.com/jumelet/multiblimp/blob/main/lm_eval_example.ipynb

Differences from the original:
- loads local Hugging Face checkpoints saved by `train.py` (config + model.safetensors + tokenizer),
- uses `minicons.MaskedLMScorer` (masked LM / pseudo-log-likelihood) instead of a causal LM scorer,
- exposes a CLI for choosing which language configs to evaluate.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import get_dataset_config_names, load_dataset
from minicons import scorer
from tqdm.auto import tqdm


def load_mlm_model(model_dir: str, device: str):
    """
    Load a masked language model + tokenizer from a local HF directory using minicons.

    The directory should contain:
      - config.json
      - model.safetensors / pytorch_model.bin
      - tokenizer_config.json, vocab / merges / spm.model, etc.
    """
    # MaskedLMScorer will internally call AutoModelForMaskedLM / AutoTokenizer.from_pretrained(model_dir)
    ilm_model = scorer.MaskedLMScorer(model_dir, device)
    return ilm_model


def score_pair(ilm_model, sen: str, wrong_sen: str, max_length: int | None):
    """
    Score a grammatical / ungrammatical sentence pair with a masked LM scorer.

    Returns token-wise log-probabilities (or scores) for both sentences.
    """
    sen_len = len(ilm_model.tokenizer.tokenize(sen))
    wrong_sen_len = len(ilm_model.tokenizer.tokenize(wrong_sen))

    if (max_length is not None) and (
        (sen_len >= max_length) or (wrong_sen_len >= max_length)
    ):
        return 0.0, 0.0

    stimuli = [sen, wrong_sen]
    # For MaskedLMScorer, `sequence_score` yields (pseudo-)log-probabilities per token
    return ilm_model.sequence_score(stimuli, reduction=lambda x: x)


def score_tse(ilm_model, tse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scores and negative log-likelihoods for each sentence pair in the TSE dataframe.
    """
    tse_df = tse_df.copy()
    tse_df["sen_prob"] = pd.Series(dtype=object).astype(object)
    tse_df["wrong_prob"] = pd.Series(dtype=object).astype(object)

    max_length = None  # can be set to a context limit if needed

    for idx, row in tqdm(
        tse_df.iterrows(), total=len(tse_df), desc="Scoring sentence pairs"
    ):
        sen_prob, wrong_prob = score_pair(ilm_model, row.sen, row.wrong_sen, max_length)

        sen_nll = -sen_prob.sum().item()
        wrong_nll = -wrong_prob.sum().item()

        tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
        tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()

        tse_df.loc[idx, "sen_nll"] = sen_nll
        tse_df.loc[idx, "wrong_nll"] = wrong_nll
        tse_df.loc[idx, "delta"] = wrong_nll - sen_nll

    return tse_df


def evaluate_language(ilm_model, dataset_name: str, language: str) -> float:
    """
    Load a specific MultiBLiMP language config and compute accuracy.
    """
    df = load_dataset(dataset_name, language)["train"].to_pandas()
    df = score_tse(ilm_model, df)
    accuracy = float(np.mean(df["delta"] > 0))
    print(f"[Result] {language}: accuracy = {accuracy:.4f}")
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved masked language model on the MultiBLiMP benchmark."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Path to the HF model directory to evaluate "
            "(e.g. models/mono_eng or models/multi_eng_to_deu)."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jumelet/multiblimp",
        help="Hugging Face dataset name for MultiBLiMP.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="nld",
        help=(
            "Comma-separated list of MultiBLiMP configs / language codes to evaluate "
            "(e.g. 'eng,nld,deu')."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation (e.g. 'cuda' or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model_dir):
        raise ValueError(f"model_dir does not exist or is not a directory: {args.model_dir}")

    print(f"[Init] Loading model from '{args.model_dir}' on device '{args.device}'")
    ilm_model = load_mlm_model(args.model_dir, args.device)

    # Resolve available configs and user selection
    all_configs = get_dataset_config_names(args.dataset_name)
    print(f"[Init] Available MultiBLiMP configs: {all_configs}")

    requested = [cfg.strip() for cfg in args.languages.split(",") if cfg.strip()]
    for cfg in requested:
        if cfg not in all_configs:
            raise ValueError(
                f"Requested config '{cfg}' is not in available configs: {all_configs}"
            )

    print(f"[Init] Evaluating configs: {requested}")

    results = {}
    for cfg in requested:
        acc = evaluate_language(ilm_model, args.dataset_name, cfg)
        results[cfg] = acc

    # Summary
    print("\n=== MultiBLiMP Evaluation Summary ===")
    for cfg, acc in results.items():
        print(f"{cfg}: {acc:.4f}")
    if results:
        mean_acc = float(np.mean(list(results.values())))
        print(f"Mean accuracy over {len(results)} configs: {mean_acc:.4f}")


if __name__ == "__main__":
    main()

# Example command:
# python multiblimp.py \
#   --model_dir ../models/multi_eng_to_deu \
#   --languages eng,deu  \
#   --device cuda