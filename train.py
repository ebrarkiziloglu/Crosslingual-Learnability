# train.py
import argparse
import math
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, set_seed, DebertaV2Tokenizer, AutoConfig, AutoModelForMaskedLM
from transformers.optimization import get_cosine_schedule_with_warmup

from dataloaders import create_dataloaders


def mask_batch(
    batch: dict,
    tokenizer,
    mlm_prob: float = 0.15,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
) -> dict:
    """
    Simpler version of the masking strategy used in `train_mask.py`.

    - Select ~mlm_prob tokens per sequence (excluding padding).
    - Of the selected tokens:
        80% -> [MASK]
        10% -> random token
        10% -> left unchanged (but still predicted)
    """
    masked_batch = batch.copy()

    input_ids = masked_batch["input_ids"]
    labels = input_ids.clone()

    # Create probability mask for each token (uniform over non-pad tokens)
    probability_matrix = torch.full(labels.shape,
                                    mlm_prob,
                                    device=input_ids.device)
    probability_matrix[input_ids == tokenizer.pad_token_id] = 0.0

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Decide which masked tokens are replaced with [MASK], random, or kept
    indices_replaced = (torch.bernoulli(
        torch.full(labels.shape, mask_replace_prob,
                   device=input_ids.device)).bool()
                        & masked_indices)
    indices_random = (torch.bernoulli(
        torch.full(labels.shape, random_replace_prob,
                   device=input_ids.device)).bool()
                      & masked_indices
                      & ~indices_replaced)
    indices_original = masked_indices & ~indices_replaced & ~indices_random

    # 80% -> [MASK]
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    random_words = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=input_ids.shape,
        device=input_ids.device,
    )
    input_ids[indices_random] = random_words[indices_random]

    # 10% left as-is (indices_original) – nothing to change in input_ids
    masked_batch["input_ids"] = input_ids
    masked_batch["labels"] = labels
    return masked_batch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train(
    model: BertForMaskedLM,
    tokenizer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    total_steps: int,
    lr: float,
    grad_acc: int,
    logging_steps: int,
    eval_steps: int,
    output_path: str,
    device: torch.device,
    save_output: bool = True,
):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 100, 1),
        num_training_steps=total_steps,
    )

    global_step = 0
    running_loss = 0.0

    for epoch in range(math.ceil(total_steps / len(train_dataloader))):
        for step, batch in enumerate(train_dataloader):
            masked_batch = move_batch_to_device(mask_batch(batch, tokenizer),
                                                device)

            outputs = model(**masked_batch)
            loss = outputs.loss / grad_acc
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    print(
                        f"Step {global_step}/{total_steps} "
                        f"- loss: {avg_loss:.4f} "
                        f"- lr: {scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                    running_loss = 0.0

                if global_step % eval_steps == 0:
                    eval_metrics = evaluate(model, tokenizer, eval_dataloader,
                                            device)
                    print(
                        f"[Eval @ step {global_step}] "
                        f"loss: {eval_metrics['loss']:.4f} "
                        f"acc: {eval_metrics['acc']:.2f}",
                        flush=True,
                    )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    # Final evaluation & save
    eval_metrics = evaluate(model, tokenizer, eval_dataloader, device)
    print(
        f"[Final Eval] loss: {eval_metrics['loss']:.4f} "
        f"acc: {eval_metrics['acc']:.2f}",
        flush=True,
    )

    if save_output:
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Saved model and tokenizer to {output_path}", flush=True)



def evaluate(model, tokenizer, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            masked_batch = mask_batch(batch, tokenizer)
            masked_batch = move_batch_to_device(masked_batch, device)

            outputs = model(**masked_batch)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            labels = masked_batch["labels"]
            mask = labels != -100

            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    model.train()
    avg_loss = total_loss / max(len(dataloader), 1)
    acc = 100 * correct / max(total, 1)
    return {"loss": avg_loss, "acc": acc}


def build_model_and_tokenizer(
    tokenizer_path: str,
    model_path: str,
    lower: bool = False,
    hidden_size: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    intermediate_size: int = 3072,
    max_position_embeddings: int = 512,
) -> Tuple[BertForMaskedLM, BertTokenizerFast]:
    # tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # config = BertConfig(
    #     vocab_size=tokenizer.vocab_size,
    #     num_hidden_layers=num_layers,
    #     hidden_size=hidden_size,
    #     intermediate_size=intermediate_size,
    #     num_attention_heads=num_heads,
    #     max_position_embeddings=max_position_embeddings,
    # )

    # model = BertForMaskedLM(config)
    # return model, tokenizer
    print(f"Building model and tokenizer from {tokenizer_path} and {model_path}")
    tokenizer = DebertaV2Tokenizer(tokenizer_path, do_lower_case=lower)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    config.vocab_size = tokenizer.vocab_size
    config.num_hidden_layers = num_hidden_layers
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_attention_heads = num_attention_heads
    config.max_position_embeddings = max_position_embeddings

    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = AutoModelForMaskedLM.from_config(config,
                                             trust_remote_code=True)
    return model, tokenizer

# python train.py --mode mono --l1 eng --l1_path BabyLM-community/babylm-eng --out models/mono_eng --tokenizer_path tokenizers/bb24.model 
# python train.py --mode multi --l1 eng --l2 deu --l1_path BabyLM-community/babylm-eng --l2_path BabyLM-community/babylm-deu --out models/multi_eng_to_deu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base", help="Path to the model to use")
    parser.add_argument(
        "--mode",
        type=str,
        default="mono",
        choices=["mono", "multi"],
        help=
        "Mode of training: 'mono' (monolingual) or 'multi' (bilingual, L1 then L2).",
    )
    parser.add_argument(
        "--l1",
        type=str,
        required=True,
        help="Language identifier for L1 (e.g. 'en'). Used in output naming.",
    )
    parser.add_argument(
        "--l2",
        type=str,
        default=None,
        help=
        "Language identifier for L2 (e.g. 'de'). Required for 'multi' mode.",
    )
    parser.add_argument(
        "--l1_path",
        type=str,
        default=None,
        help="Path to the L1 dataset. Required for 'mono' and 'multi' modes.",
    )
    parser.add_argument(
        "--l2_path",
        type=str,
        default=None,
        help="Path to the L2 dataset. Required for 'multi' mode.",
    )
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help=
        "Base output directory where run-specific subfolders will be created.",
    )

    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower", type=bool, default=False)

    # For Tier 1 languages (100M tokens), sample 10M (10%)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to sample (10M for Tier 1 languages). "
        "If None, uses all data.")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. If not provided, uses HF_TOKEN env var.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Basic argument validation depending on mode
    if args.mode == "mono":
        if args.l1_path is None:
            raise ValueError(
                "In 'mono' mode, --l1_path must be provided (path to L1 dataset)."
            )
    elif args.mode == "multi":
        missing = []
        if args.l1_path is None:
            missing.append("--l1_path")
        if args.l2_path is None:
            missing.append("--l2_path")
        if args.l2 is None:
            missing.append("--l2")
        if missing:
            raise ValueError(
                f"In 'multi' mode, the following arguments are required: {', '.join(missing)}"
            )

    # Build model + tokenizer (shared across phases)
    if args.tokenizer_path == "tokenizer":
        tokenizer_path = args.model_path
    else:
        tokenizer_path = args.tokenizer_path
    model, tokenizer = build_model_and_tokenizer(tokenizer_path,
                                                 args.model_path, args.lower)

    if args.mode == "mono":
        # Monolingual training on L1
        train_loader, eval_loader, steps_per_epoch = create_dataloaders(
            dataset_path=args.l1_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_proc=args.cpus,
            num_workers=args.cpus,
            hf_token=args.hf_token,
            max_tokens=args.max_tokens,
        )

        total_steps = args.epochs * steps_per_epoch
        print(
            f"[Mono] Training on {args.l1} with {total_steps} total steps "
            f"({args.epochs} epochs x {steps_per_epoch} steps/epoch)",
            flush=True,
        )

        mono_output = os.path.join(args.out, f"mono_{args.l1}")
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            total_steps=total_steps,
            lr=args.lr,
            grad_acc=args.grad_acc,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            output_path=mono_output,
            device=device,
            save_output=True,
        )

    elif args.mode == "multi":
        # Phase 1: train on L1
        train_loader_l1, eval_loader_l1, steps_per_epoch_l1 = create_dataloaders(
            dataset_path=args.l1_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_proc=args.cpus,
            num_workers=args.cpus,
            hf_token=args.hf_token,
            max_tokens=args.max_tokens,
        )

        total_steps_l1 = args.epochs * steps_per_epoch_l1
        print(
            f"[Multi] Phase 1: Training on {args.l1} with {total_steps_l1} total steps "
            f"({args.epochs} epochs x {steps_per_epoch_l1} steps/epoch)",
            flush=True,
        )

        train(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader_l1,
            eval_dataloader=eval_loader_l1,
            total_steps=total_steps_l1,
            lr=args.lr,
            grad_acc=args.grad_acc,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            output_path=args.out,  # not used because save_output=False
            device=device,
            save_output=False,
        )

        # Phase 2: continue training on L2
        train_loader_l2, eval_loader_l2, steps_per_epoch_l2 = create_dataloaders(
            dataset_path=args.l2_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_proc=args.cpus,
            num_workers=args.cpus,
            hf_token=args.hf_token,
            max_tokens=args.max_tokens,
        )

        total_steps_l2 = args.epochs * steps_per_epoch_l2
        print(
            f"[Multi] Phase 2: Training on {args.l2} with {total_steps_l2} total steps "
            f"({args.epochs} epochs x {steps_per_epoch_l2} steps/epoch)",
            flush=True,
        )

        multi_output = os.path.join(args.out, f"multi_{args.l1}_to_{args.l2}")
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader_l2,
            eval_dataloader=eval_loader_l2,
            total_steps=total_steps_l2,
            lr=args.lr,
            grad_acc=args.grad_acc,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            output_path=multi_output,
            device=device,
            save_output=True,
        )


if __name__ == "__main__":
    main()
