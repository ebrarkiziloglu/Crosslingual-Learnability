import argparse
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

def prepare_dataset(dataset, tokenizer):
    # Keep only the text column and tokenize dynamically
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False,
            max_length=256,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )

def train_phase(model, tokenizer, dataset, output_dir, steps=5000):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=steps,
        learning_rate=5e-4,
        save_steps=1000,
        logging_steps=100,
        warmup_steps=500,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1_path", required=True)
    parser.add_argument("--l2_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load datasets
    ds_l1 = load_dataset(args.l1_path)["train"]
    ds_l2 = load_dataset(args.l2_path)["train"]

    tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

    # Prepare tokenized datasets
    ds_l1_tok = prepare_dataset(ds_l1, tokenizer)
    ds_l2_tok = prepare_dataset(ds_l2, tokenizer)

    # Build BERT config (small)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=6,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        max_position_embeddings=512,
    )

    model = BertForMaskedLM(config)

    print("=== L1 Pretraining ===")
    model = train_phase(model, tokenizer, ds_l1_tok, args.out + "/phase1", steps=8000)

    print("=== L2 Continual Pretraining ===")
    model = train_phase(model, tokenizer, ds_l2_tok, args.out + "/phase2", steps=8000)

    print("Done!")
