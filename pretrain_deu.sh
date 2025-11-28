#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

# Load conda
source ~/.bashrc

# Activate your environment
conda activate babylm-pretrain

# Run your training
python train.py --mode mono --l1 deu --l1_path BabyLM-community/babylm-deu --out models/mono_deu --tokenizer_path tokenizers/bb24.model --logging_steps 100 --max_tokens 10000
