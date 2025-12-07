#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

# Load conda
source ~/.bashrc

# Activate your environment
conda activate babylm-pretrain

python train.py --mode mono --l1 eng --l1_path BabyLM-community/babylm-eng --out models/mono_eng --tokenizer_path tokenizers/bb24.model --logging_steps 10000 --max_tokens 10000000 --out models/10M
