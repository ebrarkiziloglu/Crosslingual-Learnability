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
python train.py --mode multi \
  --l1 eus --l2 eng --l1_path BabyLM-community/babylm-eus --l2_path BabyLM-community/babylm-eng \
  --tokenizer_path tokenizers/bb24.model --logging_steps 100000 --max_tokens 10000000 --out models/10M
