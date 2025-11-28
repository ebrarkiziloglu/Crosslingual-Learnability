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
  --l1 eng --l2 deu --l1_path BabyLM-community/babylm-eng --l2_path BabyLM-community/babylm-deu \
  --tokenizer_path tokenizers/bb24.model --logging_steps 1000 --max_tokens 100000
