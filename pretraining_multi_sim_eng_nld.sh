#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

# Load conda
source ~/.bashrc

# Activate your environment
conda activate babylm-pretrain


python train.py --mode multi \
  --multi_training_type simultaneous --l1 eng --l2 nld \
  --l1_path BabyLM-community/babylm-nld --l2_path BabyLM-community/babylm-eng \
  --tokenizer_path tokenizers/bb24.model --logging_steps 100000 \
  --max_tokens 10000000 --out models/10M_simultaneous
