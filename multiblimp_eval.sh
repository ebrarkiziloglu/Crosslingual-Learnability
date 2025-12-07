#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --output %j.out

# Load conda
source ~/.bashrc

# Activate your environment
conda activate babylm-pretrain

python multiblimp.py --model_dir models/mono_deu/ --languages eng,deu,nld,eus,ind --device cuda
python multiblimp.py --model_dir models/mono_nld/ --languages eng,deu,nld,eus,ind --device cuda
python multiblimp.py --model_dir models/mono_eng/ --languages eng,deu,nld,eus,ind --device cuda
python multiblimp.py --model_dir models/mono_eus/ --languages eng,deu,nld,eus,ind --device cuda
python multiblimp.py --model_dir models/mono_ind/ --languages eng,deu,nld,eus,ind --device cuda


