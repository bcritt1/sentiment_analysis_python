#!/usr/bin/bash
#SBATCH --job-name=sentiment
#SBATCH --output=/home/users/%u/out/sentiment.%j.out
#SBATCH --error=/home/users/%u/err/sentiment.%j.err
#SBATCH -p hns
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
module load python/3.9.0
pip3 install nltk
pip install transformers -U
module load math py-pytorch py-torchvision
#no pytorch support for 3.10 yet...use this instead of nightly if it becomes supported
#!pip3 install torch torchvision torchaudio
#pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 sentiment.py
