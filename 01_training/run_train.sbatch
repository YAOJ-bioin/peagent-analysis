#!/bin/bash
#SBATCH --job-name=peagent_train
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=peagent_train_%j.out
#SBATCH --error=peagent_train_%j.err

# Preprocess + train the PEAgent model for one species on a single H100 GPU.
# Usage: sbatch run_train.sbatch <species> <data_dir> <out_dir>
SPECIES=${1:?species}
DATA=${2:?preprocess output dir}
OUT=${3:?model output dir}

source ~/.bashrc
set -euo pipefail
conda activate scbasset_py38
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib

echo "host=$(hostname) time=$(date) species=$SPECIES"
python 01_preprocess_acrs.py --species "$SPECIES" --out "$DATA"
python 02_train_model.py --data "$DATA" --out "$OUT"
