#!/usr/bin/env bash
# Demo on the simulated mini-atlas (demo/data/demo): preprocess -> train -> evaluate, CPU only.
# Usage (from the repository root, peagent env active): bash demo/run_demo.sh [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
export PEAGENT_DATA_ROOT=demo/data
OUT=${1:-demo_output}

python 01_training/01_preprocess_acrs.py --species demo --out "$OUT/inputs"
python 01_training/02_train_model.py --data "$OUT/inputs" --out "$OUT/model" --epochs 20 --patience 5 --batch_size 32
python 02_evaluation/01_per_peak_auroc.py --species demo --data "$OUT/inputs" --model "$OUT/model/best_model.h5" --out "$OUT/eval"
python 02_evaluation/02_per_group_metrics.py --species demo --data "$OUT/inputs" --model "$OUT/model/best_model.h5" --out "$OUT/eval"
