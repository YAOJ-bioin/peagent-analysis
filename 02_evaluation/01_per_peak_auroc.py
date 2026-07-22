#!/usr/bin/env python
"""Per-held-out-peak classification accuracy.

For each test ACR, AUROC/AUPR of the model's predicted accessibility against the
observed binary accessibility across tissue x cell-type groups (>= 50 cells).
Reports the per-peak distribution and the median used in the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, MIN_CELLS_PER_GROUP
from peagent.groups import load_model, predict, cell_groups, pseudobulk_truth, group_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--data", required=True, type=Path, help="preprocess output dir")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    seqs = h5py.File(args.data / "test_seqs.h5", "r")["X"][:]
    test_idx = np.load(args.data / "peak_index.npy")            # ACR ids of test peaks
    atlas = ad.read_h5ad(cfg["atlas_h5ad"])

    model = load_model(cfg["model"], cfg["n_cells"])
    groups = cell_groups(atlas, cfg, MIN_CELLS_PER_GROUP)
    pred = group_pred(predict(model, seqs), groups)            # groups x peaks
    _, obs = pseudobulk_truth(atlas[:, test_idx], groups)      # groups x peaks (binary)

    rows = []
    for p in range(pred.shape[1]):
        y = obs[:, p]
        if 0 < y.sum() < len(y):                               # need both classes
            rows.append((roc_auc_score(y, pred[:, p]),
                         average_precision_score(y, pred[:, p])))
    df = pd.DataFrame(rows, columns=["auroc", "aupr"])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.species}_per_peak_auroc.csv", index=False)
    print(f"{args.species}: n={len(df)}  median AUROC={df.auroc.median():.3f}  "
          f"median AUPR={df.aupr.median():.3f}")


if __name__ == "__main__":
    main()
