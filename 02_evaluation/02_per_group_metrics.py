"""Per-cell-group profile metrics with a shuffled-label control.

For each tissue x cell-type group, correlate the predicted accessibility profile
over held-out ACRs with the observed pseudobulk profile (Pearson PCC) and score
classification (AUROC, AUPR). A label-shuffled control gives the null AUROC/AUPR
so the real signal can be read against chance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, MIN_CELLS_PER_GROUP, RANDOM_SEED
from peagent.groups import load_model, predict, cell_groups, pseudobulk_truth, group_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    seqs = h5py.File(args.data / "test_seqs.h5", "r")["X"][:]
    test_idx = np.load(args.data / "peak_index.npy")
    atlas = ad.read_h5ad(cfg["atlas_h5ad"])

    model = load_model(cfg["model"], cfg["n_cells"])
    groups = cell_groups(atlas, cfg, MIN_CELLS_PER_GROUP)
    pred = group_pred(predict(model, seqs), groups)             # groups x peaks
    frac, obs = pseudobulk_truth(atlas[:, test_idx], groups)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for g, name in enumerate(groups):
        y, p, f = obs[g], pred[g], frac[g]
        if not (0 < y.sum() < len(y)):
            continue
        perm = rng.permutation(len(y))                          # shuffle-label control
        rows.append({
            "group": name, "n_open": int(y.sum()),
            "pcc": pearsonr(f, p)[0],
            "auroc": roc_auc_score(y, p), "aupr": average_precision_score(y, p),
            "auroc_shuffle": roc_auc_score(y, p[perm]),
            "aupr_shuffle": average_precision_score(y, p[perm]),
        })
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.species}_per_group_metrics.csv", index=False)
    print(f"{args.species}: {len(df)} groups  median PCC={df.pcc.median():.3f}  "
          f"median AUPR={df.aupr.median():.3f} (shuffle {df.aupr_shuffle.median():.3f})")


if __name__ == "__main__":
    main()
