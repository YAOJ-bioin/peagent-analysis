"""Cross-species prediction and evaluation.

Score the target species' ACR windows with the source species' model and compare
to the target's observed pseudobulk accessibility, per semantically matched
cell-type group. Evaluated globally over all peaks and within 500-kb windows with
Pearson/Spearman correlation, MAE, AUROC and AUPR. Self (within-species) pairs
provide the upper-bound baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pysam
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, INPUT_LEN, MIN_CELLS_PER_GROUP
from peagent.groups import load_model, predict, cell_groups, pseudobulk_truth, group_pred
from peagent.sequence import one_hot, center_window


def target_windows(cfg):
    fasta = pysam.FastaFile(str(cfg["genome_fasta"]))
    seqs, coords = [], []
    for chrom, start, end in (l.split()[:3] for l in open(cfg["peaks_bed"])):
        seqs.append(one_hot(center_window(fasta, chrom, int(start), int(end), INPUT_LEN)))
        coords.append((chrom, (int(start) + int(end)) // 2))
    return np.stack(seqs), coords


def semantic_map(src_groups, tgt_groups):
    """Match source->target groups by token overlap of tissue/cell-type names."""
    pairs = {}
    for t in tgt_groups:
        tok_t = set(t.lower().replace(".", " ").split())
        best = max(src_groups, key=lambda s: len(tok_t & set(s.lower().replace(".", " ").split())))
        if tok_t & set(best.lower().replace(".", " ").split()):
            pairs[t] = best
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=SPECIES)
    ap.add_argument("--target", required=True, choices=SPECIES)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    src, tgt = SPECIES[args.source], SPECIES[args.target]

    seqs, _ = target_windows(tgt)
    tgt_atlas = ad.read_h5ad(tgt["atlas_h5ad"])
    src_atlas = ad.read_h5ad(src["atlas_h5ad"])

    model = load_model(src["model"], src["n_cells"])            # source model, target sequences
    src_groups = cell_groups(src_atlas, src, MIN_CELLS_PER_GROUP)
    tgt_groups = cell_groups(tgt_atlas, tgt, MIN_CELLS_PER_GROUP)
    mapping = semantic_map(src_groups, tgt_groups)

    pred = group_pred(predict(model, seqs), src_groups)         # src groups x peaks
    frac, obs = pseudobulk_truth(tgt_atlas, tgt_groups)

    rows = []
    for tgt_g, src_g in mapping.items():
        p = pred[list(src_groups).index(src_g)]
        f, y = frac[list(tgt_groups).index(tgt_g)], obs[list(tgt_groups).index(tgt_g)]
        if not (0 < y.sum() < len(y)):
            continue
        rows.append({"target_group": tgt_g, "source_group": src_g,
                     "pcc": pearsonr(f, p)[0], "spearman": spearmanr(f, p)[0],
                     "mae": np.mean(np.abs(f - p)),
                     "auroc": roc_auc_score(y, p), "aupr": average_precision_score(y, p)})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.source}_to_{args.target}.csv", index=False)
    print(f"{args.source}->{args.target}: {len(df)} matched groups  "
          f"median AUROC={df.auroc.median():.3f}")


if __name__ == "__main__":
    main()
