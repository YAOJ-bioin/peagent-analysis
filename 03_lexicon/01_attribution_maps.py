"""Bulk and cell-type attribution maps for a set of ACRs.

Bottleneck ISM is computed once per ACR, then projected through the final dense
layer onto (i) the cell-averaged weight vector -> bulk track and (ii) each
tissue x cell-type group -> cell-type track. Tracks are converted to hypothetical
contribution scores (mean-centred, tissue-mean subtracted) for motif discovery.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, MIN_CELLS_PER_GROUP
from peagent.groups import load_model, cell_groups
from peagent.model import cell_head_weights
from peagent.attribution import bottleneck_ism, project, contribution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--seqs", required=True, type=Path, help="one-hot ACR windows (.npy)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", choices=["bulk", "celltype"], default="bulk")
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    seqs = np.load(args.seqs)                                   # (n_acr, L, 4)
    atlas = ad.read_h5ad(cfg["atlas_h5ad"])
    model = load_model(cfg["model"], cfg["n_cells"])
    W, b = cell_head_weights(model)

    groups = cell_groups(atlas, cfg, MIN_CELLS_PER_GROUP)
    targets = {"bulk": {"bulk": np.arange(cfg["n_cells"])}}["bulk"] if args.level == "bulk" \
        else groups
    weights = {g: W[:, idx].mean(axis=1) for g, idx in targets.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out / f"{args.species}_{args.level}_contrib.h5", "w") as f:
        for g, w in weights.items():
            contribs = np.stack([
                contribution(project(bottleneck_ism(model, oh), w), oh) for oh in seqs])
            f.create_dataset(g.replace("/", "_"), data=contribs.astype(np.float32))
    print(f"{args.species} {args.level}: {seqs.shape[0]} ACRs x {len(weights)} tracks")


if __name__ == "__main__":
    main()
