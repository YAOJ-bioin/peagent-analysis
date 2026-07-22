"""Latent-space structure: cell embedding UMAP + Leiden vs annotated cell types.

The trained final-layer per-cell weights form a sequence-derived cell embedding.
Leiden-cluster it and score agreement with annotated cell types by normalised
mutual information (NMI) — the model never saw cell-type labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES
from peagent.groups import load_model
from peagent.model import cell_head_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    atlas = ad.read_h5ad(cfg["atlas_h5ad"])
    model = load_model(cfg["model"], cfg["n_cells"])
    W, _ = cell_head_weights(model)                            # (bottleneck, n_cells)

    emb = ad.AnnData(W.T, obs=atlas.obs.copy())               # cell x bottleneck
    sc.pp.neighbors(emb, use_rep="X")
    sc.tl.leiden(emb, resolution=1.0)
    sc.tl.umap(emb)

    labels = atlas.obs[cfg["celltype_col"]].astype(str).values
    nmi = normalized_mutual_info_score(labels, emb.obs["leiden"].values)

    args.out.mkdir(parents=True, exist_ok=True)
    emb.write(args.out / f"{args.species}_cell_embedding.h5ad")
    print(f"{args.species}: NMI(leiden, celltype) = {nmi:.3f}")


if __name__ == "__main__":
    main()
