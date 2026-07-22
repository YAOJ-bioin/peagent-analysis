"""Grass-conserved vs lineage-specific ACR regulatory grammar (rice).

Partition rice ACRs by cross-species orthology into grass-conserved and
rice-specific classes, compare the model-attributed contribution signal between
them, and export the highest/lowest-contribution positions per class as BED for
intersection with phyloP conservation and rice subpopulation variation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, INPUT_LEN
from peagent.groups import load_model
from peagent.model import cell_head_weights
from peagent.attribution import bottleneck_ism, project, contribution
from peagent.sequence import one_hot, center_window

GRASS = ["orth_grass_1", "orth_grass_2", "orth_grass_3", "orth_grass_4"]
RICE = ["spec_rice_1", "spec_rice_2", "spec_rice_3", "spec_rice_4"]
SPECIES_SPEC = ["spec_sp_1", "spec_sp_2", "spec_sp_3", "spec_sp_4"]


def partition(table):
    """Label ACRs grass-conserved / rice-specific from orthology flag columns."""
    t = table[(table[SPECIES_SPEC] == "no").all(axis=1)]       # drop idiosyncratic rows
    grass = t[(t[GRASS] == "yes").all(axis=1) & (t[RICE] != "yes").all(axis=1)]
    rice = t[(t[RICE] == "yes").all(axis=1) & (t[GRASS] != "yes").all(axis=1)]
    return grass, rice


def acr_contrib(model, fasta, cfg, coords):
    W, _ = cell_head_weights(model)
    w_bulk = W.mean(axis=1)
    out = []
    for chrom, start, end in coords:
        oh = one_hot(center_window(fasta, chrom, int(start), int(end), INPUT_LEN))
        out.append(contribution(project(bottleneck_ism(model, oh), w_bulk), oh))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orthology", required=True, type=Path, help="rice comparative ACR table")
    ap.add_argument("--phylop", type=Path, help="per-site grass phyloP bigwig/table (optional)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES["rice"]

    table = pd.read_csv(args.orthology, sep="\t")
    grass, rice = partition(table)
    model = load_model(cfg["model"], cfg["n_cells"])
    fasta = pysam.FastaFile(str(cfg["genome_fasta"]))

    cg = acr_contrib(model, fasta, cfg, grass[["chrom", "start", "end"]].values)
    cr = acr_contrib(model, fasta, cfg, rice[["chrom", "start", "end"]].values)
    stat, p = mannwhitneyu(np.abs(cg).ravel(), np.abs(cr).ravel(), alternative="greater")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"class": ["grass_conserved", "rice_specific"],
                  "n_acr": [len(grass), len(rice)],
                  "mean_abs_contrib": [np.abs(cg).mean(), np.abs(cr).mean()]}
                 ).to_csv(args.out / "conservation_contrib_summary.csv", index=False)
    print(f"grass={len(grass)} rice={len(rice)}  |contrib| higher in conserved p={p:.2e}")

    if args.phylop:                                            # phyloP AUROC for hi vs lo sites
        flat = np.concatenate([cg.ravel(), cr.ravel()])
        hi = flat >= np.quantile(flat, 0.99)
        lo = flat <= np.quantile(flat, 0.01)
        phylop = np.load(args.phylop)                          # site-aligned phyloP
        mask = hi | lo
        print(f"phyloP AUROC (hi vs lo contrib): {roc_auc_score(hi[mask], phylop[mask]):.3f}")


if __name__ == "__main__":
    main()
