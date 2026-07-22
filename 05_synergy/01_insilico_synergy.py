"""In-silico motif-pair synergy by marginalization.

For a motif pair in one tissue x cell-type context, insert motif A and motif B
alone and jointly (over a spacing x orientation grid) into GC-matched background
sequences, and test whether the joint effect exceeds the additive expectation
(dJ - dS > 0) with a one-sided paired Wilcoxon test. The cell-group readout is
the pre-sigmoid pseudobulk logit f_g(x) = z(x)·w_g + b_g.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, INPUT_LEN
from peagent.groups import load_model
from peagent.model import bottleneck_model, cell_head_weights
from peagent.sequence import one_hot

C = INPUT_LEN // 2
SPACINGS = list(range(0, 31)) + list(range(35, 205, 5))
HETERO = ["++", "+-", "-+", "--"]
HOMO = ["++", "+-", "--"]
RC = {"A": "T", "C": "G", "G": "C", "T": "A"}


def rc(s):
    return "".join(RC[b] for b in reversed(s))


def group_logit(embed, w_g, b_g, seqs):
    z = embed.predict(np.stack([one_hot(s, INPUT_LEN) for s in seqs]), verbose=0)
    return z @ w_g + b_g


def implant(bg, motif, pos):
    return bg[:pos] + motif + bg[pos + len(motif):]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--pairs", required=True, type=Path, help="motif pair + consensus + group tsv")
    ap.add_argument("--backgrounds", required=True, type=Path, help="GC-matched seqs (.txt)")
    ap.add_argument("--cell-index", required=True, type=Path, help="group -> cell indices (npz)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    model = load_model(cfg["model"], cfg["n_cells"])
    embed = bottleneck_model(model)
    W, b = cell_head_weights(model)
    backgrounds = [l.strip() for l in open(args.backgrounds)]
    cell_idx = dict(np.load(args.cell_index, allow_pickle=True).items())
    pairs = pd.read_csv(args.pairs, sep="\t")

    records = []
    for _, r in pairs.iterrows():
        idx = cell_idx[r["group"]]
        w_g, b_g = W[:, idx].mean(1), float(b[idx].mean())
        A, B = r["consensus_a"].strip("N"), r["consensus_b"].strip("N")
        orients = HOMO if r["motif_a"] == r["motif_b"] else HETERO

        base = group_logit(embed, w_g, b_g, backgrounds)
        dA = group_logit(embed, w_g, b_g, [implant(bg, A, C) for bg in backgrounds]) - base
        for orient in orients:
            a = A if orient[0] == "+" else rc(A)
            bmot = B if orient[1] == "+" else rc(B)
            dB = group_logit(embed, w_g, b_g, [implant(bg, bmot, C) for bg in backgrounds]) - base
            for d in SPACINGS:
                gap = d - (len(A) + len(B)) / 2
                if gap < 0:                                    # overlapping: not eligible
                    continue
                joint = [implant(implant(bg, a, C - d // 2), bmot, C - d // 2 + d)
                         for bg in backgrounds]
                dJ = group_logit(embed, w_g, b_g, joint) - base
                syn = dJ - (dA + dB)
                p = wilcoxon(syn, alternative="greater")[1] if np.any(syn) else 1.0
                records.append({"motif_a": r["motif_a"], "motif_b": r["motif_b"],
                                "orient": orient, "spacing": d, "edge_gap": gap,
                                "mean_dJ": dJ.mean(), "mean_syn": syn.mean(), "p": p})

    df = pd.DataFrame(records)
    df["padj"] = multipletests(df["p"], method="fdr_bh")[1]     # single BH across all arrangements
    best = df.sort_values("mean_dJ").groupby(["motif_a", "motif_b"]).tail(1)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "synergy_arrangements.tsv", sep="\t", index=False)
    best.to_csv(args.out / "synergy_pair_optimal.tsv", sep="\t", index=False)
    print(f"{best.shape[0]} pairs tested; {int((best.padj < 0.01).sum())} synergistic (padj<0.01)")


if __name__ == "__main__":
    main()
