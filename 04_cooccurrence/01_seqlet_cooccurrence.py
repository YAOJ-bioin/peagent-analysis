"""Base-motif co-occurrence from TF-MoDISco seqlets.

Map every base-motif seqlet back to its ACR (via example index) and count
pairwise co-occurrence as instance pairs within each ACR: n_A * n_B for a
heterotypic pair, n_A*(n_A-1)/2 for a homotypic pair. Counts are pooled across
species and every instance pair keeps its centre-to-centre distance.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_seqlets(path):
    """Long table: acr, motif, center (one row per base-motif seqlet)."""
    return pd.read_csv(path, sep="\t")


def cooccurrence(seqlets):
    rows = []
    for acr, sub in seqlets.groupby("acr"):
        by_motif = {m: g["center"].values for m, g in sub.groupby("motif")}
        motifs = sorted(by_motif)
        for a, b in combinations_with_replacement(motifs, 2):
            ca, cb = by_motif[a], by_motif[b]
            if a == b:
                pairs = [abs(x - y) for x, y in combinations(ca, 2)]     # n(n-1)/2
            else:
                pairs = [abs(x - y) for x in ca for y in cb]             # n_A * n_B
            for d in pairs:
                rows.append((a, b, d))
    df = pd.DataFrame(rows, columns=["motif_a", "motif_b", "distance"])
    summary = (df.groupby(["motif_a", "motif_b"])["distance"]
                 .agg(n_pairs="size", median="median", mean="mean", std="std")
                 .reset_index())
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlets", nargs="+", required=True, type=Path,
                    help="per-species seqlet->ACR tables")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-pairs", type=int, default=20)
    args = ap.parse_args()

    seqlets = pd.concat([load_seqlets(p) for p in args.seqlets], ignore_index=True)
    pairs, summary = cooccurrence(seqlets)
    summary = summary[summary["n_pairs"] > args.min_pairs].sort_values("n_pairs", ascending=False)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out / "instance_pairs.tsv", sep="\t", index=False)
    summary.to_csv(args.out / "cooccurrence_summary.tsv", sep="\t", index=False)
    print(f"{len(seqlets)} seqlets -> {len(summary)} candidate pairs (> {args.min_pairs} pairs)")


if __name__ == "__main__":
    main()
