"""Variant-effect prediction and caQTL benchmarking.

For each variant in an ACR, build REF and ALT 1,344-bp windows centred on the
variant, score both, and report the signed effect delta_logit = f(ALT) - f(REF)
on the pre-sigmoid accessibility logit per cell type. Fine-mapped caQTLs
(positives) are ranked against MAF/ACR-matched negatives by |delta_logit|.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy.special import logit
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, INPUT_LEN
from peagent.groups import load_model, predict
from peagent.sequence import one_hot, center_window, substitute

HALF = INPUT_LEN // 2


def score(model, fasta, variants, cfg):
    """delta_logit per variant, averaged over cell types (mean pre-sigmoid logit)."""
    deltas = []
    for _, v in variants.iterrows():
        win = center_window(fasta, v["chrom"], v["pos"], v["pos"] + 1, INPUT_LEN)
        ref = substitute(win, HALF, v["ref"])
        alt = substitute(win, HALF, v["alt"])
        pr, pa = predict(model, [one_hot(ref), one_hot(alt)])
        deltas.append(float((logit(pa) - logit(pr)).mean()))
    return np.array(deltas)


def benchmark(df, cutoffs=(0.0, 0.05, 0.1)):
    """AUROC / AP of |delta_logit| separating fine-mapped positives from negatives."""
    out = []
    for c in cutoffs:
        sub = df[df["abs_delta"] >= c]
        if sub["label"].nunique() < 2:
            continue
        w = wilcoxon(sub[sub.label == 1].abs_delta.values[:len(sub[sub.label == 0])] -
                     sub[sub.label == 0].abs_delta.values[:len(sub[sub.label == 1])],
                     alternative="greater")[1] if min((sub.label == 1).sum(),
                                                       (sub.label == 0).sum()) else 1.0
        out.append({"cutoff": c, "n": len(sub),
                    "auroc": roc_auc_score(sub.label, sub.abs_delta),
                    "ap": average_precision_score(sub.label, sub.abs_delta),
                    "wilcoxon_p": w})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--variants", required=True, type=Path,
                    help="tsv: chrom pos ref alt label(1=caQTL,0=matched negative)")
    ap.add_argument("--model", type=Path, help="override config model (e.g. population model)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    variants = pd.read_csv(args.variants, sep="\t")
    model = load_model(args.model or cfg["model"], cfg["n_cells"])
    fasta = pysam.FastaFile(str(cfg["genome_fasta"]))

    variants["delta_logit"] = score(model, fasta, variants, cfg)
    variants["abs_delta"] = variants["delta_logit"].abs()
    bench = benchmark(variants)

    args.out.mkdir(parents=True, exist_ok=True)
    variants.to_csv(args.out / f"{args.species}_variant_effects.tsv", sep="\t", index=False)
    bench.to_csv(args.out / f"{args.species}_caqtl_benchmark.csv", index=False)
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
