#!/usr/bin/env python
"""Preprocess an atlas into model inputs.

Recentre every ACR on a fixed 1,344-bp window, one-hot encode it, drop windows
with assembly gaps / ambiguous bases, and split ACRs disjointly into
train/val/test. Writes the sequence tensors and the matched binary cell x ACR
accessibility matrix used for training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pysam
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES, INPUT_LEN, RANDOM_SEED
from peagent.sequence import one_hot, center_window


def split_ids(n, seed, frac=(0.9, 0.05, 0.05)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = SPECIES[args.species]

    peaks = [l.split()[:3] for l in open(cfg["peaks_bed"])]
    fasta = pysam.FastaFile(str(cfg["genome_fasta"]))

    seqs, keep = [], []
    for i, (chrom, start, end) in enumerate(peaks):
        s = center_window(fasta, chrom, int(start), int(end), INPUT_LEN)
        oh = one_hot(s, INPUT_LEN)
        if oh.sum() < INPUT_LEN:            # drop gaps / N-containing windows
            continue
        seqs.append(oh.astype(np.int8))
        keep.append(i)
    seqs = np.stack(seqs)
    keep = np.array(keep)

    atlas = ad.read_h5ad(cfg["atlas_h5ad"])
    m = atlas.X.T.tocsr()[keep]             # ACR x cell binary matrix, aligned to seqs
    tr, va, te = split_ids(len(keep), RANDOM_SEED)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        with h5py.File(args.out / f"{name}_seqs.h5", "w") as f:
            f["X"] = seqs[ids]
        sparse.save_npz(args.out / f"m_{name}.npz", m[ids].tocsr())
    np.save(args.out / "peak_index.npy", keep)
    print(f"{args.species}: {len(keep)} ACRs -> train {len(tr)} / val {len(va)} / test {len(te)}")


if __name__ == "__main__":
    main()
