"""Motif discovery (TF-MoDISco) and genome-wide instance calling (Fi-NeMo).

TF-MoDISco-lite on the contribution maps distils recurrent contribution-weight
matrices (CWMs); Fi-NeMo then competitively assigns those CWMs back to every ACR
attribution track under an L1 penalty, tallying instances per pattern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import SPECIES


def run_modisco(onehot, contrib, out_h5, window=500, max_seqlets=2000):
    import modiscolite
    pos, neg = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=contrib[:, :, None] * onehot,
        one_hot=onehot, max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=20, flank_size=5, target_seqlet_fdr=0.05)
    modiscolite.io.save_hdf5(out_h5, pos, neg, window_size=window)


def run_finemo(onehot, contrib, modisco_h5, out_dir, alpha=0.7):
    from finemo import evaluate, hitcaller
    cwms = evaluate.load_modisco_motifs(modisco_h5, trim_threshold=0.3)
    hits = hitcaller.fit_contribs(contrib, onehot, cwms, alpha=alpha)   # L1 competitive fit
    hits.to_csv(Path(out_dir) / "instances.tsv", sep="\t", index=False)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, choices=SPECIES)
    ap.add_argument("--onehot", required=True, type=Path)
    ap.add_argument("--contrib", required=True, type=Path, help="contribution scores (.npy)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    onehot = np.load(args.onehot)
    contrib = np.load(args.contrib)
    args.out.mkdir(parents=True, exist_ok=True)
    modisco_h5 = args.out / f"{args.species}_modisco.h5"

    run_modisco(onehot, contrib, str(modisco_h5))
    hits = run_finemo(onehot, contrib, str(modisco_h5), args.out, args.alpha)
    print(f"{args.species}: {hits['pattern'].nunique()} patterns, {len(hits)} instances "
          f"@ alpha={args.alpha}")


if __name__ == "__main__":
    main()
