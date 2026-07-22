"""Cross-species meta-pattern clustering.

Aggregate every positive cell-type CWM from all three species, define pairwise
similarity as the maximum gapless-offset Pearson correlation over both
orientations, and collapse patterns into meta-patterns by constant-Potts-model
Leiden clustering. Each meta-pattern is the length-weighted mean CWM of its members.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_cwms(modisco_glob):
    from motif_compendium import MotifCompendium
    files = {Path(p).stem: p for p in glob.glob(modisco_glob)}
    return MotifCompendium.build_from_modisco(files, contrib_scores=True, region_size=500)


def offset_corr(a, b):
    """Max gapless-offset Pearson correlation over both strand orientations."""
    best = -1.0
    for mat in (b, b[::-1, ::-1]):                              # forward + reverse-complement
        for off in range(-(len(a) - 5), len(b) - 5):
            i0, j0 = max(0, off), max(0, -off)
            n = min(len(a) - i0, len(mat) - j0)
            if n < 5:
                continue
            x, y = a[i0:i0 + n].ravel(), mat[j0:j0 + n].ravel()
            if x.std() and y.std():
                best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modisco-glob", required=True, help="e.g. 'out/*_modisco.h5'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--resolution", type=float, default=1.0)
    args = ap.parse_args()

    comp = load_cwms(args.modisco_glob)
    meta = comp.cpm_leiden(similarity=offset_corr, threshold=args.threshold,
                           resolution=args.resolution, within_strand=True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta.save(args.out / "metapattern_compendium.h5")
    print(f"{comp.n_patterns} CWMs -> {meta.n_metapatterns} meta-patterns")


if __name__ == "__main__":
    main()
