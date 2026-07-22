"""Classify meta-patterns into the regulatory lexicon.

From each meta-pattern CWM we derive a trimmed core (30% of max per-position L1
norm), L1 sub-peaks, and a flank ratio, then combine them with the best TomTom
hit q-value to assign one of six structural classes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Thresholds validated against a manually curated ground-truth set
FR_COMPOSITE = 0.20     # min flank ratio (or >=2 peaks) to enter the composite path
FR_HOMO = 0.15          # single-site: >= -> homocomposite
BWF_WIDTH = 13          # trimmed width > this -> base_with_flank
Q_MATCH = 0.10          # best TomTom q below this -> annotated single site
FR_NOISY = 0.50         # diffuse signal without a motif match


def cwm_features(cwm):
    l1 = np.abs(cwm).sum(axis=1)
    core = l1 >= 0.30 * l1.max()
    lo, hi = np.argmax(core), len(core) - np.argmax(core[::-1])
    width = hi - lo
    peaks, _ = find_peaks(l1, distance=5, prominence=0.30 * l1.max())
    if len(peaks):
        primary = peaks[np.argmax(l1[peaks])]
        window = (np.arange(len(l1)) < primary - 8) | (np.arange(len(l1)) > primary + 8)
        flank_ratio = l1[core & window].sum() / max(l1[core].sum(), 1e-9)
    else:
        flank_ratio = 1.0
    return width, len(peaks), float(flank_ratio)


def classify(width, n_peaks, flank_ratio, best_q, same_family):
    composite = flank_ratio >= FR_COMPOSITE or n_peaks >= 2
    if composite:
        if flank_ratio >= FR_NOISY and best_q >= Q_MATCH:
            return "noisy"
        return "homocomposite" if same_family else "heterocomposite"
    if best_q >= Q_MATCH:
        return "unresolved" if flank_ratio < FR_NOISY else "noisy"
    if flank_ratio >= FR_HOMO:
        return "homocomposite"
    return "base_with_flank" if width > BWF_WIDTH else "base"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwms", required=True, type=Path, help="meta-pattern CWMs (.npz: name->LxA)")
    ap.add_argument("--tomtom", required=True, type=Path, help="best hit per pattern (tsv)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cwms = np.load(args.cwms)
    tt = pd.read_csv(args.tomtom, sep="\t").set_index("pattern")
    rows = []
    for name in cwms.files:
        w, n, fr = cwm_features(cwms[name])
        q = float(tt.loc[name, "q_value"]) if name in tt.index else 1.0
        same = bool(tt.loc[name, "same_family"]) if name in tt.index else False
        rows.append({"pattern": name, "trimmed_width": w, "n_peaks": n,
                     "flank_ratio": fr, "best_q": q,
                     "type": classify(w, n, fr, q, same)})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "lexicon_classification.tsv", sep="\t", index=False)
    print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
