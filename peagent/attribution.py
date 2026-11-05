"""Bottleneck in-silico mutagenesis (ISM) and contribution-score projection.

Because every single-cell output shares the 64-d bottleneck, ISM is computed
once per ACR at the bottleneck and then projected through the final dense layer
onto any cell group (bulk = cell-averaged weights, cell-type = per-group weights).
"""
from __future__ import annotations

import numpy as np

from .model import bottleneck_model


def bottleneck_ism(model, onehot: np.ndarray, batch: int = 512) -> np.ndarray:
    """Return the (L, 4, bottleneck) change-in-embedding tensor for one window."""
    embed = bottleneck_model(model)
    L = onehot.shape[0]
    ref = embed.predict(onehot[None], verbose=0)[0]                 # (bottleneck,)

    variants, index = [], []
    for pos in range(L):
        for base in range(4):
            mut = onehot.copy()
            mut[pos] = 0.0
            mut[pos, base] = 1.0
            variants.append(mut)
            index.append((pos, base))
    emb = embed.predict(np.asarray(variants), batch_size=batch, verbose=0)

    ism = np.zeros((L, 4, ref.shape[0]), dtype=np.float32)
    for (pos, base), e in zip(index, emb):
        ism[pos, base] = e - ref
    return ism


def project(ism: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Project the bottleneck ISM tensor onto a cell-group weight vector -> (L, 4)."""
    return ism @ weight


def contribution(track: np.ndarray, onehot: np.ndarray,
                 tissue_mean: np.ndarray | None = None) -> np.ndarray:
    """Hypothetical contribution scores: per-position mean-centre across the four
    nucleotides, optionally subtract a tissue-mean track, then reference-project."""
    track = track - track.mean(axis=1, keepdims=True)
    if tissue_mean is not None:
        track = track - tissue_mean
    return (track * onehot).sum(axis=1)                             # reference base
