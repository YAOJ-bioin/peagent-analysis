"""Pseudobulk cell-group construction and model prediction helpers."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import build_model


def load_model(path, n_cells, bottleneck=64, seq_len=1344):
    model = build_model(n_cells, bottleneck=bottleneck, seq_len=seq_len)
    model.load_weights(str(path))
    return model


def predict(model, seqs, batch=64):
    """Predicted per-cell accessibility for a stack of one-hot windows."""
    return model.predict(np.asarray(seqs, np.float32), batch_size=batch, verbose=0)


def cell_groups(atlas, cfg, min_cells=50):
    """Map cells to tissue x cell-type groups; keep groups with >= min_cells.

    Returns {group_name: array_of_cell_indices}.
    """
    obs = atlas.obs
    key = obs[cfg["tissue_col"]].astype(str) + "." + obs[cfg["celltype_col"]].astype(str)
    groups = {}
    for name, idx in key.groupby(key).groups.items():
        pos = np.where(key.values == name)[0]
        if len(pos) >= min_cells:
            groups[name] = pos
    return groups


def pseudobulk_truth(atlas, groups, open_frac=0.05):
    """Observed group x ACR accessibility (open fraction, and binary state)."""
    X = atlas.X.tocsc()
    frac = np.vstack([np.asarray(X[idx].mean(axis=0)).ravel() for idx in groups.values()])
    return frac, (frac >= open_frac).astype(np.int8)


def group_pred(pred_cells, groups):
    """Predicted group x ACR accessibility = mean per-cell prediction in each group."""
    return np.vstack([pred_cells[:, idx].mean(axis=1) for idx in groups.values()])
