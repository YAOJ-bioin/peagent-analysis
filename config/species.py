"""Per-species configuration and shared model constants.

Only biological constants (chromosome patterns, metadata columns, atlas sizes)
are hard-coded. File paths resolve under $PEAGENT_DATA_ROOT so the pipeline can
be pointed at any local copy of the three atlases without editing code.

Layout expected under the data root, per species <s>:
    <s>/atlas.h5ad        cell x ACR binary accessibility + cell metadata
    <s>/model.h5          trained model weights
    <s>/peaks.bed         ACR peak intervals
    <s>/genome.fa[.gz]    model-matched reference assembly (+ .fai)
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("PEAGENT_DATA_ROOT", "data"))

# Shared constants
INPUT_LEN = 1344
BOTTLENECK_SIZE = 64
STRIDES = (500, 1000)
RANDOM_SEED = 2026
MIN_CELLS_PER_GROUP = 50


def _paths(name):
    d = DATA_ROOT / name
    return {
        "atlas_h5ad": d / "atlas.h5ad",
        "model": d / "model.h5",
        "peaks_bed": d / "peaks.bed",
        "genome_fasta": d / "genome.fa",
    }


SPECIES = {
    "soybean": {
        **_paths("soybean"),
        "assembly": "Wm82.a4.v1",
        "chrom_regex": r"^Gm\d+$",
        "tissue_col": "Tissue", "celltype_col": "Celltype", "rep_col": "Replicate",
        "annotation_col": None, "annotation_sep": None,
        "n_cells": 200_732, "n_acrs": 303_113, "n_tissues": 10,
    },
    "maize": {
        **_paths("maize"),
        "assembly": "B73-NAM-5.0",
        "chrom_regex": r"^chr\d+$",
        "tissue_col": "tissue", "celltype_col": "celltype", "rep_col": "library",
        "annotation_col": None, "annotation_sep": None,
        "n_cells": 50_639, "n_acrs": 158_586, "n_tissues": 6,
    },
    "rice": {
        **_paths("rice"),
        "assembly": "Nipponbare-MSU7",
        "chrom_regex": r"^Chr\d+$",
        "tissue_col": "tissue", "celltype_col": "celltype", "rep_col": "library",
        "annotation_col": "Final_annotation_TCP_up", "annotation_sep": ".",
        "n_cells": 104_029, "n_acrs": 128_764, "n_tissues": 9,
    },
}
