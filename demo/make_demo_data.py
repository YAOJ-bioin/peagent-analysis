#!/usr/bin/env python
"""Simulate the small demo atlas in demo/data/demo/.

A random 2-Mb genome carries 1,200 ACRs. Most ACRs hold three copies of one of four
motifs, and each motif opens chromatin in one of four tissue x cell-type groups
(60 cells each); the rest are closed background. Same layout as a real species
under $PEAGENT_DATA_ROOT: atlas.h5ad, peaks.bed, genome.fa (+ .fai).
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pysam
from scipy import sparse

OUT = Path(__file__).resolve().parent / "data" / "demo"
SEED = 2026
GENOME_LEN = 2_000_000
N_ACRS = 1_200
ACR_LEN = 500
CELLS_PER_GROUP = 60
GROUPS = [("root", "epidermis"), ("root", "cortex"), ("leaf", "mesophyll"), ("leaf", "guard_cell")]
MOTIFS = ["GGGACCAC", "CCACGTGG", "AGATAAGA", "TGACGTCA"]   # TCP, G-box, GATA, bZIP-like
P_OPEN, P_BACKGROUND = 0.4, 0.01


def main():
    rng = np.random.default_rng(SEED)
    genome = rng.choice(list("ACGT"), size=GENOME_LEN)

    spacing = GENOME_LEN // N_ACRS
    starts = np.arange(N_ACRS) * spacing + (spacing - ACR_LEN) // 2
    label = rng.integers(-1, len(MOTIFS), size=N_ACRS)      # -1 = no motif (closed background)
    for s, k in zip(starts, label):
        if k < 0:
            continue
        for off in rng.choice(np.arange(20, ACR_LEN - 20, 25), size=3, replace=False):
            genome[s + off:s + off + len(MOTIFS[k])] = list(MOTIFS[k])

    obs = pd.DataFrame(
        [(t, c, "rep%d" % (i % 2 + 1)) for t, c in GROUPS for i in range(CELLS_PER_GROUP)],
        columns=["tissue", "celltype", "replicate"])
    obs.index = ["cell%04d" % i for i in range(len(obs))]
    group_of_cell = np.repeat(np.arange(len(GROUPS)), CELLS_PER_GROUP)
    p = np.where(group_of_cell[:, None] == label[None, :], P_OPEN, P_BACKGROUND)
    X = sparse.csr_matrix((rng.random(p.shape) < p).astype(np.float32))
    var = pd.DataFrame(index=["chr1_%d_%d" % (s, s + ACR_LEN) for s in starts])
    var["motif_group"] = label

    OUT.mkdir(parents=True, exist_ok=True)
    seq = "".join(genome)
    with open(OUT / "genome.fa", "w") as fh:
        fh.write(">chr1\n" + "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80)) + "\n")
    pysam.faidx(str(OUT / "genome.fa"))
    with open(OUT / "peaks.bed", "w") as fh:
        for s in starts:
            fh.write("chr1\t%d\t%d\n" % (s, s + ACR_LEN))
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(OUT / "atlas.h5ad", compression="gzip")
    print("demo atlas: %d cells x %d ACRs, %.1f%% non-zero -> %s" % (X.shape[0], X.shape[1], 100 * X.nnz / np.prod(X.shape), OUT))


if __name__ == "__main__":
    main()
