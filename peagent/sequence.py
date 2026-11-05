"""DNA sequence utilities: one-hot encoding and centred ACR window extraction."""
from __future__ import annotations

import numpy as np

BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def one_hot(seq: str, seq_len: int | None = None) -> np.ndarray:
    """One-hot encode a DNA string to (L, 4). Ambiguous bases map to all-zero."""
    if seq_len is not None:
        seq = seq[:seq_len].ljust(seq_len, "N")
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        j = BASES.get(base)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def center_window(fasta, chrom: str, start: int, end: int, seq_len: int = 1344) -> str:
    """Fetch a fixed-length window centred on a peak interval from a pysam FASTA."""
    center = (start + end) // 2
    left = center - seq_len // 2
    return fasta.fetch(chrom, max(left, 0), left + seq_len).upper()


def substitute(seq: str, pos: int, allele: str) -> str:
    """Return `seq` with the base at 0-based `pos` replaced by `allele`."""
    return seq[:pos] + allele + seq[pos + 1:]
