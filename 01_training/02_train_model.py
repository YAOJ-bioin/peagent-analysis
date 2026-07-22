#!/usr/bin/env python
"""Train the PEAgent accessibility model for one species.

Streams (sequence, per-cell accessibility) pairs from the preprocessed tensors,
trains with Adam + binary cross-entropy, monitors validation AUROC, and keeps
the best checkpoint (early stopping, patience 50).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.species import BOTTLENECK_SIZE, INPUT_LEN
from peagent.model import build_model


def dataset(seq_h5, m_npz, batch, shuffle=False):
    X = h5py.File(seq_h5, "r")["X"]
    Y = sparse.load_npz(m_npz)              # ACR x cell

    def gen():
        for i in range(X.shape[0]):
            yield X[i].astype(np.int8), np.asarray(Y[i].todense()).ravel().astype(np.int8)

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec((INPUT_LEN, 4), tf.int8),
            tf.TensorSpec((Y.shape[1],), tf.int8)))
    if shuffle:
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE), Y.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="preprocess output dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bottleneck", type=int, default=BOTTLENECK_SIZE)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds, n_cells = dataset(args.data / "train_seqs.h5", args.data / "m_train.npz",
                                args.batch_size, shuffle=True)
    val_ds, _ = dataset(args.data / "val_seqs.h5", args.data / "m_val.npz", args.batch_size)

    model = build_model(n_cells, bottleneck=args.bottleneck, seq_len=INPUT_LEN)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(args.lr, beta_1=0.95, beta_2=0.9995),
        metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auc"),
                 tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auc_pr")])

    ckpt = str(args.out / "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max",
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=args.patience, restore_best_weights=True),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    pickle.dump(hist.history, open(args.out / "history.pkl", "wb"))
    print(f"best val AUROC: {max(hist.history['val_auc']):.4f}  -> {ckpt}")


if __name__ == "__main__":
    main()
