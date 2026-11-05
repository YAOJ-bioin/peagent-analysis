"""PEAgent model: a sequence-to-single-cell chromatin-accessibility CNN.

The network maps a 1,344 bp one-hot DNA window to the accessibility state of
every single cell in an atlas. All per-cell predictions share a low-dimensional
(64-unit) sequence bottleneck, so the bottleneck learns a compact, cell-type
independent representation of regulatory sequence while the final dense layer
stores per-cell decoding weights. The convolutional design follows the
Basset/Basenji lineage (see README references).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class StochasticReverseComplement(layers.Layer):
    """Randomly reverse-complement the sequence during training (augmentation)."""

    def call(self, x, training=None):
        if training:
            rc = tf.gather(tf.reverse(x, axis=[1]), [3, 2, 1, 0], axis=2)
            flip = tf.random.uniform(shape=[]) > 0.5
            return tf.cond(flip, lambda: rc, lambda: x), flip
        return x, tf.constant(False)


class StochasticShift(layers.Layer):
    """Randomly shift the sequence by +/- `shift` bp during training."""

    def __init__(self, shift=3, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift

    def call(self, x, training=None):
        if not training:
            return x
        pad = tf.pad(x, [[0, 0], [self.shift, self.shift], [0, 0]])
        offset = tf.random.uniform([], -self.shift, self.shift + 1, dtype=tf.int32)
        return tf.slice(pad, [0, self.shift + offset, 0], tf.shape(x))


def conv_block(x, filters, kernel_size, pool_size=1, dropout=0.0):
    x = layers.Activation(tf.nn.gelu)(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(momentum=0.90)(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    if pool_size > 1:
        x = layers.MaxPool1D(pool_size, padding="same")(x)
    return x


def conv_tower(x, filters_init, filters_mult, repeat, kernel_size, pool_size):
    filters = filters_init
    for _ in range(repeat):
        x = conv_block(x, int(round(filters)), kernel_size, pool_size)
        filters *= filters_mult
    return x


def build_model(n_cells, bottleneck=64, seq_len=1344):
    """Build the PEAgent multitask accessibility model.

    n_cells    number of single-cell output tasks (atlas size)
    bottleneck width of the shared sequence embedding
    """
    seq = tf.keras.Input(shape=(seq_len, 4), name="sequence")
    x, _ = StochasticReverseComplement()(seq)
    x = StochasticShift(3)(x)

    x = conv_block(x, filters=288, kernel_size=17, pool_size=3)          # stem
    x = conv_tower(x, 288, filters_mult=1.122, repeat=6,                 # tower
                   kernel_size=5, pool_size=2)
    x = conv_block(x, filters=256, kernel_size=1)                        # pointwise

    x = layers.Activation(tf.nn.gelu)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(bottleneck, use_bias=False,
                     kernel_initializer="he_normal", name="bottleneck")(x)
    x = layers.BatchNormalization(momentum=0.90)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Activation(tf.nn.gelu)(x)

    out = layers.Dense(n_cells, activation="sigmoid", name="cell_head")(x)
    return tf.keras.Model(seq, out, name="peagent")


def bottleneck_model(model):
    """View that returns the 64-d bottleneck embedding for a sequence."""
    return tf.keras.Model(model.input, model.get_layer("bottleneck").output)


def cell_head_weights(model):
    """(W, b) mapping the bottleneck to per-cell logits (final dense layer)."""
    W, b = model.get_layer("cell_head").get_weights()
    return W, b
