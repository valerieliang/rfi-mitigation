"""
CNN-based RFI / Signal Eigenvalue boundary (knee) estimator.

Two-branch architecture:
  - Eigenvalue branch: 1D CNN over the full eigenvalue + slope profile
  - Global branch: dense network over scalar covariance / threshold-block statistics
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    ReLU, Add, MaxPooling1D,
    GlobalAveragePooling1D, Dense, Dropout, Concatenate, Reshape, Multiply
)
from tensorflow.keras.models import Model


def SEBlock(channels, ratio=8):
    """Squeeze-and-Excitation block for per-channel attention."""
    def se_block(inputs):
        se = GlobalAveragePooling1D()(inputs)
        se = Dense(max(channels // ratio, 1), activation='relu')(se)
        se = Dense(channels, activation='sigmoid')(se)
        se = Reshape((1, channels))(se)
        return Multiply()([inputs, se])
    return se_block


def res_block(x, filters, se_ratio=8, use_se=True):
    """
    Pre-activation residual block with optional Squeeze-and-Excitation.

    The skip connection (Add) is REQUIRED for this to function as a residual
    block. Earlier versions of this code had the Add commented out, which
    silently turned the network into a plain stacked-conv network.
    """
    shortcut = x

    # First conv -> BN -> ReLU
    x = Conv1D(filters, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second conv -> BN (no activation before the Add)
    x = Conv1D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    # Optional channel attention
    if use_se:
        x = SEBlock(filters, ratio=se_ratio)(x)

    # Project shortcut to match channel count if needed, then add
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Add()([shortcut, x])

    return ReLU()(x)


def build_model(
    cpi_size=32,
    n_global_features=6,
    n_knee_classes=None,
    dropout_rate=0.5,
    learning_rate=3e-4,
):
    """
    Build the two-branch knee-index classifier.

    Parameters
    ----------
    cpi_size : int
        Number of pulses per CPI (M). Eigenvalue profile length.
    n_global_features : int
        Number of scalar context features per CPI. Suggested set:
        [F_factor, sigma_min, sigma_max, mu_min, trace_db, condition_number].
    n_knee_classes : int or None
        Number of output classes for the knee index. Defaults to cpi_size + 1
        so that index 0 means "no RFI present" and indices 1..M mean
        "boundary is after eigenvalue i".
    dropout_rate : float
        Dropout in the dense head.
    learning_rate : float
        Adam learning rate.

    Inputs
    ------
    eigen_input  : (cpi_size, 2)  channels = [eigenvalues_dB, slopes_dB_padded]
    global_input : (n_global_features,)

    Output
    ------
    softmax over n_knee_classes indices.
    """
    if n_knee_classes is None:
        n_knee_classes = cpi_size + 1

    # --- Eigenvalue / slope branch -----------------------------------------
    eigen_inputs = Input(shape=(cpi_size, 2), name='eigen_input')

    x = Conv1D(128, 5, padding='same')(eigen_inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = res_block(x, 128)
    x = MaxPooling1D(2)(x)

    x = res_block(x, 256)
    x = GlobalAveragePooling1D()(x)

    # --- Global / context branch -------------------------------------------
    global_inputs = Input(shape=(n_global_features,), name='global_input')
    y = Dense(64, activation='relu')(global_inputs)
    y = Dropout(0.3)(y)
    y = Dense(32, activation='relu')(y)

    # --- Fusion + head -----------------------------------------------------
    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)

    outputs = Dense(n_knee_classes, activation='softmax', name='knee_softmax')(combined)

    model = Model(inputs=[eigen_inputs, global_inputs], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
        ],
    )
    return model


def predict_knee_with_confidence(model, eigen_batch, global_batch):
    """
    Convenience helper: returns (knee_index, confidence, entropy) per sample.

    confidence is max softmax probability; entropy is the Shannon entropy of
    the output distribution (low entropy = confident, high entropy = uncertain).
    Use entropy as a fallback signal to defer to the ST-EST baseline on
    ambiguous CPIs.
    """
    import numpy as np
    probs = model.predict([eigen_batch, global_batch], verbose=0)
    knee_index = np.argmax(probs, axis=-1)
    confidence = np.max(probs, axis=-1)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    return knee_index, confidence, entropy