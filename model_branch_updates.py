"""
model_branch_updates.py

Single-branch knee classifier with combined EV, EV slopes, SCM diagonal, and
emphasis features.

Architecture:
  - Single input branch: (37,) feature vector containing:
      * 12 valid EVs (max-normalized, dB)
      * 11 EV slopes
      * 12 valid SCM diagonal entries (max-normalized, dB)
      * 2 emphasis features [max EV, max pulse power] (normalized)

The emphasis vector [max EV, max pulse power] reinforces the lossless
transformation relationship between the eigenvalue spectrum and the SCM diagonal,
even though these values are already present in the EV and diagonal components.

Key design choices:
  1. All features are pre-normalized in the feature extraction step, so the
     network receives scale-free inputs that emphasize structural relationships.
  2. The single branch processes all 37 features through a series of Dense layers
     with residual connections and dropout for regularization.
  3. Squeeze-and-Excitation style attention is adapted for the dense architecture
     to allow the network to weight different feature groups.

Relation to other models:
  model.py               - Two-branch (eigen + global scalars)
  model_diag.py          - Three-branch (eigen + diag conv + global)
  model_new_globals.py   - Two-branch (eigen + 5 global scalars)
  model_branch_updates   - Single-branch (combined features, THIS FILE)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, ReLU, Add, Dropout, Reshape, Multiply
)
from tensorflow.keras.models import Model


def attention_block(x, units, ratio=4):
    """
    Dense attention mechanism (analogous to Squeeze-and-Excitation for dense layers).

    Computes importance weights over the feature dimensions and re-weights them.
    """
    # Squeeze: global statistics (in dense layers, this is just the identity)
    squeeze = x

    # Excitation: two-layer MLP with bottleneck
    excite = Dense(max(units // ratio, 1), activation='relu')(squeeze)
    excite = Dense(units, activation='sigmoid')(excite)

    # Scale: element-wise multiplication
    return Multiply()([x, excite])


def dense_residual_block(x, units, use_attention=True, dropout_rate=0.3):
    """
    Residual block for dense layers with optional attention.

    Similar to res_block in model.py, but adapted for fully-connected layers.
    """
    shortcut = x

    # First dense -> BN -> ReLU
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)

    # Second dense -> BN (no activation before Add)
    x = Dense(units)(x)
    x = BatchNormalization()(x)

    # Optional attention
    if use_attention:
        x = attention_block(x, units)

    # Project shortcut to match dimensions if needed
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)

    x = Add()([shortcut, x])
    return ReLU()(x)


def build_model_branch_updates(
    n_branch_features=37,
    n_knee_classes=7,
    dropout_rate=0.6,
    learning_rate=3e-4,
    weight_decay=1e-4,
):
    """
    Build the single-branch combined-features knee classifier.

    Parameters
    ----------
    n_branch_features : int
        Total number of features in the combined branch (should be 37:
        12 EVs + 11 slopes + 12 diagonal + 2 emphasis).
    n_knee_classes : int
        Number of output classes for the knee index (typically 7 for 0..6 RFI eigs).
    dropout_rate : float
        Dropout rate in the dense head.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        L2 regularization strength (AdamW weight decay).

    Inputs
    ------
    branch1_input : (n_branch_features,)
        Combined feature vector with all 37 normalized features.

    Output
    ------
    Softmax over n_knee_classes indices.
    """
    # --- Combined branch input ---
    branch1_input = Input(shape=(n_branch_features,), name='branch1_input')

    # Initial dense layer to expand feature space
    x = Dense(128)(branch1_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    # Residual blocks with attention
    x = dense_residual_block(x, 256, use_attention=True, dropout_rate=0.3)
    x = dense_residual_block(x, 256, use_attention=True, dropout_rate=0.3)
    x = dense_residual_block(x, 128, use_attention=True, dropout_rate=0.4)

    # Classification head
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(n_knee_classes, activation='softmax', name='knee_softmax')(x)

    model = Model(inputs=branch1_input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
        ],
    )

    return model


def predict_knee_with_confidence(model, branch1_batch):
    """
    Convenience helper: returns (knee_index, confidence, entropy) per sample.

    confidence is max softmax probability; entropy is the Shannon entropy of
    the output distribution (low entropy = confident, high entropy = uncertain).
    """
    import numpy as np
    probs = model.predict(branch1_batch, verbose=0)
    knee_index = np.argmax(probs, axis=-1)
    confidence = np.max(probs, axis=-1)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    return knee_index, confidence, entropy
