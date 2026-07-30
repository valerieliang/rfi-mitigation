"""
model_diag.py

Three-branch variant of model.py's knee classifier: adds a dedicated branch for
the SORTED SCM DIAGONAL PROFILE alongside the eigenvalue branch.

Why a third branch and not extra channels on the eigenvalue branch
-----------------------------------------------------------------
The two profiles live in different spaces and have different lengths:

  eigenvalues  12 values, MODE space,  normalized by lambda_max
  diagonal     16 values, PULSE space, normalized by its own median

Stacking them as channels of one tensor would force index j of the eigenvalue
profile to align with index j of the diagonal profile. Those indices are not the
same physical quantity, and it would also mean padding the 12-long eigenvalue
profile out to 16 -- injecting a fake cliff at index 12 that the conv would
happily learn as structure. A separate stack per profile avoids both problems and
costs only a concatenate in the fusion head.

Why global average pooling is the right head for this
-----------------------------------------------------
generate_amazon_data.py places each RFI band in its OWN pulse row, drawn without
replacement, so `label == number of contaminated rows` exactly. The task is
therefore COUNTING elevated entries, not locating them. GlobalAveragePooling1D
over a channel that fires on "this entry is above the floor" returns count/length
-- precisely the quantity needed. Pooling discards position, which for a counting
task is a feature and not a loss.

The diagonal branch is deliberately smaller than the eigenvalue branch (64/128 vs
128/256 filters). A sorted diagonal profile is close to a step function, which is
a much simpler shape than the eigenvalue knee and does not need the same capacity.

model.py is left untouched so the urClean benchmark stays exactly reproducible;
res_block and SEBlock are imported from it rather than copied.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    GlobalAveragePooling1D, Dense, Dropout, Concatenate,
)
from tensorflow.keras.models import Model

from model import res_block


def _profile_branch(inputs, filters, name):
    """
    Conv stack over one sorted profile: stem -> res -> pool -> res -> GAP.
    Mirrors the eigenvalue branch of model.build_model, sized by `filters`.
    """
    f1, f2 = filters
    x = Conv1D(f1, 5, padding='same', name=f'{name}_stem')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = res_block(x, f1)
    x = MaxPooling1D(2)(x)

    x = res_block(x, f2)
    return GlobalAveragePooling1D(name=f'{name}_gap')(x)


def build_model_diag(
    cpi_size=12,
    diag_size=16,
    n_global_features=3,
    n_knee_classes=7,
    dropout_rate=0.6,
    learning_rate=3e-4,
    weight_decay=1e-4,
    eigen_filters=(128, 256),
    diag_filters=(64, 128),
):
    """
    Build the three-branch knee classifier.

    Inputs (ORDER MATTERS for fit/predict -- [eigen, diag, global]):
      eigen_input  : (cpi_size, 2)  [ev_db normalized by lambda_max, first diff]
      diag_input   : (diag_size, 2) [sorted diag in dB rel. to its own median,
                                     first diff]
      global_input : (n_global_features,)

    The eigenvalue branch is byte-for-byte the same topology as
    model.build_model's, so a difference in results against the benchmark is
    attributable to the added diagonal branch rather than to a changed encoder.

    Output: softmax over n_knee_classes.
    """
    eigen_inputs = Input(shape=(cpi_size, 2), name='eigen_input')
    x = _profile_branch(eigen_inputs, eigen_filters, 'eigen')

    diag_inputs = Input(shape=(diag_size, 2), name='diag_input')
    d = _profile_branch(diag_inputs, diag_filters, 'diag')

    global_inputs = Input(shape=(n_global_features,), name='global_input')
    y = Dense(64, activation='relu')(global_inputs)
    y = Dropout(0.3)(y)
    y = Dense(32, activation='relu')(y)

    combined = Concatenate()([x, d, y])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)

    outputs = Dense(n_knee_classes, activation='softmax', name='knee_softmax')(combined)

    model = Model(inputs=[eigen_inputs, diag_inputs, global_inputs], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
        ],
    )
    return model
