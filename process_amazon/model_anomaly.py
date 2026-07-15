"""
model_anomaly.py

CNN autoencoder for one-class anomaly detection on per-CPI eigenvalue spectra.

Why an autoencoder instead of a classifier
-------------------------------------------
The previous knee-index classifier is a softmax model: it always assigns
maximum probability to its nearest known class, so it cannot flag an input
as unfamiliar. That made it useless for catching real-world RFI signatures
that do not resemble the synthetic training distribution (sim-to-real
transfer failure).

An autoencoder trained only on clean CPI tiles learns to reconstruct the
SHAPE of a clean eigenvalue spectrum. RFI-contaminated or otherwise unusual
CPIs will not compress/reconstruct well through that learned representation,
so reconstruction error becomes a natural, un-bounded anomaly score rather
than a forced choice between known classes.

Architecture notes
-------------------
- Encoder uses strided Conv1D (not pooling) to downsample the eigenvalue
  sequence, followed by Flatten (NOT GlobalAveragePooling) at the bottleneck.
  GlobalAveragePooling discards positional information -- exactly the
  failure mode identified in the previous classifier -- so it is avoided
  here as well.
- Global scalar features (condition number, effective rank) are concatenated
  into the bottleneck alongside the flattened eigen encoding, and the model
  also reconstructs them, so anomalies visible only in the global features
  contribute to the anomaly score too.
- Expects cpi_size = 12 (the truncated eigenvalue count), matching
  anomaly_features.N_KEEP_DEFAULT. Sequence length must be divisible by 4
  for the two stride-2 downsampling steps used below (12 -> 6 -> 3).

Inputs
------
eigen_input  : (cpi_size, 2) -- [eigvals_db, slope_db], both already
               z-score normalized by the caller (see train_anomaly.py).
global_input : (n_global_features,) -- [condition_number_db, eff_rank],
               also z-score normalized by the caller.

Outputs
-------
eigen_recon  : (cpi_size, 2) reconstruction of eigen_input
global_recon : (n_global_features,) reconstruction of global_input
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_anomaly_autoencoder(
    cpi_size: int = 12,
    n_global_features: int = 2,
    latent_dim: int = 8,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-3,
    eigen_loss_weight: float = 1.0,
    global_loss_weight: float = 0.5,
) -> Model:
    """
    Build and compile the CNN autoencoder for anomaly detection.

    Parameters
    ----------
    cpi_size : int, default 12
        Number of leading eigenvalues used as features (must be divisible
        by 4 for this architecture's two stride-2 downsampling steps).
    n_global_features : int, default 2
        Number of scalar global features (condition_number_db, eff_rank).
    latent_dim : int, default 8
        Size of the bottleneck latent vector.
    dropout_rate : float, default 0.1
        Dropout applied after each conv block in the encoder.
    learning_rate : float, default 1e-3
        Adam optimizer learning rate.
    eigen_loss_weight : float, default 1.0
        Loss weight for the eigen_recon reconstruction term.
    global_loss_weight : float, default 0.5
        Loss weight for the global_recon reconstruction term.

    Returns
    -------
    model : tf.keras.Model
        Compiled model with inputs [eigen_input, global_input] and
        outputs [eigen_recon, global_recon].
    """
    if cpi_size % 4 != 0:
        raise ValueError(
            f"cpi_size ({cpi_size}) must be divisible by 4 for the two "
            f"stride-2 downsampling steps used in this architecture."
        )

    eigen_input = layers.Input(shape=(cpi_size, 2), name='eigen_input')
    global_input = layers.Input(shape=(n_global_features,), name='global_input')

    # ---------------- Encoder ----------------
    x = layers.Conv1D(16, 3, padding='same', activation='relu')(eigen_input)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(32, 3, strides=2, padding='same', activation='relu')(x)  # cpi_size -> cpi_size/2
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(64, 3, strides=2, padding='same', activation='relu')(x)  # cpi_size/2 -> cpi_size/4
    x = layers.Dropout(dropout_rate)(x)

    # Flatten preserves positional structure of the downsampled sequence;
    # GlobalAveragePooling is deliberately not used here.
    x = layers.Flatten(name='encoder_flatten')(x)
    x = layers.Concatenate(name='encoder_concat_global')([x, global_input])

    latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)

    # ---------------- Decoder ----------------
    bottleneck_len = cpi_size // 4

    d = layers.Dense(bottleneck_len * 64, activation='relu')(latent)
    d = layers.Reshape((bottleneck_len, 64))(d)

    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D(2)(d)  # cpi_size/4 -> cpi_size/2

    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D(2)(d)  # cpi_size/2 -> cpi_size

    d = layers.Conv1D(16, 3, padding='same', activation='relu')(d)
    eigen_recon = layers.Conv1D(2, 3, padding='same', activation='linear', name='eigen_recon')(d)

    g = layers.Dense(16, activation='relu')(latent)
    global_recon = layers.Dense(n_global_features, activation='linear', name='global_recon')(g)

    model = Model(
        inputs=[eigen_input, global_input],
        outputs=[eigen_recon, global_recon],
        name='rfi_anomaly_autoencoder',
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'eigen_recon': 'mse', 'global_recon': 'mse'},
        loss_weights={'eigen_recon': eigen_loss_weight, 'global_recon': global_loss_weight},
    )

    return model


def compute_anomaly_scores(
    model: Model,
    eigen: "tf.Tensor | None" = None,
    global_: "tf.Tensor | None" = None,
    eigen_weight: float = 1.0,
    global_weight: float = 0.5,
    batch_size: int = 256,
):
    """
    Compute per-sample reconstruction-error anomaly scores.

    Parameters
    ----------
    model : tf.keras.Model
        Trained autoencoder from build_anomaly_autoencoder.
    eigen : (N, cpi_size, 2) array
        Normalized eigen features (same normalization used at train time).
    global_ : (N, n_global_features) array
        Normalized global features (same normalization used at train time).
    eigen_weight, global_weight : float
        Weights combining the two reconstruction error terms into a single
        total anomaly score. Should generally match the model's loss_weights.
    batch_size : int, default 256
        Prediction batch size.

    Returns
    -------
    eigen_err : (N,) float32 array
        Per-sample mean squared reconstruction error on eigen features.
    global_err : (N,) float32 array
        Per-sample mean squared reconstruction error on global features.
    total_score : (N,) float32 array
        eigen_weight * eigen_err + global_weight * global_err.
    """
    import numpy as np

    eigen_recon, global_recon = model.predict([eigen, global_], batch_size=batch_size, verbose=0)

    eigen_err = np.mean((eigen_recon - eigen) ** 2, axis=(1, 2)).astype(np.float32)
    global_err = np.mean((global_recon - global_) ** 2, axis=1).astype(np.float32)
    total_score = (eigen_weight * eigen_err + global_weight * global_err).astype(np.float32)

    return eigen_err, global_err, total_score
