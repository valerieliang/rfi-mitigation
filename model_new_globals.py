"""
model_new_globals.py

Two-branch knee classifier -- the benchmark topology, with two extra SCALAR
diagonal statistics on the global branch.

  eigen branch   (cpi_size, 2)  [ev_db normalized by lambda_max, first diff]
  global branch  (5,)           [cond_db, eff_rank, diag_median_max_ratio,
                                 schur_horn_gap, participation_ratio]

Relation to the other two models
--------------------------------
model.py          eigen branch + 3 globals                 (urClean benchmark)
model_diag.py     eigen branch + DIAG CONV BRANCH + 3 globals
model_new_globals eigen branch + 5 globals                 (this file)

This is deliberately a step BACK from model_diag.py, not forward from it. The
diagonal profile is kept out of the graph entirely and only two scale-free
summaries of it reach the network.

Why drop the diagonal conv branch
---------------------------------
generate_amazon_data.py places each RFI band in its own pulse row, drawn without
replacement, so in the training set `label == number of elevated diagonal rows`
exactly and with no noise. That is very close to a label leak, and it is a
property of the INJECTION MODEL rather than of physics: real RFI persists across
pulses and does not respect it. Measured, the shift is large -- the effective
number of occupied rows at one RFI eigenvalue is about 1.8 on synthetic tiles
and about 9.9 on real scenes.

model_diag.py devotes roughly 139k parameters (plus about 16k from the wider
fusion Dense) to that leak, against roughly 553k for the eigenvalue branch. The
two scalars here add 128. The leak is unchanged; the model's ability to carve it
out at high resolution is not.

Why the eigenvalue branch is the trustworthy one
------------------------------------------------
Eigenvalues are invariant under unitary rotation of pulse space: a rank-k RFI
perturbation produces the same k eigenvalues whether the emitter sits in one
pulse row or spreads coherently across all sixteen. The eigen branch is
therefore immune BY CONSTRUCTION to the artifact that breaks the diagonal, which
is why the eigen-only benchmark transfers to real scenes at all. Nothing here
changes it -- build_model is imported, not reimplemented, so the encoder is
byte-for-byte the benchmark's and any difference is attributable to the two
added scalars.

model.py already treats n_global_features as a pure shape parameter ("this
module makes no assumption about their identity, only their count"), so this
module is thin on purpose. It exists to pin the feature count and to give
train / test / score one import so they cannot disagree about the vector's
length or order.
"""

from model import build_model
from diag_features import N_GLOBAL_NEW


def build_model_new_globals(
    cpi_size=12,
    n_global_features=N_GLOBAL_NEW,
    n_knee_classes=7,
    dropout_rate=0.6,
    learning_rate=3e-4,
    weight_decay=1e-4,
):
    """
    Build the two-branch classifier with the 5-scalar global vector.

    Inputs (ORDER MATTERS for fit/predict -- [eigen, global]):
      eigen_input  : (cpi_size, 2)
      global_input : (n_global_features,), assembled by
                     diag_features.new_global_features

    Everything except the global vector length matches model.build_model, and
    the defaults here match the urClean benchmark's training_summary.json
    (dropout 0.6, lr 3e-4, weight decay 1e-4) so the comparison isolates the
    feature change.
    """
    if n_global_features != N_GLOBAL_NEW:
        raise ValueError(
            f'model_new_globals expects {N_GLOBAL_NEW} global features '
            f'(got {n_global_features}); build the vector with '
            f'diag_features.new_global_features')

    return build_model(
        cpi_size=cpi_size,
        n_global_features=n_global_features,
        n_knee_classes=n_knee_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
