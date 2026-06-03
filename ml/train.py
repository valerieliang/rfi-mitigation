#!/usr/bin/env python3
"""
ml/train.py

Train the knee-index classifier defined in ml/model.py.

Supports two input modes:
  Single-file (legacy):
    --data training_set.npz
    The npz must contain a 'split' column (0=train, 1=val).

  Separate files (from build_dataset.py):
    --data data/model/train.npz --val-data data/model/val.npz
    All samples in --data are used for training; val comes from --val-data.
    Add --test-data for a final held-out evaluation after training.

Outputs in --out-dir:
    best_model.keras          best val-loss checkpoint
    final_model.keras         weights at end of training
    training_history.csv      per-epoch metrics
    training_curves.png       loss / accuracy curves
    confusion_matrix.png      binned confusion matrix on val set
    evaluation_report.txt     numeric summary (test appended if given)
"""

import argparse
import os
import sys
import textwrap

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, predict_knee_with_confidence


# ---------------------------------------------------------------------------
# Tolerance-accuracy metric (handles both sparse and one-hot targets)
# ---------------------------------------------------------------------------
class ToleranceAccuracy(tf.keras.metrics.Metric):
    """Fraction of predictions within +/-tolerance of the true label."""

    def __init__(self, tolerance=1, name="tol1_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        yt   = tf.convert_to_tensor(y_true)
        # Accept one-hot (label smoothing) or sparse integer labels.
        if (yt.shape.rank is not None and yt.shape.rank >= 2
                and yt.shape[-1] is not None and yt.shape[-1] > 1):
            true = tf.cast(tf.argmax(yt, axis=-1), tf.int32)
        else:
            true = tf.cast(tf.reshape(yt, [-1]), tf.int32)
        within = tf.abs(pred - true) <= self.tolerance
        if sample_weight is not None:
            w = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            self.correct.assign_add(
                tf.reduce_sum(tf.cast(within, tf.float32) * w))
            self.total.assign_add(tf.reduce_sum(w))
        else:
            self.correct.assign_add(
                tf.reduce_sum(tf.cast(within, tf.float32)))
            self.total.assign_add(tf.cast(tf.size(true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "tolerance": self.tolerance}


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(labels, n_classes, smoothing=0.1):
    """
    Inverse-frequency weights with additive smoothing.
    w_c = N / (n_classes * count_c); smoothing prevents extreme weights
    for near-empty classes.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    N = float(len(labels))
    if smoothing > 0:
        counts = counts + smoothing * (N / n_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(counts > 0, N / (n_classes * counts), 0.0)
    return w


# ---------------------------------------------------------------------------
# Binned confusion matrix
# ---------------------------------------------------------------------------
BINS = [
    (0,  0,  "0\n(clean)"),
    (1,  2,  "1-2"),
    (3,  5,  "3-5"),
    (6,  10, "6-10"),
    (11, 20, "11-20"),
    (21, 32, "21-32"),
]


def bin_label(y, bins):
    out = np.empty(len(y), dtype=np.int32)
    for i, (lo, hi, _) in enumerate(bins):
        out[(y >= lo) & (y <= hi)] = i
    return out


def binned_confusion(y_true, y_pred, bins):
    b_true = bin_label(y_true, bins)
    b_pred = bin_label(y_pred, bins)
    n  = len(bins)
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(b_true, b_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, bins, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels   = [b[2] for b in bins]
    n        = len(labels)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm  = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="fraction of true class")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted knee bin")
    ax.set_ylabel("true knee bin")
    ax.set_title("validation confusion matrix (row-normalized)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i,
                    "{:.0f}%\n({})".format(100 * cm_norm[i, j], cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_curves(hist_path, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    epochs, tr_loss, val_loss, tr_acc, val_acc, tr_t1, val_t1 = (
        [], [], [], [], [], [], [])
    with open(hist_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
            tr_acc.append(float(row["acc"]))
            val_acc.append(float(row["val_acc"]))
            tr_t1.append(float(row.get("tol1_acc",     "nan")))
            val_t1.append(float(row.get("val_tol1_acc","nan")))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, tr_loss,  label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, tr_acc,  label="train exact")
    axes[1].plot(epochs, val_acc, label="val exact")
    axes[1].set_title("exact accuracy"); axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1); axes[1].legend()

    axes[2].plot(epochs, tr_t1,  label="train tol-1")
    axes[2].plot(epochs, val_t1, label="val tol-1")
    axes[2].set_title("tolerance-1 accuracy (+/-1 eigenvalue)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1); axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation helper (shared by val and test)
# ---------------------------------------------------------------------------
def evaluate(model, eigen, glob, y_true):
    """Return a dict of metrics for one evaluation set."""
    knee_pred, confidence, entropy = predict_knee_with_confidence(
        model, eigen, glob)

    exact = float(np.mean(knee_pred == y_true))
    tol1  = float(np.mean(np.abs(knee_pred - y_true) <= 1))
    tol2  = float(np.mean(np.abs(knee_pred - y_true) <= 2))
    mae   = float(np.mean(np.abs(knee_pred.astype(float)
                                 - y_true.astype(float))))

    rfi_true = (y_true > 0).astype(int)
    rfi_pred = (knee_pred > 0).astype(int)
    tp = int(np.sum((rfi_true == 1) & (rfi_pred == 1)))
    tn = int(np.sum((rfi_true == 0) & (rfi_pred == 0)))
    fp = int(np.sum((rfi_true == 0) & (rfi_pred == 1)))
    fn = int(np.sum((rfi_true == 1) & (rfi_pred == 0)))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    high_conf = confidence >= 0.6
    hi_acc = (float(np.mean(knee_pred[high_conf] == y_true[high_conf]))
              if high_conf.any() else float("nan"))
    lo_acc = (float(np.mean(knee_pred[~high_conf] == y_true[~high_conf]))
              if (~high_conf).any() else float("nan"))

    return dict(knee_pred=knee_pred, confidence=confidence, entropy=entropy,
                exact=exact, tol1=tol1, tol2=tol2, mae=mae,
                tp=tp, tn=tn, fp=fp, fn=fn,
                prec=prec, rec=rec, f1=f1,
                high_conf=high_conf, hi_acc=hi_acc, lo_acc=lo_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train the knee-index CNN classifier.")
    ap.add_argument("--data", default="training_set.npz",
                    help="Training .npz. Without --val-data must contain a "
                         "'split' column (0=train, 1=val).")
    ap.add_argument("--val-data", default=None, metavar="val.npz",
                    help="Separate val .npz (from build_dataset.py). All "
                         "samples in --data are then used for training.")
    ap.add_argument("--test-data", default=None, metavar="test.npz",
                    help="Held-out test .npz. Evaluated after training; "
                         "results appended to evaluation_report.txt.")
    ap.add_argument("--out-dir", default="ml/checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience in epochs.")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="ReduceLROnPlateau patience in epochs.")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable inverse-frequency class weighting.")
    ap.add_argument("--weight-smoothing", type=float, default=0.1,
                    help="Additive smoothing for class-weight computation.")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Label smoothing (0 disables). Prevents the model "
                         "from growing over-confident on wrong answers. "
                         "Enables categorical cross-entropy + one-hot "
                         "targets; class weighting is then dropped.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay. 0 = plain Adam. "
                         "Try 1e-4 to reduce overfitting.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not os.path.exists(args.data):
        sys.exit("Training set not found: {}".format(args.data))

    d = np.load(args.data, allow_pickle=True)
    M         = int(d["M"])
    n_classes = int(d["n_knee_classes"])
    n_feats   = int(d["global_input"].shape[1])

    if args.val_data:
        if not os.path.exists(args.val_data):
            sys.exit("Val set not found: {}".format(args.val_data))
        dv       = np.load(args.val_data, allow_pickle=True)
        eigen_tr = d["eigen_input"].astype(np.float32)
        glob_tr  = d["global_input"].astype(np.float32)
        y_tr     = d["labels"].astype(np.int32)
        eigen_va = dv["eigen_input"].astype(np.float32)
        glob_va  = dv["global_input"].astype(np.float32)
        y_va     = dv["labels"].astype(np.int32)
        print("Train: {}  ({} samples)".format(args.data, len(y_tr)))
        print("Val:   {}  ({} samples)".format(args.val_data, len(y_va)))
    else:
        eigen  = d["eigen_input"].astype(np.float32)
        glob   = d["global_input"].astype(np.float32)
        labels = d["labels"].astype(np.int32)
        split  = d["split"]
        tr = split == 0; va = split == 1
        eigen_tr, eigen_va = eigen[tr], eigen[va]
        glob_tr,  glob_va  = glob[tr],  glob[va]
        y_tr, y_va         = labels[tr], labels[va]
        print("Loaded {}  ({} train / {} val)".format(
            args.data, tr.sum(), va.sum()))

    print("M={}, n_classes={}, n_features={}".format(M, n_classes, n_feats))

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    if args.no_class_weights:
        class_weight = None
        print("Class weighting: DISABLED")
    else:
        w = compute_class_weights(y_tr, n_classes, args.weight_smoothing)
        class_weight = {i: float(w[i]) for i in range(n_classes)}
        top5 = sorted(class_weight.items(), key=lambda x: -x[1])[:5]
        print("Class weighting: ON  top-5: {}".format(
            [(c, round(v, 1)) for c, v in top5]))

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(
        cpi_size=M,
        n_global_features=n_feats,
        n_knee_classes=n_classes,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )

    optimizer = (tf.keras.optimizers.AdamW(
                     learning_rate=args.lr, weight_decay=args.weight_decay)
                 if args.weight_decay > 0
                 else tf.keras.optimizers.Adam(args.lr))

    use_smoothing = args.label_smoothing > 0
    if use_smoothing:
        loss       = tf.keras.losses.CategoricalCrossentropy(
                         label_smoothing=args.label_smoothing)
        acc_metric = tf.keras.metrics.CategoricalAccuracy(name="acc")
        top2       = tf.keras.metrics.TopKCategoricalAccuracy(
                         k=2, name="top2_acc")
        y_tr_fit   = tf.one_hot(y_tr, n_classes)
        y_va_fit   = tf.one_hot(y_va, n_classes)
    else:
        loss       = tf.keras.losses.SparseCategoricalCrossentropy()
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        top2       = tf.keras.metrics.SparseTopKCategoricalAccuracy(
                         k=2, name="top2_acc")
        y_tr_fit   = y_tr
        y_va_fit   = y_va

    model.compile(
        optimizer=optimizer, loss=loss,
        metrics=[acc_metric, top2,
                 ToleranceAccuracy(tolerance=1, name="tol1_acc")])
    model.summary(print_fn=print)

    # Keras rejects class_weight with one-hot targets.
    if use_smoothing and class_weight is not None:
        print("Note: label smoothing ON -> class weighting dropped "
              "(incompatible with one-hot targets; not needed on a "
              "balanced set anyway).")
        class_weight = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    best_path = os.path.join(args.out_dir, "best_model.keras")
    hist_path = os.path.join(args.out_dir, "training_history.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_loss",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=args.lr_patience, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(hist_path),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nTraining up to {} epochs (patience={})...".format(
        args.epochs, args.patience))
    history = model.fit(
        [eigen_tr, glob_tr], y_tr_fit,
        validation_data=([eigen_va, glob_va], y_va_fit),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    final_path = os.path.join(args.out_dir, "final_model.keras")
    model.save(final_path)
    print("Saved final model:", final_path)

    # ------------------------------------------------------------------
    # Val evaluation
    # ------------------------------------------------------------------
    best_model = tf.keras.models.load_model(
        best_path,
        custom_objects={"ToleranceAccuracy": ToleranceAccuracy})

    vm = evaluate(best_model, eigen_va, glob_va, y_va)

    report = textwrap.dedent("""
    ========================================================
    KNEE CLASSIFIER  --  EVALUATION REPORT
    ========================================================
    Train data:    {data}
    Val data:      {vdata}
    Best model:    {best}
    Train samples: {n_tr}   Val samples: {n_va}

    --- Knee-index accuracy (val) ---
    Exact match:         {exact:.3f}
    Tolerance-1 (+/-1):  {tol1:.3f}
    Tolerance-2 (+/-2):  {tol2:.3f}
    Mean abs error:      {mae:.2f} eigenvalue indices

    --- RFI detection binary (val) ---
    Precision:  {prec:.3f}
    Recall:     {rec:.3f}
    F1:         {f1:.3f}
    TP={tp}  TN={tn}  FP={fp}  FN={fn}

    --- Confidence calibration (val) ---
    High-conf (>=0.6): {n_hi} samples  exact acc={hi_acc:.3f}
    Low-conf  (<0.6):  {n_lo} samples  exact acc={lo_acc:.3f}
    Mean entropy:      {ent:.3f}

    --- Training ---
    Stopped at epoch: {stop_ep}
    Best val loss:    {best_loss:.4f}
    ========================================================
    """).format(
        data=args.data,
        vdata=args.val_data or "(split from --data)",
        best=best_path,
        n_tr=len(y_tr), n_va=len(y_va),
        exact=vm["exact"], tol1=vm["tol1"], tol2=vm["tol2"], mae=vm["mae"],
        prec=vm["prec"], rec=vm["rec"], f1=vm["f1"],
        tp=vm["tp"], tn=vm["tn"], fp=vm["fp"], fn=vm["fn"],
        n_hi=int(vm["high_conf"].sum()),
        n_lo=int((~vm["high_conf"]).sum()),
        hi_acc=vm["hi_acc"], lo_acc=vm["lo_acc"],
        ent=float(vm["entropy"].mean()),
        stop_ep=len(history.history["loss"]),
        best_loss=float(min(history.history["val_loss"])),
    )
    print(report)

    rpt_path = os.path.join(args.out_dir, "evaluation_report.txt")
    with open(rpt_path, "w") as f:
        f.write(report)
    print("Wrote report:", rpt_path)

    # ------------------------------------------------------------------
    # Optional held-out test evaluation
    # ------------------------------------------------------------------
    if args.test_data:
        if not os.path.exists(args.test_data):
            print("Warning: --test-data not found:", args.test_data)
        else:
            print("\nEvaluating held-out test set:", args.test_data)
            dt       = np.load(args.test_data, allow_pickle=True)
            eigen_te = dt["eigen_input"].astype(np.float32)
            glob_te  = dt["global_input"].astype(np.float32)
            y_te     = dt["labels"].astype(np.int32)
            tm = evaluate(best_model, eigen_te, glob_te, y_te)

            te_rpt = textwrap.dedent("""
    ========================================================
    TEST SET EVALUATION
    Data:    {data}
    Samples: {n}

    --- Knee-index accuracy ---
    Exact match:         {exact:.3f}
    Tolerance-1 (+/-1):  {tol1:.3f}
    Tolerance-2 (+/-2):  {tol2:.3f}
    Mean abs error:      {mae:.2f} eigenvalue indices

    --- RFI detection binary ---
    Precision:  {prec:.3f}
    Recall:     {rec:.3f}
    F1:         {f1:.3f}
    TP={tp}  TN={tn}  FP={fp}  FN={fn}
    ========================================================
            """).format(
                data=args.test_data, n=len(y_te),
                exact=tm["exact"], tol1=tm["tol1"],
                tol2=tm["tol2"], mae=tm["mae"],
                prec=tm["prec"], rec=tm["rec"], f1=tm["f1"],
                tp=tm["tp"], tn=tm["tn"], fp=tm["fp"], fn=tm["fn"])
            print(te_rpt)
            with open(rpt_path, "a") as f:
                f.write(te_rpt)
            print("Test results appended to", rpt_path)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_curves(hist_path, os.path.join(args.out_dir, "training_curves.png"))
    print("Wrote training_curves.png")

    cm = binned_confusion(y_va, vm["knee_pred"], BINS)
    plot_confusion(cm, BINS, os.path.join(args.out_dir, "confusion_matrix.png"))
    print("Wrote confusion_matrix.png")


if __name__ == "__main__":
    main()
