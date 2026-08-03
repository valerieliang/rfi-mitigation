"""
score_scene_branch_updates.py

Score a REAL NISAR granule with a model from train_branch_updates.py -- the
single-branch classifier with combined features:
    - 12 valid EVs (max-normalized, dB)
    - 11 EV slopes
    - 12 valid SCM diagonal entries (max-normalized, dB)
    - 2 emphasis features [max EV, max pulse power] (normalized)

No ground truth: k > 0 predictions are candidate detections, not verified RFI.

Usage
-----
    python score_scene_branch_updates.py /path/to/NISAR_L0_..._001.h5 \
        --model models/model_branch_updates/best_model.keras \
        --pulse-start 681445 --pulse-end 757408 \
        --output-dir score/model_branch_updates/la
"""

import os
import sys

import numpy as np

# Import the feature extraction function from train_branch_updates
from train_branch_updates import features_from_records, N_BRANCH_FEATURES, EPS, DB_FLOOR, M

# Import score_scene infrastructure
import score_scene
from score_scene import (
    read_raw_data_batch, get_subswath_mask, compute_scm_and_eigs,
    tile_signal_power, build_tone_remover, PULSE_CHUNK_DEFAULT,
    save_predictions_h5, plot_knee_map, plot_confidence_map,
    plot_power_vs_knee, plot_confidence_by_knee, plot_eigen_profiles,
    plot_selected_predictions, plot_pred_hist_all_channels,
    CPI_LEN_DEFAULT, CPI_WIDTH_DEFAULT
)


def _default_freq_a():
    """Inject --freq A into argv unless the caller already chose a frequency."""
    if any(a == '--freq' or a.startswith('--freq=') for a in sys.argv[1:]):
        return
    sys.argv.extend(['--freq', 'A'])


def score_channel_branch_updates(raw, freq, pol, model, args):
    """
    Score one channel with the branch_updates model (single input, 37 features).

    This is a specialized version of score_scene.score_channel() that builds
    the 37-feature combined branch input instead of the 2-input or 3-input
    model formats.
    """
    cpi_len, cpi_width = args.cpi_len, args.cpi_width

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    # Default to full extent if not specified
    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    p_end = min(p_end, total_pulses)

    r_start = args.range_start if args.range_start is not None else 0
    r_end = (args.range_end if args.range_end is not None else total_range)
    r_end = min(r_end, total_range)

    n_pt = (p_end - p_start) // cpi_len
    n_rt = (r_end - r_start) // cpi_width
    p_end = p_start + n_pt * cpi_len
    r_end = r_start + n_rt * cpi_width

    if n_pt <= 0 or n_rt <= 0:
        raise ValueError('Window is smaller than one CPI tile')

    n_tiles = n_pt * n_rt
    print(f"\n[{freq}-{pol}]  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pt} x {n_rt} = {n_tiles} tiles")

    # Build caltone remover
    if args.remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, total_range)
        print(f"  caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz, "
              f"window = 64)")
    else:
        remover = None
        print("  caltone removal OFF")

    print(f"  model takes 1 input with {N_BRANCH_FEATURES} features -> building combined branch")

    # Allocate arrays
    branch1_all = np.zeros((n_tiles, N_BRANCH_FEATURES), dtype=np.float32)
    eigvals_all = np.zeros((n_tiles, M), dtype=np.float32)
    power_db = np.zeros(n_tiles, dtype=np.float32)
    valid_frac = np.zeros(n_tiles, dtype=np.float32)
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)

    chunk_tiles = max(1, args.pulse_chunk // cpi_len)
    k = 0

    for chunk_start in range(0, n_pt, chunk_tiles):
        n_here = min(chunk_tiles, n_pt - chunk_start)
        cp0 = p_start + chunk_start * cpi_len
        cp1 = cp0 + n_here * cpi_len

        if remover is not None:
            raw_full = np.ascontiguousarray(
                read_raw_data_batch(
                    raw, freq, pol, slice(cp0, cp1), slice(0, total_range)
                )
            ).astype(np.complex64)
            for ip in range(raw_full.shape[0]):
                raw_full[ip] = remover.remove_tone(raw_full[ip])
            raw_chunk = raw_full[:, r_start:r_end]
        else:
            raw_chunk = read_raw_data_batch(
                raw, freq, pol, slice(cp0, cp1), slice(r_start, r_end)
            )

        mask_chunk = (
            get_subswath_mask(raw, freq, pol,
                              np.arange(cp0, cp1), np.arange(r_start, r_end))
            if args.compute_subswath_mask else None
        )

        for lp in range(n_here):
            pt = chunk_start + lp
            lp0, lp1 = lp * cpi_len, (lp + 1) * cpi_len

            for rt in range(n_rt):
                lr0, lr1 = rt * cpi_width, (rt + 1) * cpi_width

                cpi = np.ascontiguousarray(
                    raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                cpi_mask = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                            if mask_chunk is not None else None)

                _, eigvals, diag_lin, diag_valid = compute_scm_and_eigs(
                    cpi, cpi_mask,
                    args.off_diag_overlap_ratio, args.diag_valid_ratio
                )

                # Build the combined 37-feature branch
                branch1_all[k] = features_from_records(eigvals, diag_lin, diag_valid)
                eigvals_all[k] = eigvals
                power_db[k] = 10.0 * np.log10(tile_signal_power(cpi, cpi_mask))
                valid_frac[k] = (float(cpi_mask.sum()) / cpi_mask.size
                                 if cpi_mask is not None else 1.0)
                tile_pulse[k] = p_start + pt * cpi_len
                tile_range[k] = r_start + lr0
                k += 1

        print(f"    pulse tiles {chunk_start + n_here}/{n_pt}")

    print(f"  predicting on {n_tiles} tiles ...")
    probs = model.predict(branch1_all, batch_size=args.batch_size, verbose=0)

    knee = np.argmax(probs, axis=-1).astype(np.int8)
    confidence = np.max(probs, axis=-1).astype(np.float32)
    entropy = (-np.sum(probs * np.log(probs + EPS), axis=-1)).astype(np.float32)

    return {
        'freq': freq, 'pol': pol, 'chan': f'{freq}-{pol}',
        'knee': knee, 'confidence': confidence, 'entropy': entropy,
        'eigvals': eigvals_all, 'power_db': power_db, 'valid_frac': valid_frac,
        'diag_profile': None, 'diag_valid_frac': None,
        'global_features': None,  # Combined into branch1, not stored separately
        'branch1_features': branch1_all,  # Store the combined features
        'tile_pulse': tile_pulse, 'tile_range': tile_range,
        'n_pt': n_pt, 'n_rt': n_rt,
        'pulse_window': [p_start, p_end], 'range_window': [r_start, r_end],
        'n_classes': probs.shape[-1],
    }


def main():
    _default_freq_a()
    args = score_scene.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    n_inputs = len(model.inputs)
    if n_inputs != 1:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 1, so it is not a "
            f"branch-updates model. Models by input count:\n"
            f"  1 input  = branch_updates (THIS SCRIPT)\n"
            f"  2 inputs = baseline or new-globals (use score_scene.py or "
            f"score_scene_new_globals.py)\n"
            f"  3 inputs = sorted diagonal profile (use score_scene_diag.py)"
        )

    n_features = int(model.inputs[0].shape[-1])
    if n_features != N_BRANCH_FEATURES:
        raise SystemExit(
            f"{args.model} wants {n_features} features, not {N_BRANCH_FEATURES}, "
            f"so it does not match the branch_updates architecture. Expected:\n"
            f"  12 EVs + 11 EV slopes + 12 SCM diagonal + 2 emphasis = {N_BRANCH_FEATURES} features"
        )

    print(f"1 input confirmed, {n_features} features: single-branch combined model")

    # Import NISAR reader
    from nisar.products.readers.Raw import Raw
    raw = Raw(hdf5file=args.l0b_file)

    # Determine channels to process
    if args.freq is not None:
        freqs = args.freq if isinstance(args.freq, list) else [args.freq]
    else:
        freqs = ['A', 'B']

    pols = args.pol if args.pol is not None else ['HH', 'HV', 'VH', 'VV']
    if not isinstance(pols, list):
        pols = [pols]

    available_channels = []
    for f in freqs:
        for p in pols:
            try:
                raw.getRawDataset(f, p)
                available_channels.append((f, p))
            except (KeyError, RuntimeError):
                pass

    if not available_channels:
        raise ValueError(f"No valid channels found for freq={freqs}, pol={pols}")

    print(f"  channels: {', '.join(f'{f}-{p}' for f, p in available_channels)}")

    # Score each channel
    recs = []
    for freq, pol in available_channels:
        rec = score_channel_branch_updates(raw, freq, pol, model, args)
        recs.append(rec)

    # Save results and generate plots
    os.makedirs(args.output_dir, exist_ok=True)

    for rec in recs:
        save_predictions_h5(rec, args, args.output_dir)
        plot_knee_map(rec, args.output_dir)
        plot_confidence_map(rec, args.output_dir)
        plot_power_vs_knee(rec, args.output_dir)
        plot_confidence_by_knee(rec, args.output_dir)
        plot_eigen_profiles(rec, args.output_dir)
        plot_selected_predictions(rec, args.output_dir)

    if len(recs) > 1:
        plot_pred_hist_all_channels(recs, args.output_dir)

    # Save summary
    summary = {
        'granule': os.path.basename(args.l0b_file),
        'model': os.path.basename(args.model),
        'model_variant': 'branch_updates',
        'n_inputs': 1,
        'n_features': N_BRANCH_FEATURES,
        'channels': [rec['chan'] for rec in recs],
        'predictions_by_channel': {}
    }

    for rec in recs:
        chan = rec['chan']
        summary['predictions_by_channel'][chan] = {
            'n_tiles': len(rec['knee']),
            'class_counts': {int(k): int(v) for k, v in
                           zip(*np.unique(rec['knee'], return_counts=True))},
            'mean_confidence': float(rec['confidence'].mean()),
            'mean_entropy': float(rec['entropy'].mean()),
        }

    summary_path = os.path.join(args.output_dir, 'results.json')
    import json
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print("Scene scoring complete!")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
