"""
plot_top_anomalies.py

Reads one or more top_anomalies_<pol>.json files (produced by
score_anomaly.py) and plots every listed tile using the same 4-panel
diagnostic figure as plot_random_cpi.py -- raw data, mask overlay, SCM,
and eigenvalue profile -- so the highest-scoring tiles from a scoring run
can be visually inspected without hand-computing pulse/range windows.

This does NOT run any anomaly scoring itself; it only re-reads and
re-plots tiles whose coordinates were already identified by
score_anomaly.py. Each figure is also annotated with that tile's rank and
anomaly score, so you can cross-reference it against the JSON directly.

Usage
-----
    python plot_top_anomalies.py \
        eval/next_8000/top_anomalies_HV.json \
        eval/next_8000/top_anomalies_HH.json \
        --output-dir plots/top_anomalies

    # Only plot the first 10 entries of each file
    python plot_top_anomalies.py eval/next_8000/top_anomalies_HV.json \
        --output-dir plots/top_anomalies --n-top 10

Each top_anomalies_<pol>.json entry produced by the current version of
score_anomaly.py includes its own 'l0b_file' and 'freq', so no additional
arguments are needed. If you are working with an OLDER json file that
predates that field, pass --l0b-file / --freq to use as a fallback for any
entry missing them.

Feature-extraction settings (n_keep, CPI size, gap-exclusion ratios) are, by
default, auto-loaded from norm_stats.npz via --model-dir, so the plotted
normalized eigenvalue vector matches exactly what the model actually scored.
If --model-dir is not given, the anomaly_features.py defaults are used
instead (fine as long as those match what score_anomaly.py used).

Outputs (in --output-dir)
--------------------------
    <pol>_rank<NNN>_pulse<P>_range<R>.png   -- 4-panel diagnostic figure
    <pol>_rank<NNN>_pulse<P>_range<R>.json  -- numeric report for that tile
"""

import os
import json
import argparse
import numpy as np

from anomaly_features import (
    compute_gap_exclusion_scm,
    eigen_decompose_descending,
    extract_anomaly_features,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    N_KEEP_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    EPS,
)
from plot_random_cpi import (
    read_raw_data_batch,
    get_subswath_mask,
    plot_one_cpi_sample,
)

from nisar.products.readers.Raw import Raw


def load_anomaly_entries(json_paths, n_top, fallback_l0b_file, fallback_freq):
    """
    Load and flatten anomaly entries from one or more top_anomalies_<pol>.json
    files, filling in l0b_file/freq from the fallback args for older files
    that predate those fields.

    Parameters
    ----------
    json_paths : list[str]
    n_top : int or None
        If given, only the first n_top entries of EACH file are kept
        (files are already rank-sorted, so this is "top N per file").
    fallback_l0b_file : str or None
    fallback_freq : str

    Returns
    -------
    entries : list[dict]
    """
    entries = []
    for path in json_paths:
        with open(path, 'r') as fh:
            file_entries = json.load(fh)

        if n_top is not None:
            file_entries = file_entries[:n_top]

        for entry in file_entries:
            if 'l0b_file' not in entry or entry['l0b_file'] is None:
                if fallback_l0b_file is None:
                    raise ValueError(
                        f"Entry in {path} has no 'l0b_file' field and no --l0b-file "
                        f"fallback was given. Pass --l0b-file to specify it."
                    )
                entry['l0b_file'] = fallback_l0b_file
            if 'freq' not in entry or entry['freq'] is None:
                entry['freq'] = fallback_freq
            entries.append(entry)

    return entries


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot every tile listed in one or more top_anomalies_<pol>.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('anomalies_json', nargs='+',
                        help='One or more top_anomalies_<pol>.json files from score_anomaly.py')
    parser.add_argument('--output-dir', type=str, default='plots/top_anomalies', help='Output directory for figures')

    parser.add_argument('--l0b-file', type=str, default=None,
                        help='Fallback L0B file path for entries that predate the l0b_file field')
    parser.add_argument('--freq', choices=['A', 'B'], default='A',
                        help='Fallback frequency for entries that predate the freq field (default: A)')

    parser.add_argument('--n-top', type=int, default=None,
                        help='Only plot the first N entries of each json file (default: all entries present)')

    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory containing norm_stats.npz to auto-load cpi_len/cpi_width/n_keep/'
                             'gap-exclusion ratios (recommended, so the plotted features match what was '
                             'actually scored). If omitted, anomaly_features.py defaults are used.')
    parser.add_argument('--cpi-len', type=int, default=None, help='Overrides CPI length (pulses)')
    parser.add_argument('--cpi-width', type=int, default=None, help='Overrides CPI width (range samples)')
    parser.add_argument('--n-keep', type=int, default=None, help='Overrides number of leading eigenvalues kept')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=None, help='Overrides gap-exclusion off-diag ratio')
    parser.add_argument('--diag-valid-ratio', type=float, default=None, help='Overrides gap-exclusion diag ratio')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve feature-extraction settings: CLI overrides > model-dir > defaults
    # ------------------------------------------------------------------
    cpi_len = CPI_LEN_DEFAULT
    cpi_width = CPI_WIDTH_DEFAULT
    n_keep = N_KEEP_DEFAULT
    off_diag_overlap_ratio = OFF_DIAG_OVERLAP_RATIO_DEFAULT
    diag_valid_ratio = DIAG_VALID_RATIO_DEFAULT

    if args.model_dir is not None:
        norm_stats_path = os.path.join(args.model_dir, 'norm_stats.npz')
        print(f"Auto-loading feature-extraction settings from {norm_stats_path} ...")
        stats = np.load(norm_stats_path)
        cpi_len = int(stats['cpi_len'])
        cpi_width = int(stats['cpi_width'])
        n_keep = int(stats['n_keep'])
        off_diag_overlap_ratio = float(stats['off_diag_overlap_ratio'])
        diag_valid_ratio = float(stats['diag_valid_ratio'])

    if args.cpi_len is not None:
        cpi_len = args.cpi_len
    if args.cpi_width is not None:
        cpi_width = args.cpi_width
    if args.n_keep is not None:
        n_keep = args.n_keep
    if args.off_diag_overlap_ratio is not None:
        off_diag_overlap_ratio = args.off_diag_overlap_ratio
    if args.diag_valid_ratio is not None:
        diag_valid_ratio = args.diag_valid_ratio

    print(f"Using cpi_len={cpi_len}, cpi_width={cpi_width}, n_keep={n_keep}, "
          f"off_diag_overlap_ratio={off_diag_overlap_ratio}, diag_valid_ratio={diag_valid_ratio}")

    # ------------------------------------------------------------------
    # Load anomaly entries from all provided json files
    # ------------------------------------------------------------------
    entries = load_anomaly_entries(args.anomalies_json, args.n_top, args.l0b_file, args.freq)
    print(f"\nLoaded {len(entries)} anomaly entries from {len(args.anomalies_json)} file(s)")

    if not entries:
        print("No entries to plot.")
        return

    # ------------------------------------------------------------------
    # Open each distinct L0B file only once
    # ------------------------------------------------------------------
    raw_readers = {}

    def get_raw_reader(l0b_file):
        if l0b_file not in raw_readers:
            print(f"\nOpening {l0b_file} ...")
            raw = Raw(hdf5file=str(l0b_file))
            raw.parsePolarizations()
            raw_readers[l0b_file] = raw
        return raw_readers[l0b_file]

    # ------------------------------------------------------------------
    # Plot every entry
    # ------------------------------------------------------------------
    for entry in entries:
        l0b_file = entry['l0b_file']
        freq = entry['freq']
        pol = entry['pol']
        rank = entry['rank']
        pulse_start = entry['pulse_start']
        range_start = entry['range_start']
        total_score = entry.get('total_score')

        raw = get_raw_reader(l0b_file)

        pulse_end = pulse_start + cpi_len
        range_end = range_start + cpi_width

        cpi_data = read_raw_data_batch(raw, freq, pol, slice(pulse_start, pulse_end), slice(range_start, range_end))
        pulse_indices = np.arange(pulse_start, pulse_end)
        range_indices = np.arange(range_start, range_end)
        cpi_mask = get_subswath_mask(raw, freq, pol, pulse_indices, range_indices)

        eigen_feat, global_feat, diag_valid_frac, _ = extract_anomaly_features(
            cpi_data,
            mask_valid_cpi=cpi_mask,
            n_keep=n_keep,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )

        scm, _, _ = compute_gap_exclusion_scm(
            cpi_data,
            mask_valid_cpi=cpi_mask,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
        eigvals_linear = eigen_decompose_descending(scm)
        eigvals_raw_db = (10.0 * np.log10(np.clip(eigvals_linear, EPS, None))).astype(np.float32)

        title_info = {
            'file': os.path.basename(l0b_file),
            'freq': freq,
            'pol': pol,
            'pulse_start': pulse_start,
            'range_start': range_start,
        }

        score_str = f"{total_score:.4f}" if total_score is not None else "n/a"
        extra_title = f"Anomaly rank {rank} ({pol})  |  total_score = {score_str}"

        tag = f"{pol}_rank{rank:03d}_pulse{pulse_start}_range{range_start}"
        out_png = os.path.join(args.output_dir, f'{tag}.png')
        out_json = os.path.join(args.output_dir, f'{tag}.json')

        plot_one_cpi_sample(
            cpi_data, cpi_mask, scm, eigvals_raw_db, eigen_feat, global_feat,
            title_info, out_png, extra_title=extra_title,
        )

        report = dict(entry)
        report['diag_valid_frac_recomputed'] = float(diag_valid_frac)
        report['normalized_eigvals_db_used'] = eigen_feat[:, 0].tolist()
        report['slopes_db_used'] = eigen_feat[:, 1].tolist()
        report['condition_number_db_recomputed'] = float(global_feat[0])
        report['effective_rank_recomputed'] = float(global_feat[1])
        report['eigvals_raw_db'] = eigvals_raw_db.tolist()
        with open(out_json, 'w') as fh:
            json.dump(report, fh, indent=2)

        print(f"  [{tag}] rank={rank} pol={pol} pulse_start={pulse_start} range_start={range_start} "
              f"score={score_str}")

    print(f"\nDone. Plotted {len(entries)} anomaly tiles to {args.output_dir}")


if __name__ == '__main__':
    main()