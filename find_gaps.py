"""
find_gaps.py

Scan the raw NISAR L0B HDF5 file range-sample axis for large runs of
zero or near-zero power, which indicate radar transmission gaps.

Reads the data column-by-column in chunks to avoid loading the full
(182760 x 52866) array into memory.

Strategy:
  - Collapse the pulse axis: compute mean magnitude across all pulses
    for each range sample.  A gap shows up as a sustained region where
    this mean is at or near zero.
  - Find contiguous runs where mean magnitude < THRESHOLD.
  - Report any run longer than MIN_GAP_WIDTH samples.

Output:
  - Console table of gap start/end/width
  - gap_profile.png  --  mean magnitude vs range sample with gaps marked
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HDF5_PATH    = './nisar_data/raw/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5'
DATASET_PATH = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'

# Samples with mean magnitude below this are considered dead
THRESHOLD    = 10.0     # raw ADC units (u16, so signal is typically 100s-1000s)

# Only report gaps wider than this many range samples
MIN_GAP_WIDTH = 100

# Read this many range samples at a time to keep memory reasonable
CHUNK_COLS   = 1000

OUT_PNG = 'gap_profile.png'


def main():
    with h5py.File(HDF5_PATH, 'r') as f:
        ds = f[DATASET_PATH]
        n_pulses, n_range = ds.shape
        print(f"Dataset shape : {n_pulses} pulses x {n_range} range samples")
        print(f"Reading in chunks of {CHUNK_COLS} range samples ...\n")

        mean_mag = np.empty(n_range, dtype=np.float32)

        for col_start in range(0, n_range, CHUNK_COLS):
            col_end = min(col_start + CHUNK_COLS, n_range)
            chunk   = ds[:, col_start:col_end]              # (n_pulses, chunk_width)

            # Assemble complex and take magnitude, then mean over pulses
            c = chunk['r'].astype(np.float32) + 1j * chunk['i'].astype(np.float32)
            mean_mag[col_start:col_end] = np.abs(c).mean(axis=0)

            if col_start % 10000 == 0:
                print(f"  processed range samples 0..{col_end} / {n_range}")

    print(f"\nDone reading.  mean_mag range: {mean_mag.min():.2f} .. {mean_mag.max():.2f}")

    # ── Find dead zones ────────────────────────────────────────────────────
    is_dead = mean_mag < THRESHOLD

    # Find run-length encoded transitions
    transitions = np.diff(is_dead.astype(np.int8), prepend=0, append=0)
    starts = np.where(transitions ==  1)[0]
    ends   = np.where(transitions == -1)[0]

    print(f"\nGap detection (threshold={THRESHOLD}, min_width={MIN_GAP_WIDTH}):")
    print(f"{'Start':>8}  {'End':>8}  {'Width':>8}  {'MeanMag_in_gap':>16}")
    print("-" * 50)

    gaps = []
    for s, e in zip(starts, ends):
        width = e - s
        if width >= MIN_GAP_WIDTH:
            mean_in = float(mean_mag[s:e].mean())
            print(f"{s:>8}  {e:>8}  {width:>8}  {mean_in:>16.4f}")
            gaps.append((s, e, width))

    if not gaps:
        print("  No gaps found.")
        return

    # ── Tile impact summary ────────────────────────────────────────────────
    BLOCK_WIDTH  = 250
    N_RANGE_COL  = 211
    RANGE_USED   = BLOCK_WIDTH * N_RANGE_COL   # 52750

    print(f"\nTile impact (BLOCK_WIDTH={BLOCK_WIDTH}, {N_RANGE_COL} tiles):")
    print(f"{'Tile':>6}  {'RangeStart':>11}  {'RangeEnd':>9}  {'Status'}")
    print("-" * 50)

    exclude = set()
    for i in range(N_RANGE_COL):
        ts = i * BLOCK_WIDTH
        te = ts + BLOCK_WIDTH
        for gs, ge, _ in gaps:
            if ts < ge and te > gs:   # any overlap with a gap
                exclude.add(i)

    # Print only tiles near gap boundaries
    boundary_tiles = set()
    for i in range(N_RANGE_COL):
        if i in exclude or (i - 1) in exclude or (i + 1) in exclude:
            boundary_tiles.add(i)

    for i in sorted(boundary_tiles):
        ts     = i * BLOCK_WIDTH
        te     = ts + BLOCK_WIDTH
        status = "EXCLUDE" if i in exclude else "ok"
        print(f"{i:>6}  {ts:>11}  {te:>9}  {status}")

    # Valid contiguous segments
    valid  = sorted(set(range(N_RANGE_COL)) - exclude)
    groups = []
    group  = [valid[0]]
    for t in valid[1:]:
        if t == group[-1] + 1:
            group.append(t)
        else:
            groups.append(group)
            group = [t]
    groups.append(group)

    print(f"\nValid tile segments (exclude {len(exclude)} gap tiles):")
    for gi, g in enumerate(groups):
        print(f"  Segment {gi}: tiles {g[0]:>3}..{g[-1]:>3}  "
              f"range {g[0]*BLOCK_WIDTH:>6}..{(g[-1]+1)*BLOCK_WIDTH:>6}  "
              f"({len(g)} tiles)")
    print(f"  Total valid: {len(valid)} / {N_RANGE_COL} tiles")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    ax = axes[0]
    ax.plot(mean_mag, linewidth=0.5, color='steelblue')
    for gs, ge, _ in gaps:
        ax.axvspan(gs, ge, color='red', alpha=0.3, label='gap')
    ax.set_xlabel('Range sample index')
    ax.set_ylabel('Mean magnitude (ADC units)')
    ax.set_title('Mean pulse magnitude per range sample -- gaps marked in red')
    ax.set_xlim(0, n_range)

    ax = axes[1]
    with np.errstate(divide='ignore'):
        log_mag = 20 * np.log10(mean_mag + 1e-6)
    ax.plot(log_mag, linewidth=0.5, color='tomato')
    for gs, ge, _ in gaps:
        ax.axvspan(gs, ge, color='red', alpha=0.3)
    ax.axhline(20 * np.log10(THRESHOLD + 1e-6), color='black',
               linestyle='--', linewidth=1, label=f'threshold={THRESHOLD}')
    ax.set_xlabel('Range sample index')
    ax.set_ylabel('Mean magnitude (dB)')
    ax.set_title('Log scale -- easier to see gap floor level')
    ax.set_xlim(0, n_range)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")


if __name__ == '__main__':
    main()
