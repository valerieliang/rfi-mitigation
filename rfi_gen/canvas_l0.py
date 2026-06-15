"""
rfi_gen.canvas_l0

Pull raw slow-time CPI blocks from NISAR L0B (Level-0) HDF5 files so
synthetic RFI can be overlaid on real pulse data rather than on focused
RSLC azimuth lines.

Relationship to canvas.py
--------------------------
canvas.py   -- reads RSLC (focused); slow-time axis is azimuth lines,
               which is an approximation of the raw pulse dimension.
canvas_l0.py -- reads L0B (raw); slow-time axis IS the pulse axis,
               which is exactly what the ST-EVD method assumes.

Both modules expose the same interface so augment.py can swap between
them with a single flag (--l0).

NISAR L0B HDF5 layout (auto-discovered, mirrors multiple product versions)
---------------------------------------------------------------------------
  science/<BAND>/L0B/swaths/frequency<A|B>/<POL>   [n_pulses x n_range]

The dataset may be:
  - complex64 (native complex, already decoded)
  - compound dtype with fields ('r','i') or ('real','imag') for
    half-precision or float32 complex
  - a flat real array (amplitude only; rare but handled)

A [n_pulses x n_range] block is stored pulse-major, so slicing
raw[az0:az0+M, rg0:rg0+K] gives exactly the (M, K) slow-time block
the ST-EVD pipeline needs.

QC CSV produced by check_l0_clean.py has the same columns as the RSLC
QC CSV (file, flag, ...) so read_clean_list is shared.

No non-ASCII characters are used in this file.

Dependencies: numpy, h5py.
"""

import csv
import os

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

KNOWN_POLS = {"HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"}

# Standard L0B path prefix; also tried as a fallback by _find_swaths_l0.
_L0B_ROOTS = [
    "science/LSAR/L0B/swaths",
    "science/SSAR/L0B/swaths",
]


# ---------------------------------------------------------------------------
# HDF5 structure discovery
# ---------------------------------------------------------------------------

def _find_swaths_l0(h5):
    """
    Locate the L0B swaths group. Returns the path string or None.

    Tries the standard NISAR L0B paths first, then falls back to a tree
    walk looking for any '.../swaths' group that contains a frequency
    sub-group, which also covers non-standard or simulated L0 products.
    """
    for root in _L0B_ROOTS:
        if root in h5:
            return root

    # Fallback: walk the tree.
    found = {"p": None}

    def visit(name, obj):
        if found["p"] is not None:
            return
        if isinstance(obj, h5py.Group) and name.endswith("swaths"):
            for k in obj.keys():
                if k.lower().startswith("frequency"):
                    found["p"] = name
                    return

    h5.visititems(visit)
    return found["p"]


def _first_pol_dataset(h5, swaths, freq):
    """
    Return (dataset, pol_name) for the preferred polarization under
    swaths/freq. Co-pol is preferred over cross-pol.
    """
    grp = h5["{}/{}".format(swaths, freq)]
    pols = [k for k in grp.keys()
            if k.upper() in KNOWN_POLS and isinstance(grp[k], h5py.Dataset)]
    # Co-pol first (HH or VV before HV/VH), then alphabetical.
    pols.sort(key=lambda p: (p[0] != p[1], p))
    if not pols:
        raise ValueError("no polarization dataset under {}/{}".format(
            swaths, freq))
    return grp[pols[0]], pols[0]


def _to_complex(block):
    """
    Convert a raw HDF5 block to complex64.

    Handles compound dtypes (r/i or real/imag fields), native complex,
    and real-valued fallback.
    """
    if block.dtype.names:
        names = {n.lower(): n for n in block.dtype.names}
        if "r" in names and "i" in names:
            return (block[names["r"]].astype(np.float32)
                    + 1j * block[names["i"]].astype(np.float32))
        if "real" in names and "imag" in names:
            return (block[names["real"]].astype(np.float32)
                    + 1j * block[names["imag"]].astype(np.float32))
        first = block.dtype.names[0]
        return block[first].astype(np.float32).astype(np.complex64)
    if np.iscomplexobj(block):
        return block.astype(np.complex64)
    return block.astype(np.float32).astype(np.complex64)


# ---------------------------------------------------------------------------
# Public API (mirrors canvas.py)
# ---------------------------------------------------------------------------

def read_clean_list(qc_csv):
    """Return filenames flagged CLEAN in the QC summary CSV."""
    clean = []
    with open(qc_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flag", "").strip().upper() == "CLEAN":
                clean.append(row["file"])
    return clean


def extract_block(h5_path, n_pulses=32, n_range=128,
                  az_offset=None, rg_offset=None,
                  freq=None, rng=None, pol=None):
    """
    Extract one (n_pulses, n_range) raw complex block from an L0B file.

    Parameters
    ----------
    h5_path : str
        Path to the L0B HDF5 file.
    n_pulses : int
        Number of pulses to read (the M dimension / CPI size).
    n_range : int
        Number of range samples to read (the K dimension).
    az_offset : int or None
        Pulse-axis start index. Randomized if None.
    rg_offset : int or None
        Range-axis start index. Randomized if None.
    freq : str or None
        'A' or 'B' to select a specific frequency; defaults to first found.
    rng : np.random.Generator or None
        Random generator for offset selection.
    pol : str or None
        Specific polarization ('HH', 'VV', ...). If None, co-pol is used.

    Returns
    -------
    np.ndarray of shape (n_pulses, n_range), dtype complex64.
    """
    if h5py is None:
        raise ImportError("h5py is required to read L0B granules.")
    rng = np.random.default_rng() if rng is None else rng

    with h5py.File(h5_path, "r") as h5:
        swaths = _find_swaths_l0(h5)
        if swaths is None:
            raise ValueError("no L0B swaths group found in {}".format(h5_path))

        freqs = [k for k in h5[swaths].keys()
                 if k.lower().startswith("frequency")]
        if not freqs:
            raise ValueError("no frequency groups under {}".format(swaths))
        if freq:
            matched = [f for f in freqs if f.lower().endswith(freq.lower())]
            if matched:
                freqs = matched

        freq_key = sorted(freqs)[0]

        if pol is not None:
            grp = h5["{}/{}".format(swaths, freq_key)]
            if pol not in grp:
                raise ValueError("polarization {} not in {}/{}".format(
                    pol, swaths, freq_key))
            dset = grp[pol]
        else:
            dset, _ = _first_pol_dataset(h5, swaths, freq_key)

        n_total_pulses, n_total_range = dset.shape

        if n_pulses > n_total_pulses:
            raise ValueError(
                "requested {} pulses but scene only has {}".format(
                    n_pulses, n_total_pulses))
        if n_range > n_total_range:
            raise ValueError(
                "requested {} range samples but scene only has {}".format(
                    n_range, n_total_range))

        if az_offset is None:
            az_offset = int(rng.integers(0, n_total_pulses - n_pulses + 1))
        if rg_offset is None:
            rg_offset = int(rng.integers(0, n_total_range - n_range + 1))

        raw = dset[az_offset:az_offset + n_pulses,
                   rg_offset:rg_offset + n_range]
        return _to_complex(raw).astype(np.complex64)


def iter_clean_blocks(qc_csv, data_dir, n_pulses=32, n_range=128,
                      blocks_per_file=8, freq=None, rng=None):
    """
    Yield (granule_name, az_offset, rg_offset, block) for every CLEAN
    L0B granule listed in qc_csv, sampling blocks_per_file random blocks
    from each.

    Drop-in replacement for canvas.iter_clean_blocks.
    """
    rng = np.random.default_rng() if rng is None else rng
    for name in read_clean_list(qc_csv):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        for _ in range(blocks_per_file):
            try:
                block = extract_block(path, n_pulses, n_range,
                                      freq=freq, rng=rng)
                yield name, None, None, block
            except Exception:
                continue