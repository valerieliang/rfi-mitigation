"""
rfi_gen.canvas

Pull clean slow-time CPI blocks from the RSLC granules that passed the
quality screen, so synthetic RFI can be overlaid on real speckle rather
than synthetic noise.

The QC summary CSV (from check_nisar_clean.py) lists each granule and a
flag. This module reads it, keeps the CLEAN rows, opens the .h5, and
extracts a complex (M, K) block: M consecutive slow-time samples
(azimuth lines, the pulse proxy in a focused product) by K range
samples.

Reminder: for a focused RSLC the slow-time axis is the azimuth
direction, which is an approximation of the raw pulse dimension the EVD
method assumes. It is adequate for prototyping; the faithful pipeline
uses L0B raw pulses.

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


def read_clean_list(qc_csv):
    """Return granule file names flagged CLEAN in the QC summary CSV."""
    clean = []
    with open(qc_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flag", "").strip().upper() == "CLEAN":
                clean.append(row["file"])
    return clean


def _find_swaths(h5):
    for band in ("LSAR", "SSAR"):
        for prod in ("RSLC", "SLC"):
            path = "science/{}/{}/swaths".format(band, prod)
            if path in h5:
                return path
    found = {"p": None}

    def visit(name, obj):
        if found["p"] is None and isinstance(obj, h5py.Group) \
                and name.endswith("swaths"):
            for k in obj.keys():
                if k.lower().startswith("frequency"):
                    found["p"] = name
                    return
    h5.visititems(visit)
    return found["p"]


def _first_pol_dataset(h5, swaths, freq):
    grp = h5["{}/{}".format(swaths, freq)]
    pols = [k for k in grp.keys()
            if k.upper() in KNOWN_POLS and isinstance(grp[k], h5py.Dataset)]
    # Prefer co-pol.
    pols.sort(key=lambda p: (p[0] != p[1], p))
    if not pols:
        raise ValueError("no polarization dataset under {}".format(freq))
    return grp[pols[0]], pols[0]


def _to_complex(block):
    """Convert a read block (compound half/float or native complex)."""
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


def extract_block(h5_path, n_pulses=32, n_range=128, az_offset=None,
                  rg_offset=None, freq=None, rng=None):
    """
    Extract one clean complex (n_pulses, n_range) slow-time block.

    Offsets default to a random interior location so repeated calls
    sample different parts of the scene. Returns the complex block.
    """
    if h5py is None:
        raise ImportError("h5py is required to read NISAR granules.")
    rng = np.random.default_rng() if rng is None else rng

    with h5py.File(h5_path, "r") as h5:
        swaths = _find_swaths(h5)
        if swaths is None:
            raise ValueError("no swaths group in {}".format(h5_path))
        freqs = [k for k in h5[swaths].keys()
                 if k.lower().startswith("frequency")]
        if freq:
            freqs = [f for f in freqs
                     if f.lower().endswith(freq.lower())] or freqs
        dset, pol = _first_pol_dataset(h5, swaths, sorted(freqs)[0])
        naz, nrg = dset.shape

        if n_pulses > naz or n_range > nrg:
            raise ValueError("requested block larger than scene {}x{}"
                             .format(naz, nrg))
        if az_offset is None:
            az_offset = int(rng.integers(0, naz - n_pulses + 1))
        if rg_offset is None:
            rg_offset = int(rng.integers(0, nrg - n_range + 1))

        block = dset[az_offset:az_offset + n_pulses,
                     rg_offset:rg_offset + n_range]
        return _to_complex(block).astype(np.complex64)


def iter_clean_blocks(qc_csv, data_dir, n_pulses=32, n_range=128,
                      blocks_per_file=8, freq=None, rng=None):
    """
    Yield (granule_name, az_offset, rg_offset, block) over all CLEAN
    granules, sampling blocks_per_file random blocks from each.
    """
    rng = np.random.default_rng() if rng is None else rng
    for name in read_clean_list(qc_csv):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        for _ in range(blocks_per_file):
            with h5py.File(path, "r") as h5:
                swaths = _find_swaths(h5)
                freqs = [k for k in h5[swaths].keys()
                         if k.lower().startswith("frequency")]
                dset, _ = _first_pol_dataset(h5, swaths, sorted(freqs)[0])
                naz, nrg = dset.shape
            az = int(rng.integers(0, naz - n_pulses + 1))
            rg = int(rng.integers(0, nrg - n_range + 1))
            block = extract_block(path, n_pulses, n_range, az, rg, freq, rng)
            yield name, az, rg, block
