"""
nisar/fetch_nisar.py -- Search and download NISAR RSLC data from ASF Vertex.

Run from the project root:
    python nisar/fetch_nisar.py

Downloads .h5 files to data/nisar_rslc/ and saves aligned .npy tiles
ready for rfi_gen.overlay. No synthetic fallback -- all training data
must come from real satellite acquisitions.

Configuration: edit the block at the top of this file.
Credentials:   set EARTHDATA_TOKEN in .env (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import make_asf_session   # noqa: E402

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary AOI: Amazon basin interior -- clean L-band, low anthropogenic RFI
AOI = 'POLYGON((-75 -15, -50 -15, -50 5, -75 5, -75 -15))'

# Broader fallback if primary returns zero results
AOI_BROAD = 'POLYGON((-82 -56, -34 -56, -34 13, -82 13, -82 -56))'

MAX_RESULTS   = 10          # granules to download (each is 2-8 GB)
FREQUENCY     = 'A'         # 'A' = 40 MHz L-band primary band
POLARIZATION  = 'HH'        # channel to load for the .npy tile

# Tile size to extract from each .h5 for training
LOAD_AZ_LINES    = 32000    # ~1000 CPIs at M=32
LOAD_RNG_SAMPLES = 5120     # 10 range blocks at K=512

DATA_DIR = PROJECT_ROOT / 'data' / 'nisar_rslc'


# ---------------------------------------------------------------------------
# Search and download
# ---------------------------------------------------------------------------

def search_and_download(aoi: str, max_results: int, output_dir: Path) -> list[Path]:
    try:
        import asf_search as asf
    except ImportError as exc:
        raise ImportError("pip install asf_search") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    session = make_asf_session()

    print(f"Searching ASF for NISAR RSLC ...")
    print(f"  AOI         : {aoi}")
    print(f"  Max results : {max_results}")
    print(f"  Output dir  : {output_dir}")
    print(f"  (may take 10-30 seconds) ...")

    try:
        results = asf.search(
            platform=asf.PLATFORM.NISAR,
            processingLevel='RSLC',
            intersectsWith=aoi,
            maxResults=max_results,
        )
    except Exception as exc:
        print(f"\nSearch failed: {exc}")
        print("Ensure you have signed into https://search.asf.alaska.edu at least once.")
        raise

    print(f"\nFound {len(results)} granule(s).")
    if not results:
        return []

    for r in results:
        p   = r.properties
        lat = p.get('centerLat') or 0.0
        lon = p.get('centerLon') or 0.0
        print(f"  {p.get('sceneName','?')[:60]:60s}  "
              f"lat={float(lat):.2f}  lon={float(lon):.2f}")

    print(f"\nDownloading {len(results)} file(s) ...")
    print("  (each file is 2-8 GB -- this will take a while)")
    results.download(path=str(output_dir), session=session)
    print("  Download complete.")
    return sorted(output_dir.glob('*.h5'))


# ---------------------------------------------------------------------------
# HDF5 utilities
# ---------------------------------------------------------------------------

def inspect_rslc(h5_path: Path) -> None:
    """Print the internal HDF5 tree of a NISAR RSLC file."""
    def _node(name, obj):
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")
    with h5py.File(str(h5_path), 'r') as f:
        print(f"\n=== {h5_path.name} ===")
        f.visititems(_node)


def list_polarizations(h5_path: Path, frequency: str = 'A') -> list[str]:
    key = f'/science/LSAR/RSLC/swaths/frequency{frequency}'
    with h5py.File(str(h5_path), 'r') as f:
        if key not in f:
            return []
        return [k for k in f[key].keys()
                if isinstance(f[f'{key}/{k}'], h5py.Dataset)]


def load_rslc_frame(
    h5_path: Path,
    frequency: str = 'A',
    polarization: str = 'HH',
    az_end: int | None = None,
    rng_end: int | None = None,
) -> np.ndarray:
    """
    Load a tile from a NISAR RSLC .h5 file as (N_pulses, N_range) complex64.
    Azimuth axis = slow time (pulses), range axis = fast time.
    """
    ds_path = f'/science/LSAR/RSLC/swaths/frequency{frequency}/{polarization}'
    with h5py.File(str(h5_path), 'r') as f:
        if ds_path not in f:
            avail = list_polarizations(h5_path, frequency)
            raise KeyError(
                f"'{polarization}' not in frequency{frequency}. "
                f"Available: {avail}"
            )
        ds = f[ds_path]
        n_az, n_rng = ds.shape
        print(f"  Full shape : ({n_az}, {n_rng})  dtype: {ds.dtype}")
        S = ds[:az_end, :rng_end]
    print(f"  Loaded tile: {S.shape}")
    return S.astype(np.complex64)


def align_frame_to_cpi(
    S: np.ndarray,
    cpi_size: int = 32,
    n_range_bins: int = 512,
) -> np.ndarray:
    """Trim frame so dimensions are divisible by (cpi_size, n_range_bins)."""
    n_az  = S.shape[0] // cpi_size
    n_rng = S.shape[1] // n_range_bins
    trimmed = S[: n_az * cpi_size, : n_rng * n_range_bins]
    print(f"  Aligned : {S.shape} -> {trimmed.shape} "
          f"({n_az} az-CPIs x {n_rng} range-blocks)")
    return trimmed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("fetch_nisar.py -- NISAR RSLC downloader")
    print("=" * 60)

    h5_files = search_and_download(AOI, MAX_RESULTS, DATA_DIR)

    if not h5_files:
        print("\nRetrying with broader South America AOI ...")
        h5_files = search_and_download(AOI_BROAD, MAX_RESULTS, DATA_DIR)

    if not h5_files:
        print("\nNo data found. Check Vertex manually:")
        print("  https://search.asf.alaska.edu")
        print("  Dataset=NISAR, Product Type=RSLC, draw polygon over S. America")
        sys.exit(1)

    # Inspect first file and extract tile
    inspect_rslc(h5_files[0])

    pols = list_polarizations(h5_files[0], FREQUENCY)
    print(f"\nAvailable polarizations (frequency{FREQUENCY}): {pols}")

    print(f"\nExtracting tile from {h5_files[0].name} ...")
    S = load_rslc_frame(
        h5_files[0],
        frequency=FREQUENCY,
        polarization=POLARIZATION,
        az_end=LOAD_AZ_LINES,
        rng_end=LOAD_RNG_SAMPLES,
    )

    S_aligned = align_frame_to_cpi(S, cpi_size=32, n_range_bins=512)

    npy_path = DATA_DIR / 'frame_000_aligned.npy'
    np.save(str(npy_path), S_aligned)
    print(f"\nSaved : {npy_path}")
    print(f"Shape : {S_aligned.shape}  dtype: {S_aligned.dtype}")
    print("\nNext: python nisar/render_nisar.py")