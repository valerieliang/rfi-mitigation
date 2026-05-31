"""
nisar/fetch_nisar.py -- Search and download NISAR RSLC data from ASF Vertex.

Run from the project root:
    python nisar/fetch_nisar.py

Downloads .h5 files to data/nisar_rslc/ and saves one aligned .npy tile
per granule, named frame_NNN_aligned.npy, ready for rfi_gen.overlay.

No synthetic fallback -- all training data must come from real satellite
acquisitions.

Configuration: edit the CONFIG block at the top of this file.
Credentials:   set EARTHDATA_TOKEN in .env (see .env.example).

Key fixes vs. previous version
-------------------------------
1. Search now uses shortName='NISAR_L1_RSLC_BETA_V1' + dataset=DATASET.NISAR.
   Using processingLevel='RSLC' alone returns zero results from CMR because
   NISAR CMR attributes are not searchable as additional attributes.
2. Authenticated session is threaded into asf.search() (not just download()).
   NISAR granule enumeration in CMR requires authentication; passing only
   to download() caused silent empty result sets with the old code.
3. Multi-file pipeline: every downloaded .h5 gets its own frame_NNN_aligned.npy.
   Previous code silently discarded granules 1..N-1.
4. AOI covers the full Amazon basin interior with a denser polygon to reduce
   edge-of-swath artifacts in the search geometry intersection test.
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
# CONFIG
# ---------------------------------------------------------------------------

# Primary AOI: Amazon basin interior -- clean L-band, low anthropogenic RFI.
# Denser polygon reduces false positives from bounding-box edge tiles.
AOI = (
    'POLYGON(('
    '-75 -10, -60 -10, -50 -5, -50 5, -60 5, -75 0, -75 -10'
    '))'
)

# Broader fallback if primary AOI returns zero results
AOI_BROAD = 'POLYGON((-82 -56, -34 -56, -34 13, -82 13, -82 -56))'

# NISAR short name for the L1 RSLC beta product.
# FIX: was using processingLevel='RSLC' which CMR does not index for NISAR.
RSLC_SHORT_NAME = 'NISAR_L1_RSLC_BETA_V1'

MAX_RESULTS   = 10          # granules to download (each is 2-8 GB)
FREQUENCY     = 'A'         # 'A' = 40 MHz L-band primary band
POLARIZATION  = 'HH'        # channel to extract for the .npy tile

# Tile size to extract from each .h5 for training
LOAD_AZ_LINES    = 32000    # ~1000 CPIs at M=32
LOAD_RNG_SAMPLES = 5120     # 10 range blocks at K=512

DATA_DIR = PROJECT_ROOT / 'data' / 'nisar_rslc'


# ---------------------------------------------------------------------------
# Search and download
# ---------------------------------------------------------------------------

def search_and_download(aoi: str, max_results: int, output_dir: Path) -> list[Path]:
    """
    Search ASF CMR for NISAR RSLC granules intersecting aoi, download them.

    FIX: session is now passed to asf.search() as well as download().
    NISAR granule enumeration in CMR requires authentication; passing only
    to download() caused silent empty result sets with the old code.

    FIX: search uses shortName + dataset instead of processingLevel='RSLC'.
    CMR does not expose NISAR processing level as a searchable attribute.
    """
    try:
        import asf_search as asf
    except ImportError as exc:
        raise ImportError("pip install asf_search") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build authenticated session once; reuse for both search and download.
    session = make_asf_session()

    print(f"Searching ASF for NISAR RSLC ...")
    print(f"  Short name  : {RSLC_SHORT_NAME}")
    print(f"  AOI         : {aoi}")
    print(f"  Max results : {max_results}")
    print(f"  Output dir  : {output_dir}")
    print(f"  (may take 10-30 seconds) ...")

    try:
        # FIX: session belongs in ASFSearchOptions, not as a kwarg to search().
        # search() raises TypeError if session is passed directly.
        opts = asf.ASFSearchOptions(
            shortName=RSLC_SHORT_NAME,
            dataset=asf.DATASET.NISAR,
            intersectsWith=aoi,
            maxResults=max_results,
            session=session,
        )
        results = asf.search(opts=opts)
    except Exception as exc:
        print(f"\nSearch failed: {exc}")
        print("Ensure you have signed into https://search.asf.alaska.edu at least once.")
        raise

    print(f"\nFound {len(results)} granule(s).")
    if not results:
        return []

    for r in results:
        p = r.properties
        # geometry centroid is more reliable than centerLat/centerLon for
        # NISAR products whose CMR records omit those scalar fields.
        try:
            coords = r.geometry.get('coordinates', [[]])[0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            clat = sum(lats) / len(lats) if lats else 0.0
            clon = sum(lons) / len(lons) if lons else 0.0
        except Exception:
            clat = float(p.get('centerLat') or 0.0)
            clon = float(p.get('centerLon') or 0.0)

        print(
            f"  {str(p.get('sceneName', '?'))[:60]:60s}  "
            f"lat={clat:.2f}  lon={clon:.2f}"
        )

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
    print(
        f"  Aligned : {S.shape} -> {trimmed.shape} "
        f"({n_az} az-CPIs x {n_rng} range-blocks)"
    )
    return trimmed


# ---------------------------------------------------------------------------
# Per-granule extraction pipeline
# FIX: previous __main__ only processed h5_files[0], discarding the rest.
# This function is called for every downloaded .h5 so all granules are saved.
# ---------------------------------------------------------------------------

def extract_and_save(h5_path: Path, frame_idx: int) -> Path | None:
    """
    Extract an aligned CPI tile from one RSLC .h5 and save as .npy.

    Returns the output path on success, or None if the polarization is missing.
    """
    print(f"\n[{frame_idx:03d}] Processing {h5_path.name} ...")

    pols = list_polarizations(h5_path, FREQUENCY)
    print(f"  Available polarizations (frequency{FREQUENCY}): {pols}")

    if POLARIZATION not in pols:
        fallback = pols[0] if pols else None
        if fallback is None:
            print(f"  No polarizations found -- skipping.")
            return None
        print(f"  {POLARIZATION} not available; using {fallback} instead.")
        pol = fallback
    else:
        pol = POLARIZATION

    try:
        S = load_rslc_frame(
            h5_path,
            frequency=FREQUENCY,
            polarization=pol,
            az_end=LOAD_AZ_LINES,
            rng_end=LOAD_RNG_SAMPLES,
        )
    except Exception as exc:
        print(f"  Load failed: {exc} -- skipping.")
        return None

    S_aligned = align_frame_to_cpi(S, cpi_size=32, n_range_bins=512)

    npy_path = DATA_DIR / f'frame_{frame_idx:03d}_aligned.npy'
    np.save(str(npy_path), S_aligned)
    print(f"  Saved : {npy_path}  shape={S_aligned.shape}  dtype={S_aligned.dtype}")
    return npy_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("fetch_nisar.py -- NISAR RSLC Amazon basin scanner")
    print("=" * 60)

    h5_files = search_and_download(AOI, MAX_RESULTS, DATA_DIR)

    if not h5_files:
        print("\nNo results for primary Amazon AOI. Retrying with broader S. America AOI ...")
        h5_files = search_and_download(AOI_BROAD, MAX_RESULTS, DATA_DIR)

    if not h5_files:
        print("\nNo NISAR RSLC data found. Check Vertex manually:")
        print("  https://search.asf.alaska.edu")
        print("  Dataset=NISAR, Short Name=NISAR_L1_RSLC_BETA_V1")
        print("  Draw polygon over Amazon basin, e.g. -75W to -50W, -10S to +5N")
        sys.exit(1)

    # Inspect the first file so the user can verify the HDF5 tree
    inspect_rslc(h5_files[0])

    # FIX: iterate all downloaded files, not just index 0
    saved = []
    for idx, h5_path in enumerate(h5_files):
        result = extract_and_save(h5_path, frame_idx=idx)
        if result is not None:
            saved.append(result)

    print(f"\n{'=' * 60}")
    print(f"Done. Saved {len(saved)} / {len(h5_files)} tiles:")
    for p in saved:
        print(f"  {p}")
    print("\nNext: python nisar/render_nisar.py")