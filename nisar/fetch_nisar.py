#!/usr/bin/env python3
"""
fetch_nisar.py

Search the NASA Earthdata / ASF archive for NISAR RSLC (Range-Doppler
Single Look Complex) products over a polygon region and optionally
download them.

Context for this project (RFI / ST-EVD work):
  - We want relatively RFI-clean L-band scenes as a reference baseline.
    South America is comparatively quiet in L-band, so the default
    polygon is drawn over the central South American interior.
  - Target product: NISAR L1 RSLC, L-band, 40+5 MHz split-spectrum mode
    (Frequency A = 40 MHz main band, Frequency B = 5 MHz auxiliary band).

Auth:
  Reads an Earthdata Login (EDL) bearer token from a .env file
  (variable EARTHDATA_TOKEN). Generate one at:
  https://urs.earthdata.nasa.gov/ -> Generate Token.

About band / mode filtering:
  Every NISAR RSLC product that ASF serves is L-band (L-SAR); the
  S-band instrument is archived elsewhere. So there is no need to
  filter for L-band.

  The 40+5 bandwidth is NOT written into the granule name as a literal
  "40+5" string. The acquisition mode is encoded as a 4-digit "mode
  code" field in the file name, and the true bandwidth lives in the
  HDF5 metadata (science/LSAR/.../acquiredRangeBandwidth). This script
  therefore decodes the mode code for every granule and prints a tally,
  so you can identify which mode code corresponds to 40+5 empirically
  (confirm once by opening a single downloaded .h5 file). Once you know
  the code, filter for it with, e.g.:  --name-contains _2005_

Reference: https://search.asf.alaska.edu/  and  asf_search docs.

Screen-and-keep mode (--screen-and-keep):
  Instead of bulk downloading, fetch candidates one at a time, run the
  RFI cleanliness screen from check_nisar_clean.py on each, keep the
  clean ones and discard the rest, stopping once --target-clean clean
  scenes are collected. A scene must be downloaded before it can be
  screened (cleanliness is judged from the focused pixel data, which is
  inside the multi-GB file; there is no API field for it), so this saves
  disk and manual effort rather than bandwidth. Rejected files are
  deleted immediately unless --keep-rejected is set. 

  Works with auto-search (--max-results 0) to find all matching granules
  before screening. Example:

    python fetch_nisar.py --screen-and-keep --target-clean 25 \\
      --name-contains _2005_ --max-results 0

  check_nisar_clean.py must be in the same directory as this script.

Auto-search (--max-results 0):
  Instead of returning a fixed number of granules, iteratively double the
  search limit until the yield drops below the request (indicating the
  catalog end), collecting all unique matching granules. Useful when you
  want every granule matching a specific mode code or other filter, not
  just the first N. Takes longer but exhaustive. Pair with
  --screen-and-keep to automatically collect a target number of clean
  scenes from the full matching set."""

import argparse
import collections
import csv
import json
import os
import sys

try:
    import asf_search as asf
except ImportError:
    sys.exit(
        "asf_search is not installed. Install it with:\n"
        "    pip install asf-search python-dotenv"
    )

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit(
        "python-dotenv is not installed. Install it with:\n"
        "    pip install python-dotenv"
    )

# The cleanliness screen lives in check_nisar_clean.py (a sibling file).
# It is only needed for --screen-and-keep, so the import is attempted
# lazily there rather than at module load, keeping plain search/download
# working even if that file is absent.
def _import_assess_file():
    """Import assess_file from check_nisar_clean, however it is located."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from check_nisar_clean import assess_file
        return assess_file
    except ImportError as exc:
        sys.exit(
            "Could not import assess_file from check_nisar_clean.py.\n"
            "Make sure check_nisar_clean.py is in the same directory as "
            "this script.\nOriginal error: {}".format(exc))


# ----------------------------------------------------------------------
# Default area of interest: a polygon over the central Amazon rainforest
# basin (roughly western Brazil across to central Brazil, southern Peru
# / Colombia border up into the northern basin). The deep Amazon
# interior is among the most RFI-quiet L-band regions on Earth: sparse
# population, few ground-based radars, and little high-power emission in
# the NISAR passband. That makes it a strong clean reference for
# eigenvalue / threshold studies.
#
# Box: lon -72 to -52, lat -9 to +3.
#
# WKT uses (lon lat) ordering and must be closed (first point == last).
# Adjust freely, or pass your own with --wkt.
# ----------------------------------------------------------------------
DEFAULT_WKT = (
    "POLYGON(("
    "-72.0 -9.0,"
    "-52.0 -9.0,"
    "-52.0 3.0,"
    "-72.0 3.0,"
    "-72.0 -9.0"
    "))"
)

# Fallback short names for NISAR L1 RSLC. The dataset + processing-level
# route is preferred; these are tried only if that yields nothing.
RSLC_SHORTNAMES = [
    "NISAR_L1_PR_RSLC_V1",
    "NISAR_L1_RSLC_BETA_V1",
]


def load_token(env_path):
    """Load the EDL bearer token from the given .env file."""
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("EARTHDATA_TOKEN")
    if not token:
        sys.exit(
            "EARTHDATA_TOKEN not found.\n"
            "Create a .env file containing:\n"
            "    EARTHDATA_TOKEN=your_edl_token_here\n"
            "Get a token at https://urs.earthdata.nasa.gov (Generate Token)."
        )
    return token.strip()


def make_session(token):
    """Build an authenticated ASF session from an EDL token."""
    try:
        session = asf.ASFSession().auth_with_token(token)
    except asf.ASFAuthenticationError as exc:
        sys.exit("Earthdata authentication failed: {}".format(exc))
    return session


def search_rslc_all(wkt, start, end, session):
    """
    Auto-search: iteratively raise the result limit until the yield drops,
    collecting all unique granules in the catalog matching the criteria.
    
    Start with 500, double each iteration. Stop when a search returns
    fewer results than requested (meaning we've found them all).
    """
    all_results = {}  # keyed by granule name to avoid duplicates
    limit = 500
    iteration = 0
    
    print("Auto-searching for all matching granules...")
    
    while True:
        iteration += 1
        print("  Attempt {}: searching for up to {} granules...".format(
            iteration, limit))
        
        common = dict(
            intersectsWith=wkt,
            maxResults=limit,
        )
        if start:
            common["start"] = start
        if end:
            common["end"] = end
        
        opts = asf.ASFSearchOptions(
            dataset=asf.DATASET.NISAR,
            processingLevel=["RSLC"],
            **common,
        )
        results = asf.search(opts=opts)
        
        if len(results) == 0 and iteration == 1:
            # Try fallback short names on first attempt only.
            print("    No results via dataset+level; trying short names...")
            opts = asf.ASFSearchOptions(shortName=RSLC_SHORTNAMES, **common)
            results = asf.search(opts=opts)
        
        # Collect unique by granule name to avoid duplicates.
        for product in results:
            gname = granule_name(product)
            all_results[gname] = product
        
        n_this = len(results)
        n_total = len(all_results)
        print("    got {} results (total unique: {})".format(n_this, n_total))
        
        if n_this < limit:
            # Yield dropped below the request, so we've hit the catalog
            # end. Stop.
            print("  Complete: {} total unique granules found.".format(
                n_total))
            break
        
        # Yield met or exceeded the request; there may be more.
        limit = min(limit * 2, 10000)
        if limit == 10000 and n_this == limit:
            print("  Hit safety limit (10000 max_results); stopping. "
                  "Consider narrowing --wkt or date range.")
            break
    
    return asf.ASFSearchResults(list(all_results.values()))


def search_rslc(wkt, start, end, max_results, session):
    """
    Search NISAR RSLC products intersecting the polygon.

    If max_results is 0, auto-search: iteratively double the limit until
    the yield drops (indicating catalog end), collecting all unique
    granules. Otherwise, search once with the given limit.

    Tries the dataset + processing-level route first (preferred), then
    falls back to explicit short names if nothing comes back.
    """
    if max_results == 0:
        return search_rslc_all(wkt, start, end, session)
    
    common = dict(
        intersectsWith=wkt,
        maxResults=max_results,
    )
    if start:
        common["start"] = start
    if end:
        common["end"] = end

    opts = asf.ASFSearchOptions(
        dataset=asf.DATASET.NISAR,
        processingLevel=["RSLC"],
        **common,
    )
    results = asf.search(opts=opts)

    if len(results) == 0:
        print("No hits via dataset+level; retrying with short names...")
        opts = asf.ASFSearchOptions(shortName=RSLC_SHORTNAMES, **common)
        results = asf.search(opts=opts)

    return results


def parse_granule_name(name):
    """
    Decode positional fields from a NISAR RSLC granule name.

    Example:
      NISAR_L1_PR_RSLC_087_039_D_114_2005_DHDH_A_20251102T222008_...
    After the RSLC token the order is:
      cycle, track, direction, frame, mode_code, polarization.
    Returns a dict of strings (values may be None if the pattern does
    not match, e.g. for an unexpected name layout).
    """
    fields = {"cycle": None, "track": None, "direction": None,
              "frame": None, "mode_code": None, "pol": None}
    if not name:
        return fields
    tokens = name.split("_")
    try:
        i = tokens.index("RSLC")
        fields["cycle"] = tokens[i + 1]
        fields["track"] = tokens[i + 2]
        fields["direction"] = tokens[i + 3]
        fields["frame"] = tokens[i + 4]
        fields["mode_code"] = tokens[i + 5]
        fields["pol"] = tokens[i + 6]
    except (ValueError, IndexError):
        pass
    return fields


def granule_name(product):
    """Best-available name string for a product."""
    return (product.properties.get("fileID")
            or product.properties.get("sceneName") or "")


def keep_by_name(product, substrings):
    """Keep product only if its name contains every given substring."""
    if not substrings:
        return True
    name = granule_name(product).upper()
    return all(s.upper() in name for s in substrings)


def summarize(results):
    """Return a list of plain dicts describing each product."""
    rows = []
    for p in results:
        props = p.properties
        name = granule_name(p)
        decoded = parse_granule_name(name)
        rows.append({
            "name": name,
            "mode_code": decoded["mode_code"],
            "pol": decoded["pol"],
            "track": decoded["track"],
            "frame": decoded["frame"],
            "direction": decoded["direction"],
            "start": props.get("startTime"),
            "stop": props.get("stopTime"),
            "size_mb": props.get("bytes"),
            "url": props.get("url"),
        })
    return rows


def print_mode_tally(rows):
    """Print how many granules fall under each decoded mode code / pol."""
    modes = collections.Counter(r["mode_code"] for r in rows)
    pols = collections.Counter(r["pol"] for r in rows)
    print("\nMode code distribution (4-digit field after RSLC):")
    for code, n in sorted(modes.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print("  mode {:>6}  ->  {} granule(s)".format(str(code), n))
    print("Polarization distribution:")
    for pol, n in sorted(pols.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print("  pol  {:>6}  ->  {} granule(s)".format(str(pol), n))
    print("Tip: to confirm which mode code is 40+5 MHz, download one "
          "granule and read science/LSAR/.../acquiredRangeBandwidth in "
          "the .h5 file, then filter with --name-contains _<code>_\n")


def select_diverse(products, n, prefer_pol=None):
    """
    Pick a diverse, low-redundancy subset of n products.

    Why this exists: a raw catalog query returns many consecutive frames
    from the same track/pass (adjacent tiles of one strip). Downloading
    20 in a row gives near-duplicate data. This selector instead:

      1. Groups products into "passes" keyed by (track, acquisition date).
      2. Picks ONE representative frame per pass (the middle frame, which
         tends to be the most fully-illuminated), preferring a chosen
         polarization (e.g. DHDH dual-pol) when available in that pass.
      3. Round-robins across tracks so the final set spans as many
         distinct tracks (i.e. ground locations) as possible.

    Important: this maximizes spatial / acquisition diversity. It does
    NOT rank RFI cleanliness, which can only be judged from the actual
    eigenvalue profiles after download. Treat the result as a good set
    to download and then inspect.

    Returns (chosen_products, chosen_records) where each record is a dict
    with name/track/date/frame/pol for reporting.
    """
    def frame_num(rec):
        try:
            return int(rec["frame"])
        except (TypeError, ValueError):
            return 0

    # Build per-product records.
    recs = []
    for p in products:
        name = granule_name(p)
        decoded = parse_granule_name(name)
        date = (p.properties.get("startTime") or "")[:10]
        recs.append({
            "p": p,
            "name": name,
            "track": decoded["track"],
            "frame": decoded["frame"],
            "pol": decoded["pol"],
            "date": date,
        })

    # Group into passes: same track AND same acquisition date.
    passes = collections.OrderedDict()
    for r in recs:
        passes.setdefault((r["track"], r["date"]), []).append(r)

    # One representative per pass: prefer pol, then take the middle frame.
    reps = []
    for group in passes.values():
        candidates = group
        if prefer_pol:
            pref = [g for g in group
                    if (g["pol"] or "").upper() == prefer_pol.upper()]
            if pref:
                candidates = pref
        candidates = sorted(candidates, key=frame_num)
        reps.append(candidates[len(candidates) // 2])

    # Round-robin across tracks for maximum spatial spread.
    by_track = collections.OrderedDict()
    for r in sorted(reps, key=lambda x: (str(x["track"]), str(x["date"]))):
        by_track.setdefault(r["track"], []).append(r)

    queues = [list(v) for v in by_track.values()]
    chosen = []
    while len(chosen) < n:
        progressed = False
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop(0))
                progressed = True
        if not progressed:
            break  # all passes exhausted before reaching n

    return [r["p"] for r in chosen], chosen


def write_outputs(results, rows, out_dir, prefix="nisar_rslc"):
    """Write a GeoJSON footprint file and a CSV metadata summary."""
    os.makedirs(out_dir, exist_ok=True)

    geojson_path = os.path.join(out_dir, "{}_footprints.geojson".format(prefix))
    with open(geojson_path, "w") as f:
        f.write(json.dumps(results.geojson(), indent=2))

    csv_path = os.path.join(out_dir, "{}_results.csv".format(prefix))
    fields = ["name", "mode_code", "pol", "track", "frame", "direction",
              "start", "stop", "size_mb", "url"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return geojson_path, csv_path


def download_one(product, dest_dir, session):
    """
    Download a single product into dest_dir and return its local path,
    or None on failure. Works whether the product exposes a single URL
    or several files (picks the .h5).
    """
    before = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()
    try:
        # asf_search products download via the same .download API; wrap a
        # single product in a results container so one call suffices.
        asf.ASFSearchResults([product]).download(
            path=dest_dir, session=session, processes=1)
    except Exception as exc:
        print("    download failed: {}".format(exc))
        return None
    after = set(os.listdir(dest_dir))
    new = [f for f in (after - before) if f.lower().endswith(".h5")]
    if not new:
        # Some products may already be present; fall back to name match.
        name = granule_name(product)
        cand = [f for f in after
                if f.lower().endswith(".h5") and name.split("_")[0] in f]
        new = cand
    if not new:
        return None
    # If several, take the largest (the full image, not an aux file).
    paths = [os.path.join(dest_dir, f) for f in new]
    paths.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return paths[0]


def _purge_dir(path):
    """Delete every file in a directory (used to clear staging)."""
    if not os.path.isdir(path):
        return
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except OSError:
            pass


def screen_and_keep(products, args, session):
    """
    Download candidates one at a time, screen each for RFI cleanliness,
    keep the clean ones and discard the rest, until --target-clean clean
    scenes are collected (or the candidate list / optional --max-attempts
    cap is exhausted).

    This is the integrated fetch + check + keep loop. Because cleanliness
    can only be judged from the focused pixel data, each candidate must
    be downloaded before it can be screened; rejected files are deleted
    immediately (unless --keep-rejected) so disk use stays bounded.

    Interrupt safety: every download lands in a private staging
    directory and is only promoted to the keep directory AFTER it passes
    the screen. If the run is interrupted (Ctrl-C) or fails partway, the
    staging directory is purged on exit, so a half-finished multi-GB
    download can never be left behind to masquerade as a real scene. The
    manifest gathered so far is still returned and written.

    Returns a list of manifest rows (one per attempted scene).
    """
    assess_file = _import_assess_file()

    keep_dir = args.out_dir
    stage_dir = os.path.join(args.out_dir, "_staging")
    qc_dir = os.path.join(args.out_dir, "_qc")
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    # Clear any leftovers from a previously interrupted run before we
    # start, so stale partial files never get screened.
    _purge_dir(stage_dir)

    # Which screen verdicts count as "keep". CLEAN always; REVIEW only if
    # the user opts in (those are the borderline single-tone cases).
    accept = {"CLEAN"}
    if args.accept_review:
        accept.add("REVIEW")

    manifest = []
    n_clean = 0
    n_attempt = 0
    # An optional hard cap; 0 means "keep going until the target is met
    # or candidates run out".
    max_attempts = args.max_attempts if args.max_attempts > 0 else None
    cap_str = str(max_attempts) if max_attempts else "no cap"

    print("\nScreen-and-keep: collecting {} clean scene(s), accepting {} "
          "({}).".format(args.target_clean, "/".join(sorted(accept)),
                         cap_str))
    print("Safe to stop anytime with Ctrl-C; the in-progress download is "
          "removed and progress so far is saved.")

    interrupted = False
    try:
        for product in products:
            if n_clean >= args.target_clean:
                break
            if max_attempts is not None and n_attempt >= max_attempts:
                print("Reached attempt cap ({}).".format(max_attempts))
                break
            n_attempt += 1
            name = granule_name(product)
            print("\n[try {}, {}/{} clean] {}".format(
                n_attempt, n_clean, args.target_clean, name))

            # Download into staging. A Ctrl-C during this call raises
            # KeyboardInterrupt, caught below, and the finally-block
            # purges whatever partial file landed in staging.
            path = download_one(product, stage_dir, session)
            if path is None:
                print("    skipped (no file)")
                manifest.append({"name": name, "flag": "DOWNLOAD_FAIL",
                                 "kept": 0, "reasons": "download failed"})
                continue

            size_mb = os.path.getsize(path) / 1e6
            print("    downloaded {:.0f} MB; screening...".format(size_mb))

            try:
                row = assess_file(path, qc_dir, args.max_dim, args.freq)
            except Exception as exc:
                print("    screen error: {}".format(exc))
                row = {"file": os.path.basename(path), "flag": "ERROR",
                       "reasons": str(exc)}

            flag = row.get("flag", "ERROR")
            reasons = row.get("reasons", "")
            print("    verdict: {} ({})".format(flag, reasons))

            if flag in accept:
                final = os.path.join(keep_dir, os.path.basename(path))
                if os.path.abspath(final) != os.path.abspath(path):
                    os.replace(path, final)
                n_clean += 1
                kept = 1
                print("    KEPT  ->  {}/{} clean so far".format(
                    n_clean, args.target_clean))
            else:
                kept = 0
                if not args.keep_rejected:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    print("    discarded (freed {:.0f} MB)".format(size_mb))
                else:
                    rej = os.path.join(keep_dir, "_rejected")
                    os.makedirs(rej, exist_ok=True)
                    os.replace(path, os.path.join(rej,
                                                  os.path.basename(path)))
                    print("    rejected; moved to _rejected/ "
                          "(--keep-rejected)")

            row.update({"name": name, "kept": kept})
            manifest.append(row)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted by user. Cleaning up any partial download "
              "and saving progress...")
    finally:
        # Always clear staging: removes a partially downloaded file from
        # an interrupt, and tidies a clean finish.
        _purge_dir(stage_dir)
        try:
            os.rmdir(stage_dir)
        except OSError:
            pass

    print("\nScreen-and-keep {}: {} clean kept out of {} attempted."
          .format("stopped" if interrupted else "done",
                  n_clean, n_attempt))
    if not interrupted and n_clean < args.target_clean:
        print("WARNING: target of {} not reached (candidates exhausted). "
              "Widen the search: raise --max-results, broaden --wkt or the "
              "date range, or allow --accept-review."
              .format(args.target_clean))
    return manifest


def write_manifest(manifest, out_dir, prefix="nisar_screen_manifest"):
    """Write the screen-and-keep manifest to CSV."""
    if not manifest:
        return None
    # Union of keys across rows so partial rows still serialize.
    cols = []
    for row in manifest:
        for k in row:
            if k not in cols:
                cols.append(k)
    path = os.path.join(out_dir, prefix + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in manifest:
            w.writerow(row)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NISAR L-band RSLC products over a polygon "
                    "from the ASF / Earthdata archive."
    )
    parser.add_argument("--env", default=".env",
                        help="Path to .env holding EARTHDATA_TOKEN.")
    parser.add_argument("--wkt", default=DEFAULT_WKT,
                        help="Search polygon as WKT (lon lat, closed ring). "
                             "Defaults to a clean central Amazon region.")
    parser.add_argument("--start", default=None,
                        help="Start date/time, e.g. 2025-01-01. "
                             "Omit to search the whole archive.")
    parser.add_argument("--end", default=None,
                        help="End date/time, e.g. 2025-12-31. "
                             "Omit to search the whole archive.")
    parser.add_argument("--max-results", type=int, default=100,
                        help="Maximum number of granules to return from search "
                             "(0 = auto-search: iteratively double the limit "
                             "until the yield drops, collecting all matching "
                             "granules. Auto mode takes longer but finds "
                             "everything; use it with --name-contains filters "
                             "when you want all granules matching a mode code).")
    parser.add_argument("--out-dir", default="nisar_out",
                        help="Directory for result metadata and downloads.")
    parser.add_argument("--name-contains", action="append", default=[],
                        metavar="SUBSTR",
                        help="Keep only granules whose name contains this "
                             "substring (case-insensitive). Repeatable; all "
                             "must match. Example: --name-contains _2005_ "
                             "to keep one mode code, or --name-contains DHDH "
                             "for a polarization.")
    parser.add_argument("--select", type=int, default=None, metavar="N",
                        help="Reduce to a diverse, low-redundancy subset of "
                             "N granules (one representative frame per "
                             "track/date pass, spread across tracks). Use "
                             "this to pick, e.g., the best 20 to download "
                             "and then inspect. Note: this maximizes spatial "
                             "diversity, not RFI cleanliness.")
    parser.add_argument("--prefer-pol", default="DHDH", metavar="POL",
                        help="When selecting, prefer this polarization within "
                             "each pass if available (default: DHDH dual-pol). "
                             "Set to empty string to disable the preference.")
    parser.add_argument("--download", action="store_true",
                        help="Download matched granules (large HDF5 files). "
                             "When --select is set, only the subset is "
                             "downloaded.")
    parser.add_argument("--processes", type=int, default=4,
                        help="Parallel download workers when --download is set.")

    # Integrated fetch + screen + keep mode.
    parser.add_argument("--screen-and-keep", action="store_true",
                        help="Download candidates one at a time, screen each "
                             "for RFI cleanliness, keep the clean ones and "
                             "discard the rest until --target-clean is met. "
                             "Requires check_nisar_clean.py alongside this "
                             "script.")
    parser.add_argument("--target-clean", type=int, default=20, metavar="N",
                        help="Screen-and-keep: how many clean scenes to "
                             "collect, then stop. This is the main control: "
                             "the run continues until N clean scenes are "
                             "kept or candidates run out.")
    parser.add_argument("--max-attempts", type=int, default=0, metavar="N",
                        help="Screen-and-keep: optional cap on how many "
                             "scenes to download and screen (0 = no cap, "
                             "run until --target-clean is met). Use it only "
                             "to bound bandwidth on a poor-yield region.")
    parser.add_argument("--accept-review", action="store_true",
                        help="Screen-and-keep: also keep REVIEW-flagged "
                             "scenes (borderline single-tone cases), not "
                             "only CLEAN.")
    parser.add_argument("--keep-rejected", action="store_true",
                        help="Screen-and-keep: do not delete rejected "
                             "scenes (default deletes them to save disk).")
    parser.add_argument("--max-dim", type=int, default=1600, metavar="N",
                        help="Screen-and-keep: decimate images to about this "
                             "size before screening (passed to the screen).")
    parser.add_argument("--freq", default=None, metavar="A_or_B",
                        help="Screen-and-keep: frequency sub-band to screen "
                             "(default first available).")
    args = parser.parse_args()

    token = load_token(args.env)
    session = make_session(token)

    print("Searching NISAR RSLC over the requested polygon...")
    results = search_rslc(
        wkt=args.wkt,
        start=args.start,
        end=args.end,
        max_results=args.max_results,
        session=session,
    )
    print("Raw granules returned: {}".format(len(results)))

    # Optional, transparent name-substring filter (off unless requested).
    if args.name_contains:
        kept = [p for p in results if keep_by_name(p, args.name_contains)]
        print("After name filter {}: {}".format(args.name_contains, len(kept)))
    else:
        kept = list(results)

    filtered = asf.ASFSearchResults(kept)

    if len(filtered) == 0:
        print("No granules left after filtering. Loosen --name-contains, "
              "widen the dates, or adjust --wkt.")
        return

    rows = summarize(filtered)
    print_mode_tally(rows)

    geojson_path, csv_path = write_outputs(filtered, rows, args.out_dir)
    print("Wrote full footprints: {}".format(geojson_path))
    print("Wrote full metadata:   {}".format(csv_path))

    # Decide what gets downloaded: the full set, or a diverse subset.
    target = filtered
    if args.select is not None:
        prefer = args.prefer_pol if args.prefer_pol else None
        chosen_products, chosen_recs = select_diverse(
            list(filtered), args.select, prefer_pol=prefer)
        target = asf.ASFSearchResults(chosen_products)
        sel_rows = summarize(target)

        sel_geojson, sel_csv = write_outputs(
            target, sel_rows, args.out_dir, prefix="nisar_rslc_selected")
        print("\nSelected {} diverse granule(s) "
              "(spanning {} track/date passes):".format(
                  len(target),
                  len(set((r["track"], r["date"]) for r in chosen_recs))))
        for r in chosen_recs:
            print("  {}  track={} date={} pol={}".format(
                r["name"], r["track"], r["date"], r["pol"]))
        print("Wrote selected footprints: {}".format(sel_geojson))
        print("Wrote selected metadata:   {}".format(sel_csv))
    else:
        for r in rows[:10]:
            print("  {}  mode={} pol={}".format(
                r["name"], r["mode_code"], r["pol"]))
        if len(rows) > 10:
            print("  ... and {} more".format(len(rows) - 10))

    if args.screen_and_keep:
        # For screen-and-keep, use the full candidate list (no diversity
        # selection), because we'll reject ~70% anyway. Diversity selection
        # was meant to avoid downloading redundant near-duplicate frames
        # for bulk download; here we want *all* candidates to maximize
        # yield of clean scenes. Order them by catalog (oldest first) or
        # apply a simple diversity spread if desired for better inter-scene
        # variety, but don't drop candidates.
        candidates = list(filtered)

        manifest = screen_and_keep(candidates, args, session)
        man_path = write_manifest(manifest, args.out_dir)
        if man_path:
            print("Wrote screen manifest: {}".format(man_path))
        n_kept = sum(r.get("kept", 0) for r in manifest)
        print("\nClean scenes are in: {}".format(args.out_dir))
        print("Next: build a training set from them with\n"
              "  python ml/build_training_set.py --qc-csv <your_qc_csv> "
              "--data-dir {} --target-per-band 300 --figure"
              .format(args.out_dir))
        return

    if args.download:
        print("\nDownloading {} granule(s) to {} ...".format(
            len(target), args.out_dir))
        target.download(path=args.out_dir, session=session,
                        processes=args.processes)
        print("Download complete.")
    else:
        print("\nRun again with --download to retrieve the granule files.")


if __name__ == "__main__":
    main()