#!/usr/bin/env python
"""
profile_all_mountains.py

Batch profiler for Czech Republic mountain ranges within the NISAR swath.
Runs mountain_profile.py on each targeted mountain range with appropriate
pulse ranges, organizing outputs into separate subdirectories.

Scene geometry:
  - Pulse 435777-489625
  - Swath runs SW to NE (angled)
  - Range samples: 26750

Mountain ranges (in order from south to north):
  1. Bohemian Forest (Šumava) - southern tri-border area
  2. Bohemian-Moravian Highlands - central interior
  3. Jeseníky Mountains - eastern Czech-Polish border
  4. Giant Mountains (Krkonoše) - highest peaks, western Czech-Polish border
  5. Jizerské Mountains - Czech-Polish border, northeast
"""

import argparse
import subprocess
import sys
import os

# Mountain range definitions: name, pulse_start, pulse_end, description
MOUNTAIN_RANGES = [
    {
        "name": "bohemian_forest",
        "display_name": "Bohemian Forest (Šumava)",
        "pulse_start": 435777,
        "pulse_end": 441000,
        "description": "Southern tri-border area (Czech-German-Austrian), includes Plechý (1,378m)",
    },
    {
        "name": "bohemian_moravian_highlands",
        "display_name": "Bohemian-Moravian Highlands (Českomoravská vrchovina)",
        "pulse_start": 448000,
        "pulse_end": 458000,
        "description": "Central Czech interior, rolling highlands with Javořice (837m)",
    },
    {
        "name": "jeseniky",
        "display_name": "Jeseníky Mountains",
        "pulse_start": 463000,
        "pulse_end": 473000,
        "description": "Eastern Czech-Polish border, includes Praděd (1,491m - second highest)",
    },
    {
        "name": "giant_mountains",
        "display_name": "Giant Mountains (Krkonoše)",
        "pulse_start": 473000,
        "pulse_end": 482000,
        "description": "Western Czech-Polish border, includes Sněžka (1,603m - HIGHEST in Czechia)",
    },
    {
        "name": "jizerske",
        "display_name": "Jizerské Mountains",
        "pulse_start": 480000,
        "pulse_end": 487000,
        "description": "Czech-Polish border northeast, includes Smrk (1,124m)",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch profile Czech Republic mountain ranges within the NISAR swath.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mountain ranges profiled (south to north):
  1. Bohemian Forest (Šumava)           : pulse 435777-441000
  2. Bohemian-Moravian Highlands        : pulse 448000-458000
  3. Jeseníky Mountains                 : pulse 463000-473000
  4. Giant Mountains (Krkonoše)         : pulse 473000-482000
  5. Jizerské Mountains                 : pulse 480000-487000

Examples:
  # Process all mountains
  python profile_all_mountains.py /path/to/scene.h5 --freq A --pol HH

  # Process specific mountains only
  python profile_all_mountains.py scene.h5 --ranges giant_mountains jeseniky

  # Adjust grid plot size
  python profile_all_mountains.py scene.h5 --max-grid 100
        """
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')

    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Frequency to process (default: all frequencies)')
    parser.add_argument('--pol', default=None,
                        help='Polarization to process (default: all polarizations)')

    parser.add_argument('--ranges', nargs='+',
                        choices=[r["name"] for r in MOUNTAIN_RANGES],
                        default=None,
                        help='Specific mountain ranges to process (default: all)')

    parser.add_argument('--output-base', default='results/mountains',
                        help='Base output directory (each range gets a subdirectory)')

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Pass --compute-subswath-mask to mountain_profile.py')

    parser.add_argument('--cpi-len', type=int, default=16,
                        help='CPI length (default: 16)')
    parser.add_argument('--cpi-width', type=int, default=250,
                        help='CPI width (default: 250)')

    parser.add_argument('--max-grid', type=int, default=None,
                        help='Max number of CPIs in grid plot (passed to mountain_profile.py)')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')

    return parser.parse_args()


def build_command(l0b_file, mountain, output_dir, args):
    """Build the mountain_profile.py command for a single mountain range."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "mountain_profile.py"),
        l0b_file,
        "--pulse-start", str(mountain["pulse_start"]),
        "--pulse-end", str(mountain["pulse_end"]),
        "--output-dir", output_dir,
        "--cpi-len", str(args.cpi_len),
        "--cpi-width", str(args.cpi_width),
    ]

    if args.freq:
        cmd.extend(["--freq", args.freq])
    if args.pol:
        cmd.extend(["--pol", args.pol])
    if args.compute_subswath_mask:
        cmd.append("--compute-subswath-mask")

    if args.max_grid is not None:
        cmd.extend(["--max-grid", str(args.max_grid)])

    return cmd


def main():
    args = parse_args()

    # Filter mountain ranges if specific ones requested
    if args.ranges:
        ranges_to_process = [r for r in MOUNTAIN_RANGES if r["name"] in args.ranges]
    else:
        ranges_to_process = MOUNTAIN_RANGES

    print("=" * 80)
    print("Batch Mountain Profile - Czech Republic NISAR Scene")
    print("=" * 80)
    print(f"L0B file: {args.l0b_file}")
    print(f"Frequency: {args.freq or 'all'}")
    print(f"Polarization: {args.pol or 'all'}")
    print(f"Processing {len(ranges_to_process)} mountain range(s)")
    print("=" * 80)
    print()

    success_count = 0
    fail_count = 0

    for i, mountain in enumerate(ranges_to_process, 1):
        print(f"[{i}/{len(ranges_to_process)}] {mountain['display_name']}")
        print(f"    Pulse range: {mountain['pulse_start']}-{mountain['pulse_end']}")
        print(f"    {mountain['description']}")

        output_dir = os.path.join(args.output_base, mountain["name"])
        os.makedirs(output_dir, exist_ok=True)

        cmd = build_command(args.l0b_file, mountain, output_dir, args)

        if args.dry_run:
            print(f"    [DRY RUN] Would execute:")
            print(f"    {' '.join(cmd)}")
            print()
            continue

        print(f"    Output: {output_dir}")
        print(f"    Running mountain_profile.py...")

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"    ✓ SUCCESS")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"    ✗ FAILED (exit code {e.returncode})")
            fail_count += 1
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            fail_count += 1

        print()

    print("=" * 80)
    print(f"Batch processing complete: {success_count} succeeded, {fail_count} failed")
    if not args.dry_run:
        print(f"Results saved to: {args.output_base}")
    print("=" * 80)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
