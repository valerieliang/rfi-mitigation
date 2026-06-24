import os
import h5py
import numpy as np

HDF5_PATH = './nisar_data/raw/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5'
DATASET_PATH = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'
OUT_DIR = './nisar_data/processed'

# Each entry: (pulse_start, pulse_end, range_start, range_end, output_filename)
# Range spans the full swath (0 : range_count) -- set range_start=None to use full width
BLOCKS = [
    (50000,  58000,  None, None, 'block_mid.h5'),
    (170000, 178000, None, None, 'block_bottom.h5'),
]

os.makedirs(OUT_DIR, exist_ok=True)

with h5py.File(HDF5_PATH, 'r') as src:
    raw = src[DATASET_PATH]
    pulse_count, range_count = raw.shape

    for pulse_start, pulse_end, range_start, range_end, fname in BLOCKS:
        # Resolve None range bounds to full width
        rs = range_start if range_start is not None else 0
        re = range_end   if range_end   is not None else range_count

        print(f"Extracting pulses [{pulse_start}:{pulse_end}], "
              f"range [{rs}:{re}] -> {fname}")

        # Read only the required rows from disk
        chunk = raw[pulse_start:pulse_end, rs:re]

        # Assemble complex from structured dtype fields
        block = chunk['r'].astype(np.float32) + 1j * chunk['i'].astype(np.float32)

        out_path = os.path.join(OUT_DIR, fname)
        with h5py.File(out_path, 'w') as dst:
            # Store block origin as scalar attributes on the dataset
            ds = dst.create_dataset(
                'data',
                data=block,
                compression='gzip',
                compression_opts=4,
            )
            ds.attrs['pulse_start']  = pulse_start
            ds.attrs['pulse_end']    = pulse_end
            ds.attrs['range_start']  = rs
            ds.attrs['range_end']    = re
            ds.attrs['source_file']  = HDF5_PATH
            ds.attrs['source_path']  = DATASET_PATH

        print(f"  Saved {out_path}  shape={block.shape}  dtype={block.dtype}")