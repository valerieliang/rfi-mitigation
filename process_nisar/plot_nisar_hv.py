import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

BLOCKS = [
    ('./nisar_data/processed/block_mid.h5',    'nisar_hv_mid.png'),
    ('./nisar_data/processed/block_bottom.h5', 'nisar_hv_bottom.png'),
]

for block_path, fname in BLOCKS:
    with h5py.File(block_path, 'r') as f:
        block       = f['data'][:]
        pulse_start = f['data'].attrs['pulse_start']
        pulse_end   = f['data'].attrs['pulse_end']

    range_count = block.shape[1]
    pwr_db = 10 * np.log10(np.maximum(np.abs(block)**2 / range_count, 1e-12))

    title = f'Pulses {pulse_start}-{pulse_end}'

    fig, ax = plt.subplots()
    im = ax.imshow(
        pwr_db,
        aspect='auto',
        cmap='gray',
        origin='lower',
    )
    fig.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_title(title)
    ax.set_xlabel('Range Sample')
    ax.set_ylabel('Pulse Index (block-relative)')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}  ({title})")