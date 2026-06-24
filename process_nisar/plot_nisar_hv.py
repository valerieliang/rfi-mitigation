import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

BLOCK_PATH = './nisar_data/processed/block_mid.h5'

with h5py.File(BLOCK_PATH, 'r') as f:
    ds = f['data']
    matrix = ds[:]
    pulse_start = ds.attrs['pulse_start']
    range_start = ds.attrs['range_start']

print("Block Shape:", matrix.shape)
print("Pulse offset:", pulse_start, "  Range offset:", range_start)

pulse_count, range_count = matrix.shape

img_pwr_db = 10 * np.log10(np.abs(matrix)**2 / range_count)

# Plot the first CPI block (pulses 0:32 within the block)
CPI = 32
SLICES = [
    (0, CPI, f'CPI 0 (true pulses {pulse_start}:{pulse_start + CPI})', 'nisar_hv_cpi0.png'),
]

for start, end, title, fname in SLICES:
    fig, ax = plt.subplots()
    im = ax.imshow(
        img_pwr_db[start:end, :],
        aspect='auto',
        cmap='gray',
        origin='lower',
    )
    fig.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_title(title)
    ax.set_xlabel('Range Sample')
    ax.set_ylabel('Pulse Index (block-local)')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")