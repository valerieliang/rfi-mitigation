import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HDF5_PATH = './nisar_data/raw/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5'
DATASET_PATH = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'

with h5py.File(HDF5_PATH, 'r') as f:
    dataset = f[DATASET_PATH]
    raw = dataset[:]

print("Matrix Shape:", raw.shape)
print("Data Type:", raw.dtype)

# Structured dtype with 'r' and 'i' fields -- assemble as complex
matrix = raw['r'].astype(np.float32) + 1j * raw['i'].astype(np.float32)

pulse_count, range_count = matrix.shape

img_pwr_db = 10 * np.log10(np.abs(matrix)**2 / range_count)

SLICES = [
    (50000,  58000,  'Pulses 50000-58000',   'nisar_hv_mid.png'),
    (170000, 178000, 'Pulses 170000-178000', 'nisar_hv_bottom.png'),
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
    ax.set_ylabel('Pulse Index')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")