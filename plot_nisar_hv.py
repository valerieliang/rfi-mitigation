import h5py
import numpy as np
from matplotlib import pyplot as plt

HDF5_PATH = r'nisar_data\raw'  # update to full filename if needed
DATASET_PATH = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'

with h5py.File(HDF5_PATH, 'r') as f:
    dataset = f[DATASET_PATH]
    print("Dataset shape :", dataset.shape)
    print("Data type     :", dataset.dtype)
    print("Chunks        :", dataset.chunks)

    # Read the full array (slow_time x range_samples)
    matrix = dataset[:]

# NISAR L0B is complex I/Q -- compute power in dB
# shape is typically (num_pulses, num_range_samples)
num_pulses, num_range = matrix.shape

# Normalize by range dimension (consistent with your existing convention)
matrix_pwr_db = 10 * np.log10(np.abs(matrix) ** 2 / num_range + 1e-30)

# --- full-frame image ---
plt.figure(figsize=(12, 6))
plt.imshow(
    matrix_pwr_db,
    aspect='auto',
    cmap='gray',
    origin='lower',
    vmin=np.percentile(matrix_pwr_db, 2),
    vmax=np.percentile(matrix_pwr_db, 98),
)
plt.colorbar(label='Power (dB)')
plt.xlabel('Range Sample Index')
plt.ylabel('Pulse (Slow-Time) Index')
plt.title('NISAR L0B  HV  Power (dB)')
plt.tight_layout()
plt.savefig('nisar_hv_power.png', dpi=150)
plt.show()

# --- range power spectrum of a single pulse (sanity check) ---
mid_pulse = num_pulses // 2
pulse_fft = np.fft.fftshift(np.fft.fft(matrix[mid_pulse, :]))
pulse_psd = 10 * np.log10(np.abs(pulse_fft) ** 2 + 1e-30)

plt.figure(figsize=(10, 4))
plt.plot(pulse_psd)
plt.xlabel('Frequency Bin')
plt.ylabel('Power (dB)')
plt.title(f'Range Frequency Spectrum  --  Pulse {mid_pulse}')
plt.tight_layout()
plt.savefig('nisar_hv_range_spectrum.png', dpi=150)
plt.show()