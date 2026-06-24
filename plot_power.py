import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

BLOCK_PATH = './nisar_data/processed/block_mid.h5'

with h5py.File(BLOCK_PATH, 'r') as f:
    ds          = f['data']
    block       = ds[:]
    pulse_start = int(ds.attrs['pulse_start'])
    range_start = int(ds.attrs['range_start'])

print(f"Block shape : {block.shape}  dtype={block.dtype}")
print(f"True pulse origin : {pulse_start}")

n_pulses, n_range = block.shape

# ── Per-sample power in dB ─────────────────────────────────────────────────
with np.errstate(divide='ignore'):
    pwr_db = 10 * np.log10(np.abs(block)**2 / n_range)

# ── Mean power per pulse (azimuth profile) ─────────────────────────────────
pulse_mean_db = np.nanmean(pwr_db, axis=1)   # (8000,)

# ── Mean power per range sample (range profile) ───────────────────────────
range_mean_db = np.nanmean(pwr_db, axis=0)   # (52866,)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# --- 1. Full power image (subsampled in range for display) ----------------
stride = 10
ax = axes[0]
im = ax.imshow(
    pwr_db[:, ::stride],
    aspect='auto',
    cmap='inferno',
    origin='upper',
    vmin=np.nanpercentile(pwr_db, 2),
    vmax=np.nanpercentile(pwr_db, 98),
)
fig.colorbar(im, ax=ax, label='Power (dB)')
ax.set_title(
    f'block_mid raw power  (pulses {pulse_start}:{pulse_start + n_pulses})\n'
    f'range subsampled x{stride} for display',
    fontweight='bold',
)
ax.set_xlabel('Range sample (subsampled)')
ax.set_ylabel('Pulse index (block-local)')

# Mark CPI boundaries every 16 pulses
for cpi_row in range(0, n_pulses, 16):
    ax.axhline(cpi_row, color='cyan', linewidth=0.3, alpha=0.4)

# --- 2. Mean power vs pulse index (should be flat if no RFI spikes) -------
ax = axes[1]
ax.plot(pulse_mean_db, linewidth=0.6, color='steelblue')
ax.set_xlabel('Pulse index (block-local)')
ax.set_ylabel('Mean power (dB)')
ax.set_title('Mean power per pulse  —  spikes indicate RFI bursts', fontweight='bold')
ax.set_xlim(0, n_pulses)
# Shade every other CPI for readability
for ci in range(0, n_pulses // 16, 2):
    ax.axvspan(ci * 16, (ci + 1) * 16, alpha=0.05, color='orange')

# --- 3. Mean power vs range sample ----------------------------------------
ax = axes[2]
ax.plot(range_mean_db, linewidth=0.5, color='tomato')
ax.set_xlabel('Range sample index')
ax.set_ylabel('Mean power (dB)')
ax.set_title('Mean power per range sample  —  peaks indicate narrowband RFI', fontweight='bold')
ax.set_xlim(0, n_range)

fig.tight_layout()
fig.savefig('block_mid_power.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved block_mid_power.png")
