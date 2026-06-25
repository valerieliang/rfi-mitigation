import h5py
import numpy as np
from matplotlib import pyplot as plt

# Use 'r' before the string to fix the SyntaxWarning invalid escape sequence
with h5py.File(r'data\multi_band\image_0.h5', 'r') as f:
    dataset = f['/cpi_0_0']
    
    # Read the data directly
    matrix = dataset[:]

# Print your results
print("Matrix Shape:", matrix.shape)
print("Data Type:", matrix.dtype)
#print(matrix)

num_rows, num_cols = matrix.shape

matrix_pwr_db = 10 * np.log10(np.abs(matrix)**2/num_cols)

plt.figure()
plt.imshow(matrix_pwr_db, aspect='auto', cmap='grey', origin='lower')
plt.colorbar(label='Power (dB)')
plt.show()

