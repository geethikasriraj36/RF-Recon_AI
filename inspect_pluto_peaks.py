
from pathlib import Path
import numpy as np
from scipy.signal import periodogram, find_peaks

capture = Path("real_dataset/pluto_rx_smoke_test.dat")
sample_rate = 30_720_000
center_frequency = 2_400_000_000

iq = np.fromfile(capture, dtype=np.complex64)

freq, psd = periodogram(
    iq,
    fs=sample_rate,
    window="hann",
    detrend=False,
    return_onesided=False,
    scaling="density"
)

freq = np.fft.fftshift(freq)
psd = np.fft.fftshift(psd)

psd_db = 10 * np.log10(np.maximum(psd, 1e-30))

# Ignore the very center region to reduce DC-spike interference.
valid = np.abs(freq) > 100_000
indices, _ = find_peaks(psd_db, distance=20, prominence=3)
indices = [i for i in indices if valid[i]]
indices = sorted(indices, key=lambda i: psd_db[i], reverse=True)

print("Strongest spectral peaks:")
for i in indices[:10]:
    absolute_mhz = (center_frequency + freq[i]) / 1e6
    print(
        f"{absolute_mhz:.6f} MHz | "
        f"offset {freq[i] / 1e3:+.2f} kHz | "
        f"{psd_db[i]:.2f} dB relative"
    )
