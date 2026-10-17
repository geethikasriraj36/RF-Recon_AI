
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

capture = Path("real_dataset/pluto_rx_smoke_test.dat")
sample_rate = 30_720_000
center_frequency = 2_400_000_000

iq = np.fromfile(capture, dtype=np.complex64)
if iq.size == 0:
    raise ValueError("Capture is empty.")

freq, psd = periodogram(
    iq,
    fs=sample_rate,
    window="hann",
    detrend=False,
    return_onesided=False,
    scaling="density",
)

freq = np.fft.fftshift(freq)
psd = np.fft.fftshift(psd)

psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
frequency_mhz = (center_frequency + freq) / 1e6

plt.figure(figsize=(12, 5))
plt.plot(frequency_mhz, psd_db, linewidth=0.8)
plt.xlabel("Frequency (MHz)")
plt.ylabel("PSD (dB per Hz, relative sample scale)")
plt.title("RFRecon — PlutoSDR Test Capture Spectrum")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = Path("real_dataset/pluto_smoke_test_spectrum.png")
plt.savefig(output, dpi=160)
print("Saved spectrum plot:", output)
plt.show()
