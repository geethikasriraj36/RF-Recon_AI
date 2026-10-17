import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

FILE = "real_wifi_capture_02.dat"
FS = 20_000_000

x = np.fromfile(FILE, dtype=np.complex64)

# Use a manageable portion
x = x[:1_000_000]

# Welch PSD
f, psd = welch(
    x,
    fs=FS,
    nperseg=8192,
    return_onesided=False,
    scaling="density"
)

f = np.fft.fftshift(f)
psd = np.fft.fftshift(psd)

psd_db = 10 * np.log10(psd + 1e-15)

peak_index = np.argmax(psd_db)
peak_freq = f[peak_index]

print("======================================")
print("RFRecon - SMOOTHED Wi-Fi PSD")
print("======================================")
print(f"Peak frequency offset : {peak_freq/1e6:.3f} MHz")
print(f"Peak PSD              : {psd_db[peak_index]:.2f} dB")

plt.figure(figsize=(12, 6))

plt.plot(f / 1e6, psd_db)

plt.xlabel("Frequency Offset from 2.437 GHz (MHz)")
plt.ylabel("PSD (dB)")
plt.title("RFRecon - Smoothed Real PlutoSDR PSD")

plt.grid(True)
plt.xlim(-10, 10)

plt.tight_layout()
plt.savefig("real_wifi_capture_02_psd.png", dpi=150)

plt.show()