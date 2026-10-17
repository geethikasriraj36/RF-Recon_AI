import numpy as np
import matplotlib.pyplot as plt

FILE = "real_wifi_capture_02.dat"
FS = 20_000_000

# Read complex64 IQ data
x = np.fromfile(FILE, dtype=np.complex64)

# Use first 262144 samples for spectrum
x = x[:262144]

# FFT
X = np.fft.fftshift(np.fft.fft(x))
freqs = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / FS)
)

power = 20 * np.log10(np.abs(X) + 1e-12)

# Peak
peak_index = np.argmax(power)

print("====================================")
print("RFRecon - Wi-Fi Capture 02")
print("====================================")
print("Samples analyzed :", len(x))
print("Sample rate      : 20 MHz")
print(
    "Peak offset      : {:.3f} MHz".format(
        freqs[peak_index] / 1e6
    )
)

plt.figure(figsize=(12, 6))

plt.plot(
    freqs / 1e6,
    power
)

plt.xlabel("Frequency Offset from 2.437 GHz (MHz)")
plt.ylabel("Power (dB)")
plt.title("RFRecon - Real Wi-Fi PlutoSDR Capture 02")
plt.grid(True)

plt.xlim(-10, 10)

plt.tight_layout()

plt.savefig(
    "real_wifi_capture_02_spectrum.png",
    dpi=150
)

plt.show()