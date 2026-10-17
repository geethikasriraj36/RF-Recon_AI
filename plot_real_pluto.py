import numpy as np
import matplotlib.pyplot as plt

# Load real Pluto sample
x = np.load(r"real_pluto_data\real_pluto_0000.npy")
# Remove DC offset from the IQ samples
x = x - np.mean(x)

fs = 2_000_000

# FFT
spectrum = np.fft.fftshift(np.fft.fft(x))
power = np.abs(spectrum) ** 2

# Convert to dB
power_db = 10 * np.log10(power + 1e-12)

# Frequency axis
freqs = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / fs)
)

# Plot
plt.figure(figsize=(10, 5))

plt.plot(
    freqs / 1e6,
    power_db
)

plt.xlabel("Frequency Offset (MHz)")
plt.ylabel("Power (dB)")
plt.title("RFRecon - Real PlutoSDR Spectrum")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "real_pluto_spectrum.png",
    dpi=200
)

plt.show()

print("Spectrum saved as real_pluto_spectrum.png")
print(
    "Peak frequency offset:",
    freqs[np.argmax(power)] / 1e3,
    "kHz"
)