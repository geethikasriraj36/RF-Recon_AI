import numpy as np
import matplotlib.pyplot as plt

# Load one real Wi-Fi capture window
x = np.load(r"real_wifi_data\real_wifi_0000.npy")

fs = 2_000_000

# Remove DC offset
x = x - np.mean(x)

# FFT
spectrum = np.fft.fftshift(np.fft.fft(x))
power = np.abs(spectrum) ** 2

# Convert to dB
power_db = 10 * np.log10(power + 1e-12)

# Frequency offset from 2.437 GHz LO
freqs = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / fs)
)

# Plot
plt.figure(figsize=(11, 5))

plt.plot(
    freqs / 1e6,
    power_db
)

plt.xlabel("Frequency Offset from 2.437 GHz (MHz)")
plt.ylabel("Power (dB)")
plt.title("RFRecon - Real Wi-Fi PlutoSDR Spectrum")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "real_wifi_spectrum.png",
    dpi=200
)

plt.show()

peak_index = np.argmax(power)

print("\n===== REAL WI-FI SPECTRUM =====")
print(
    f"Strongest frequency offset: "
    f"{freqs[peak_index] / 1e6:.6f} MHz"
)

print(
    f"Approximate RF frequency: "
    f"{(2437e6 + freqs[peak_index]) / 1e6:.6f} MHz"
)

print("Spectrum saved as real_wifi_spectrum.png")