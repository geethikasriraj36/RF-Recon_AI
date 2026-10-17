import numpy as np
import matplotlib.pyplot as plt

FILE = r"real_dataset\real_cellular_01.dat"

SAMPLE_RATE = 20e6
CENTER_FREQ = 1.856e9

# Read a manageable portion of the capture
x = np.memmap(FILE, dtype=np.complex64, mode="r")

# Use first 2^20 samples for spectrum
N = 2**20
samples = x[:N]

# FFT
window = np.hanning(N)
spectrum = np.fft.fftshift(np.fft.fft(samples * window))

power = 20 * np.log10(np.abs(spectrum) + 1e-12)

freq = np.fft.fftshift(
    np.fft.fftfreq(N, d=1/SAMPLE_RATE)
)

freq_hz = CENTER_FREQ + freq

# Find peak
# Ignore the center/DC region
dc_mask = np.abs(freq_hz - CENTER_FREQ) < 100e3

power_no_dc = power.copy()
power_no_dc[dc_mask] = -np.inf

# Find strongest non-DC RF component
peak_index = np.argmax(power_no_dc)

peak_frequency = freq_hz[peak_index]
peak_power = power_no_dc[peak_index]

print("=" * 60)
print("RFRecon - REAL CELLULAR CAPTURE CHECK")
print("=" * 60)

print(f"Center frequency       : {CENTER_FREQ/1e9:.6f} GHz")
print(f"Non-DC peak frequency  : {peak_frequency/1e9:.9f} GHz")
print(f"Non-DC frequency offset: {(peak_frequency-CENTER_FREQ)/1e6:+.6f} MHz")
print(f"Non-DC peak level      : {peak_power:.2f} dB")

print("=" * 60)
# Plot
plt.figure(figsize=(12, 5))

plt.plot(freq_hz / 1e6, power)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative Spectral Level (dB)")
plt.title("RFRecon - Real Cellular Candidate Spectrum")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "real_cellular_spectrum.png",
    dpi=200
)

plt.show()