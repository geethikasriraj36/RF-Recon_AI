import numpy as np
import matplotlib.pyplot as plt

FILE = r"real_dataset\real_cellular_01.dat"

SAMPLE_RATE = 20e6
CENTER_FREQ = 1.856e9

x = np.memmap(
    FILE,
    dtype=np.complex64,
    mode="r"
)

# Use 2^20 samples
N = 2**20
samples = x[:N]

# FFT
window = np.hanning(N)

spectrum = np.fft.fftshift(
    np.fft.fft(samples * window)
)

power = np.abs(spectrum) ** 2

power_db = 10 * np.log10(
    power + 1e-12
)

frequency = np.fft.fftshift(
    np.fft.fftfreq(N, d=1/SAMPLE_RATE)
)

frequency_hz = CENTER_FREQ + frequency

# ---------------------------------------------------------
# Ignore DC region
# ---------------------------------------------------------

dc_mask = np.abs(frequency_hz - CENTER_FREQ) < 100e3

power_for_peak = power.copy()
power_for_peak[dc_mask] = 0

peak_index = np.argmax(power_for_peak)

peak_frequency = frequency_hz[peak_index]

print("=" * 70)
print("RFRecon - CELLULAR CANDIDATE BANDWIDTH ANALYSIS")
print("=" * 70)

print(f"Capture center frequency : {CENTER_FREQ/1e9:.6f} GHz")
print(f"Non-DC peak frequency   : {peak_frequency/1e9:.9f} GHz")
print(
    f"Peak offset             : "
    f"{(peak_frequency-CENTER_FREQ)/1e6:+.6f} MHz"
)

# ---------------------------------------------------------
# Occupied bandwidth using 99% power
# ---------------------------------------------------------

total_power = np.sum(power)

sorted_indices = np.argsort(power)[::-1]

cumulative = np.cumsum(
    power[sorted_indices]
)

threshold_index = np.searchsorted(
    cumulative,
    0.99 * total_power
)

selected_indices = sorted_indices[:threshold_index + 1]

low_frequency = np.min(
    frequency_hz[selected_indices]
)

high_frequency = np.max(
    frequency_hz[selected_indices]
)

occupied_bandwidth = (
    high_frequency - low_frequency
)

print()
print("99% OCCUPIED BANDWIDTH")
print("-" * 70)

print(
    f"Lower edge : {low_frequency/1e6:.6f} MHz"
)

print(
    f"Upper edge : {high_frequency/1e6:.6f} MHz"
)

print(
    f"Bandwidth  : {occupied_bandwidth/1e6:.6f} MHz"
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    frequency_hz / 1e6,
    power_db
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Spectral Level (dB)")
plt.title(
    "RFRecon - Real Cellular Candidate "
    "Bandwidth Analysis"
)

plt.xlim(
    1846,
    1866
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "real_cellular_bandwidth_analysis.png",
    dpi=200
)

plt.show()

print()
print("Saved:")
print("real_cellular_bandwidth_analysis.png")