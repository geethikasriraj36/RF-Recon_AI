import numpy as np
import matplotlib.pyplot as plt

FILE = r"real_dataset\real_cellular_01.dat"

SAMPLE_RATE = 20e6
CENTER_FREQ = 1.856e9

# =========================================================
# LOAD CAPTURE
# =========================================================

x = np.memmap(FILE, dtype=np.complex64, mode="r")

N = 2**20
samples = x[:N]

window = np.hanning(N)

spectrum = np.fft.fftshift(
    np.fft.fft(samples * window)
)

power = np.abs(spectrum) ** 2
power_db = 10 * np.log10(power + 1e-12)

frequency = np.fft.fftshift(
    np.fft.fftfreq(N, d=1 / SAMPLE_RATE)
)

frequency_hz = CENTER_FREQ + frequency

# =========================================================
# REMOVE PLUTO DC ARTIFACT
# =========================================================

dc_mask = np.abs(frequency_hz - CENTER_FREQ) < 100e3

valid_power = power_db.copy()
valid_power[dc_mask] = np.nan

# =========================================================
# FIND STRONGEST NON-DC PEAK
# =========================================================

peak_index = np.nanargmax(valid_power)

peak_frequency = frequency_hz[peak_index]
peak_power = power_db[peak_index]

print("=" * 70)
print("RFRecon - CONTIGUOUS CELLULAR CANDIDATE ANALYSIS")
print("=" * 70)

print(f"Center frequency : {CENTER_FREQ/1e9:.6f} GHz")
print(f"Peak frequency   : {peak_frequency/1e9:.9f} GHz")
print(f"Peak offset      : {(peak_frequency-CENTER_FREQ)/1e6:+.6f} MHz")
print(f"Peak level       : {peak_power:.2f} dB")

# =========================================================
# ESTIMATE NOISE FLOOR
# =========================================================

noise_mask = (
    (frequency_hz < CENTER_FREQ - 2e6) |
    (frequency_hz > CENTER_FREQ + 2e6)
)

noise_values = power_db[noise_mask]

noise_median = np.median(noise_values)
noise_std = np.std(noise_values)

print()
print("NOISE FLOOR")
print("-" * 70)
print(f"Median noise level : {noise_median:.2f} dB")
print(f"Noise std          : {noise_std:.2f} dB")

# =========================================================
# SIGNAL THRESHOLD
# =========================================================

threshold = noise_median + 6

print()
print("SIGNAL THRESHOLD")
print("-" * 70)
print(f"Threshold          : {threshold:.2f} dB")
print(f"Peak               : {peak_power:.2f} dB")
print(f"Peak above noise   : {peak_power-noise_median:.2f} dB")

# =========================================================
# FIND CONTIGUOUS SIGNAL REGION
# =========================================================

signal_mask = power_db > threshold

# Remove DC region
signal_mask[dc_mask] = False

# Start at strongest peak
left = peak_index
right = peak_index

while left > 0 and signal_mask[left - 1]:
    left -= 1

while right < len(signal_mask) - 1 and signal_mask[right + 1]:
    right += 1

lower_edge = frequency_hz[left]
upper_edge = frequency_hz[right]

bandwidth = upper_edge - lower_edge

print()
print("CONTIGUOUS OCCUPIED BANDWIDTH")
print("-" * 70)
print(f"Lower edge         : {lower_edge/1e6:.6f} MHz")
print(f"Upper edge         : {upper_edge/1e6:.6f} MHz")
print(f"Bandwidth          : {bandwidth/1e6:.6f} MHz")

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    frequency_hz / 1e6,
    power_db,
    linewidth=0.8
)

plt.axhline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.2f} dB"
)

plt.axvline(
    lower_edge / 1e6,
    linestyle="--",
    label="Detected lower edge"
)

plt.axvline(
    upper_edge / 1e6,
    linestyle="--",
    label="Detected upper edge"
)

plt.axvline(
    peak_frequency / 1e6,
    linestyle=":",
    label=f"Peak = {peak_frequency/1e6:.3f} MHz"
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Spectral Level (dB)")

plt.title(
    "RFRecon - Contiguous Bandwidth of Real Cellular Candidate"
)

plt.xlim(1846, 1866)

plt.grid(True, alpha=0.3)

plt.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    "real_cellular_contiguous_bandwidth.png",
    dpi=200
)

plt.show()

print()
print("Saved:")
print("real_cellular_contiguous_bandwidth.png")