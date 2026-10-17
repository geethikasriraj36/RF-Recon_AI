import numpy as np
from scipy.signal import welch

# ============================================================
# RFRecon - REAL PLUTO SDR RF ANALYSIS
# Frequency + Power + Occupied Bandwidth
# ============================================================

CAPTURE_FILE = "real_wifi_capture_03.dat"
SAMPLE_RATE = 20_000_000       # 20 MHz
CENTER_FREQUENCY = 2_400_000_000  # 2.4 GHz

WINDOW_SIZE = 4096


print("=" * 65)
print("RFRecon - REAL PLUTO SDR RF ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# Load capture
# ------------------------------------------------------------

samples = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

print("\nCapture file:", CAPTURE_FILE)
print("Total samples:", len(samples))
print("Sample rate:", SAMPLE_RATE / 1e6, "MHz")
print("Center frequency:", CENTER_FREQUENCY / 1e9, "GHz")


if len(samples) < WINDOW_SIZE:
    raise ValueError("Capture is too small for analysis.")


# ------------------------------------------------------------
# Use a manageable portion of the capture
# ------------------------------------------------------------

samples = samples[:min(len(samples), 2_000_000)]

print("Samples analyzed:", len(samples))


# ------------------------------------------------------------
# Remove DC component
# ------------------------------------------------------------

samples = samples - np.mean(samples)


# ------------------------------------------------------------
# Power estimation
# ------------------------------------------------------------

power_linear = np.mean(np.abs(samples) ** 2)

power_db = 10 * np.log10(power_linear + 1e-12)


# ------------------------------------------------------------
# FFT / PSD
# ------------------------------------------------------------

frequencies, psd = welch(
    samples,
    fs=SAMPLE_RATE,
    nperseg=WINDOW_SIZE,
    return_onesided=False
)

psd = np.fft.fftshift(psd)
frequencies = np.fft.fftshift(frequencies)


# ------------------------------------------------------------
# Find strongest RF component
# ------------------------------------------------------------

peak_index = np.argmax(psd)

peak_offset = frequencies[peak_index]

peak_absolute_frequency = (
    CENTER_FREQUENCY + peak_offset
)


# ------------------------------------------------------------
# Occupied bandwidth
# 99% power bandwidth
# ------------------------------------------------------------

power_spectrum = psd.copy()

total_power = np.sum(power_spectrum)

cumulative_power = np.cumsum(power_spectrum)

lower_limit = 0.005 * total_power
upper_limit = 0.995 * total_power

lower_index = np.searchsorted(
    cumulative_power,
    lower_limit
)

upper_index = np.searchsorted(
    cumulative_power,
    upper_limit
)

occupied_bandwidth = (
    frequencies[min(upper_index, len(frequencies) - 1)]
    -
    frequencies[min(lower_index, len(frequencies) - 1)]
)

occupied_bandwidth = abs(occupied_bandwidth)


# ------------------------------------------------------------
# Peak PSD
# ------------------------------------------------------------

peak_psd_db = 10 * np.log10(
    psd[peak_index] + 1e-12
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("RF ANALYSIS RESULTS")
print("=" * 65)

print(
    f"\nCenter Frequency     : "
    f"{CENTER_FREQUENCY / 1e9:.6f} GHz"
)

print(
    f"Peak Frequency       : "
    f"{peak_absolute_frequency / 1e9:.6f} GHz"
)

print(
    f"Frequency Offset     : "
    f"{peak_offset / 1e6:.3f} MHz"
)

print(
    f"Relative Power       : "
    f"{power_db:.2f} dB"
)

print(
    f"Peak PSD             : "
    f"{peak_psd_db:.2f} dB"
)

print(
    f"Occupied Bandwidth   : "
    f"{occupied_bandwidth / 1e6:.3f} MHz"
)

print("\n" + "=" * 65)
print("RFRecon RF DECISION")
print("=" * 65)

print(
    f"\nFrequency : "
    f"{peak_absolute_frequency / 1e9:.6f} GHz"
)

print(
    f"Power     : "
    f"{power_db:.2f} dB"
)

print(
    f"Bandwidth : "
    f"{occupied_bandwidth / 1e6:.3f} MHz"
)

print("=" * 65)