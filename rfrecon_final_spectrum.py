import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# RFRecon - Final RF Spectrum
# Uses the same RF analysis method as complete system
# ============================================================

CAPTURE_FILE = "real_wifi_capture_03.dat"

SAMPLE_RATE = 20_000_000
CENTER_FREQUENCY = 2_400_000_000


# ============================================================
# LOAD CAPTURE
# ============================================================

print("=" * 60)
print("RFRecon - FINAL RF SPECTRUM ANALYSIS")
print("=" * 60)

iq = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

print("\nCapture:", CAPTURE_FILE)
print("Samples:", len(iq))
print("Sample rate:", SAMPLE_RATE / 1e6, "MS/s")
print("Center frequency:", CENTER_FREQUENCY / 1e9, "GHz")


# ============================================================
# FFT
# ============================================================

print("\nCalculating FFT...")

spectrum = np.fft.fftshift(
    np.fft.fft(iq)
)

power_spectrum = (
    np.abs(spectrum) ** 2
)


# ============================================================
# FREQUENCY AXIS
# ============================================================

freqs = np.fft.fftshift(
    np.fft.fftfreq(
        len(iq),
        d=1 / SAMPLE_RATE
    )
)

absolute_frequency = (
    CENTER_FREQUENCY + freqs
)


# ============================================================
# PSD-LIKE LOG SCALE
# ============================================================

power_db = 10 * np.log10(
    power_spectrum + 1e-12
)


# ============================================================
# PEAK
# ============================================================

peak_index = np.argmax(
    power_spectrum
)

peak_frequency = (
    absolute_frequency[peak_index]
)

peak_power = (
    power_db[peak_index]
)

frequency_offset = (
    peak_frequency
    - CENTER_FREQUENCY
)


# ============================================================
# OCCUPIED BANDWIDTH
# ============================================================

threshold = (
    0.01
    * np.max(power_spectrum)
)

occupied = freqs[
    power_spectrum >= threshold
]

if len(occupied) > 0:

    occupied_bandwidth = (
        np.max(occupied)
        -
        np.min(occupied)
    )

else:

    occupied_bandwidth = 0


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RF ANALYSIS")
print("=" * 60)

print(
    f"\nCenter frequency:"
    f" {CENTER_FREQUENCY / 1e9:.6f} GHz"
)

print(
    f"Peak frequency:"
    f" {peak_frequency / 1e9:.6f} GHz"
)

print(
    f"Frequency offset:"
    f" {frequency_offset / 1e6:+.6f} MHz"
)

print(
    f"Peak spectral level:"
    f" {peak_power:.2f} dB"
)

print(
    f"Occupied bandwidth:"
    f" {occupied_bandwidth / 1e6:.3f} MHz"
)


# ============================================================
# PLOT
# ============================================================

# Downsample ONLY for visualization.
# RF calculations above still use the complete FFT.

plot_points = 20000

step = max(
    1,
    len(absolute_frequency) // plot_points
)

plot_frequency = (
    absolute_frequency[::step] / 1e9
)

plot_power = (
    power_db[::step]
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    plot_frequency,
    plot_power,
    linewidth=0.8,
    label="RF Spectrum"
)

plt.axvline(
    CENTER_FREQUENCY / 1e9,
    linestyle="--",
    linewidth=1.2,
    label="Center Frequency"
)

plt.axvline(
    peak_frequency / 1e9,
    linestyle="--",
    linewidth=1.2,
    label="Peak Frequency"
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Spectral Level (dB)"
)

plt.title(
    "RFRecon - Real PlutoSDR RF Spectrum"
)

plt.grid(
    True,
    alpha=0.3
)

# Fixed legend location avoids the slow "loc=best" warning
plt.legend(
    loc="upper right"
)

plt.tight_layout()


OUTPUT_FILE = (
    "RFRecon_final_RF_spectrum_clean.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nPresentation-ready spectrum saved as:"
)

print(
    OUTPUT_FILE
)
plt.axvline(
    CENTER_FREQUENCY / 1e9,
    linestyle="--",
    linewidth=1.2,
    label="Center Frequency"
)

plt.axvline(
    peak_frequency / 1e9,
    linestyle="--",
    linewidth=1.2,
    label="Peak Frequency"
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Relative Spectral Level (dB)"
)

plt.title(
    "RFRecon - Real PlutoSDR RF Spectrum"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    "RFRecon_final_RF_spectrum.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSpectrum saved as:"
)

print(
    OUTPUT_FILE
)

print("\nRF spectrum analysis completed.")