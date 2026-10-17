
import numpy as np
import pandas as pd

FILE = r"real_dataset\real_bluetooth_01.dat"
OUTPUT = r"real_dataset\real_bluetooth_features_bwtest.csv"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096

FEATURES = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio"
]


def extract_features(x):
    amplitude = np.abs(x)
    power = amplitude ** 2

    mean_power = np.mean(power)
    rms = np.sqrt(mean_power)
    peak_amplitude = np.max(amplitude)
    variance = np.var(amplitude)

    # Windowed FFT
    window = np.hanning(len(x))
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    power_spectrum = np.abs(spectrum) ** 2

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
    )

    # Avoid division by zero in spectral calculations
    power_spectrum = power_spectrum + 1e-12
    total_spectral_power = np.sum(power_spectrum)

    # Peak frequency
    peak_index = np.argmax(power_spectrum)
    peak_frequency = frequencies[peak_index]
    peak_frequency_normalized = peak_frequency / (SAMPLE_RATE / 2)

    # Spectral centroid
    spectral_centroid = (
        np.sum(np.abs(frequencies) * power_spectrum)
        / total_spectral_power
    )
    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            (np.abs(frequencies) - spectral_centroid) ** 2
            * power_spectrum
        ) / total_spectral_power
    )
    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # 95% occupied bandwidth: cumulative power in frequency order
    cumulative_power = (
        np.cumsum(power_spectrum) / total_spectral_power
    )

    low_idx = np.searchsorted(cumulative_power, 0.025)
    high_idx = np.searchsorted(cumulative_power, 0.975)

    low_idx = min(low_idx, len(frequencies) - 1)
    high_idx = min(high_idx, len(frequencies) - 1)

    occupied_bandwidth = (
        frequencies[high_idx] - frequencies[low_idx]
    )
    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(np.log(power_spectrum))
    )
    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness = (
        geometric_mean / (arithmetic_mean + 1e-12)
    )

    # Peak-to-average spectral power ratio
    peak_to_average_ratio = (
        np.max(power_spectrum)
        / (np.mean(power_spectrum) + 1e-12)
    )

    return [
        mean_power,
        rms,
        peak_amplitude,
        variance,
        peak_frequency_normalized,
        spectral_centroid_normalized,
        spectral_bandwidth_normalized,
        occupied_bandwidth_normalized,
        spectral_flatness,
        peak_to_average_ratio
    ]


print("=" * 65)
print("RFRecon - REAL BLUETOOTH FEATURE EXTRACTION")
print("=" * 65)

data = np.memmap(FILE, dtype=np.complex64, mode="r")
num_windows = len(data) // WINDOW_SIZE

print(f"Samples: {len(data):,}")
print(f"Windows: {num_windows:,}")

rows = []

for i in range(num_windows):
    start = i * WINDOW_SIZE
    x = data[start:start + WINDOW_SIZE]

    row = dict(zip(FEATURES, extract_features(x)))
    row["label"] = "bluetooth_like"
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print(f"Rows: {len(df):,}")
print("\nFeature means:")
print(df[FEATURES].mean().to_string())
print("\nDone. No model was retrained or overwritten.")
