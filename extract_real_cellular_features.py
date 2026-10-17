import numpy as np
import pandas as pd
import os

INPUT_DIR = r"real_dataset\cellular_like"
OUTPUT_FILE = r"real_dataset\real_cellular_features.csv"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096


def extract_features(x):

    power = np.abs(x) ** 2
    amplitude = np.abs(x)

    mean_power = np.mean(power)
    rms = np.sqrt(np.mean(power))
    peak_amplitude = np.max(amplitude)
    variance = np.var(power)

    # FFT
    spectrum = np.fft.fftshift(
        np.fft.fft(x * np.hanning(len(x)))
    )

    psd = np.abs(spectrum) ** 2
    psd = psd / (np.sum(psd) + 1e-12)

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
    )

    # Peak frequency
    peak_idx = np.argmax(psd)
    peak_frequency = frequencies[peak_idx]

    # Normalize frequency
    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    # Spectral centroid
    spectral_centroid = np.sum(
        np.abs(frequencies) * psd
    )

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            ((np.abs(frequencies) - spectral_centroid) ** 2)
            * psd
        )
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # Occupied bandwidth
    cumulative = np.cumsum(psd)

    low_idx = np.searchsorted(
        cumulative, 0.05
    )

    high_idx = np.searchsorted(
        cumulative, 0.95
    )

    occupied_bandwidth = (
        frequencies[high_idx] -
        frequencies[low_idx]
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    # Spectral flatness
    spectral_flatness = (
        np.exp(np.mean(np.log(psd + 1e-12))) /
        (np.mean(psd) + 1e-12)
    )

    # Peak-to-average ratio
    peak_to_average_ratio = (
        np.max(power) /
        (np.mean(power) + 1e-12)
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


feature_names = [
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


files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".npy")
])

rows = []

print("=" * 60)
print("RFRecon - REAL CELLULAR FEATURE EXTRACTION")
print("=" * 60)

print(f"Windows found: {len(files)}")

for i, filename in enumerate(files):

    path = os.path.join(INPUT_DIR, filename)

    x = np.load(path)

    features = extract_features(x)

    row = dict(zip(feature_names, features))

    row["filename"] = filename
    row["label"] = "cellular_like"

    rows.append(row)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(files)} windows")


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"Feature matrix: {df.shape}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Feature means:")
print(df[feature_names].mean())