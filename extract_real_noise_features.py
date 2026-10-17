import numpy as np
import pandas as pd

FILE = r"real_dataset\real_noise_01.dat"
OUTPUT = r"real_dataset\real_noise_features.csv"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096


def extract_features(x):

    amplitude = np.abs(x)
    power = amplitude ** 2

    mean_power = np.mean(power)
    rms = np.sqrt(mean_power)
    peak_amplitude = np.max(amplitude)
    variance = np.var(amplitude)

    window = np.hanning(len(x))

    spectrum = np.fft.fftshift(
        np.fft.fft(x * window)
    )

    power_spectrum = np.abs(spectrum) ** 2 + 1e-12

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(x),
            d=1 / SAMPLE_RATE
        )
    )

    peak_index = np.argmax(power_spectrum)

    peak_frequency = frequencies[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    spectral_centroid = (
        np.sum(
            np.abs(frequencies) * power_spectrum
        )
        / np.sum(power_spectrum)
    )

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(frequencies)
                - spectral_centroid
            ) ** 2
            * power_spectrum
        )
        / np.sum(power_spectrum)
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # 95% occupied bandwidth
    sorted_indices = np.argsort(power_spectrum)
    sorted_power = power_spectrum[sorted_indices]
    cumulative = np.cumsum(sorted_power)
    total = cumulative[-1]

    low_idx = np.searchsorted(
        cumulative,
        0.025 * total
    )

    high_idx = np.searchsorted(
        cumulative,
        0.975 * total
    )

    low_freq = frequencies[sorted_indices[low_idx]]
    high_freq = frequencies[sorted_indices[high_idx]]

    occupied_bandwidth = high_freq - low_freq

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    geometric_mean = np.exp(
        np.mean(np.log(power_spectrum))
    )

    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    peak_to_average_ratio = (
        np.max(power)
        / (np.mean(power) + 1e-12)
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


features = [
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


data = np.memmap(
    FILE,
    dtype=np.complex64,
    mode="r"
)

num_windows = len(data) // WINDOW_SIZE

print("=" * 70)
print("RFRecon - REAL NOISE FEATURE EXTRACTION")
print("=" * 70)

print(f"Samples : {len(data):,}")
print(f"Windows : {num_windows:,}")

rows = []

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    x = data[start:end]

    values = extract_features(x)

    row = dict(zip(features, values))

    row["label"] = "noise"

    rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)

print()
print("REAL NOISE FEATURE MEANS")
print("-" * 70)

for feature in features:

    print(
        f"{feature:35s}"
        f"{df[feature].mean():.6f}"
    )