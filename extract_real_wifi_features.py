import numpy as np
import pandas as pd

FILE = r"real_dataset\real_wifi_01.dat"
OUTPUT = r"real_dataset\real_wifi_features_bwtest.csv"
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

    
    # ---------------------------------------------------------
    # 95% occupied bandwidth — cumulative power in frequency order
    # ---------------------------------------------------------

    total_spectral_power = np.sum(power_spectrum) + 1e-12

    cumulative_power = np.cumsum(power_spectrum) / total_spectral_power

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


# ---------------------------------------------------------
# LOAD REAL WIFI CAPTURE
# ---------------------------------------------------------

data = np.memmap(
    FILE,
    dtype=np.complex64,
    mode="r"
)

num_windows = len(data) // WINDOW_SIZE

print("=" * 70)
print("RFRecon - REAL WIFI FEATURE EXTRACTION")
print("=" * 70)

print(f"Samples : {len(data):,}")
print(f"Windows : {num_windows:,}")

rows = []

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    x = data[start:end]

    values = extract_features(x)

    row = dict(zip(FEATURES, values))

    row["label"] = "wifi_like"

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
print("REAL WIFI FEATURE MEANS")
print("-" * 70)

for feature in FEATURES:

    print(
        f"{feature:35s}"
        f"{df[feature].mean():.6f}"
    )