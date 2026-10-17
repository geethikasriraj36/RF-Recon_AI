import numpy as np
import pandas as pd
import os

INPUT_DIR = "real_wifi_dataset"
OUTPUT_FILE = "real_wifi_dataset.csv"

FS = 20_000_000
NFFT = 4096

rows = []

for filename in sorted(os.listdir(INPUT_DIR)):

    if not filename.endswith(".npy"):
        continue

    x = np.load(os.path.join(INPUT_DIR, filename))

    # Time-domain features
    magnitude = np.abs(x)

    mean_power = np.mean(magnitude ** 2)
    rms = np.sqrt(np.mean(magnitude ** 2))
    peak_amplitude = np.max(magnitude)
    variance = np.var(magnitude)

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(x, n=NFFT))
    power_spectrum = np.abs(spectrum) ** 2

    freqs = np.fft.fftshift(
        np.fft.fftfreq(NFFT, 1 / FS)
    )

    # Normalize spectrum
    total_power = np.sum(power_spectrum) + 1e-12
    normalized_power = power_spectrum / total_power

    # Peak frequency
    peak_index = np.argmax(power_spectrum)
    peak_frequency = abs(freqs[peak_index])

    peak_frequency_normalized = peak_frequency / (FS / 2)

    # Spectral centroid
    spectral_centroid = np.sum(
        abs(freqs) * normalized_power
    )

    spectral_centroid_normalized = (
        spectral_centroid / (FS / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            ((abs(freqs) - spectral_centroid) ** 2)
            * normalized_power
        )
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (FS / 2)
    )

    # Occupied bandwidth: 90% of spectral energy
    cumulative = np.cumsum(normalized_power)

    low_index = np.searchsorted(cumulative, 0.05)
    high_index = np.searchsorted(cumulative, 0.95)

    occupied_bandwidth = (
        abs(freqs[high_index] - freqs[low_index])
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / FS
    )

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(np.log(power_spectrum + 1e-12))
    )

    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness = (
        geometric_mean / (arithmetic_mean + 1e-12)
    )

    # Peak-to-average ratio
    peak_to_average_ratio = (
        peak_amplitude / (rms + 1e-12)
    )

    rows.append({
        "mean_power": mean_power,
        "rms": rms,
        "peak_amplitude": peak_amplitude,
        "variance": variance,
        "peak_frequency_normalized":
            peak_frequency_normalized,
        "spectral_centroid_normalized":
            spectral_centroid_normalized,
        "spectral_bandwidth_normalized":
            spectral_bandwidth_normalized,
        "occupied_bandwidth_normalized":
            occupied_bandwidth_normalized,
        "spectral_flatness":
            spectral_flatness,
        "peak_to_average_ratio":
            peak_to_average_ratio,
        "label": "wifi_like"
    })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print("======================================")
print("REAL Wi-Fi FEATURE DATASET")
print("======================================")
print("Samples :", len(df))
print("Features:", len(df.columns) - 1)
print("Output  :", OUTPUT_FILE)
print()
print(df.head())