import numpy as np
import pandas as pd

FILE = r"real_dataset\real_wifi_01.dat"
OUTPUT = r"real_dataset\real_wifi_consistent_test.csv"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096
NUM_WINDOWS = 200

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
    "peak_to_average_ratio",
]

def extract_features(x):
    amplitude = np.abs(x)
    power = amplitude ** 2

    mean_power = np.mean(power)
    rms = np.sqrt(mean_power)
    peak_amplitude = np.max(amplitude)
    variance = np.var(amplitude)

    spectrum = np.fft.fftshift(
        np.fft.fft(x * np.hanning(len(x)))
    )
    psd = np.abs(spectrum) ** 2 + 1e-12

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
    )

    peak_idx = np.argmax(psd)
    peak_frequency_normalized = (
        frequencies[peak_idx] / (SAMPLE_RATE / 2)
    )

    abs_freq = np.abs(frequencies)
    total = np.sum(psd)

    centroid = np.sum(abs_freq * psd) / total
    centroid_norm = centroid / (SAMPLE_RATE / 2)

    bandwidth = np.sqrt(
        np.sum((abs_freq - centroid) ** 2 * psd) / total
    )
    bandwidth_norm = bandwidth / (SAMPLE_RATE / 2)

    order = np.argsort(psd)
    cumulative = np.cumsum(psd[order])

    low_idx = np.searchsorted(cumulative, 0.025 * total)
    high_idx = np.searchsorted(cumulative, 0.975 * total)

    occupied_norm = (
        frequencies[order[high_idx]] - frequencies[order[low_idx]]
    ) / SAMPLE_RATE

    flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)

    # Keep the real-capture time-domain definition.
    peak_to_average = np.max(power) / (np.mean(power) + 1e-12)

    return [
        mean_power,
        rms,
        peak_amplitude,
        variance,
        peak_frequency_normalized,
        centroid_norm,
        bandwidth_norm,
        occupied_norm,
        flatness,
        peak_to_average,
    ]

data = np.memmap(FILE, dtype=np.complex64, mode="r")
count = min(NUM_WINDOWS, len(data) // WINDOW_SIZE)

rows = []
for i in range(count):
    x = data[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
    row = dict(zip(FEATURES, extract_features(x)))
    row["window"] = i
    row["label"] = "wifi_like"
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print("=" * 65)
print("RFRecon - CONSISTENT WIFI FEATURE TEST")
print("=" * 65)
print(f"Windows processed: {len(df)}")
print(f"Saved: {OUTPUT}")
print("\nFeature means:")
print(df[FEATURES].mean().to_string())
print("\nNo existing dataset or model was modified.")
