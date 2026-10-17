import numpy as np
import pandas as pd
from scipy.signal import periodogram

FILE = r"real_dataset\real_wifi_01.dat"
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

data = np.memmap(FILE, dtype=np.complex64, mode="r")
count = min(NUM_WINDOWS, len(data) // WINDOW_SIZE)
rows = []

for i in range(count):
    x = data[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
    magnitude = np.abs(x)
    power = magnitude ** 2

    # Current real-capture implementation
    window = np.hanning(len(x))
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    psd_real = np.abs(spectrum) ** 2 + 1e-12
    freq_real = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
    )
    peak_real = np.argmax(psd_real)
    centroid_real = np.sum(np.abs(freq_real) * psd_real) / np.sum(psd_real)
    bandwidth_real = np.sqrt(
        np.sum((np.abs(freq_real) - centroid_real) ** 2 * psd_real)
        / np.sum(psd_real)
    )

    order = np.argsort(psd_real)
    cumulative_sorted = np.cumsum(psd_real[order])
    low_idx = np.searchsorted(cumulative_sorted, 0.025 * cumulative_sorted[-1])
    high_idx = np.searchsorted(cumulative_sorted, 0.975 * cumulative_sorted[-1])
    occupied_real = (
        freq_real[order[high_idx]] - freq_real[order[low_idx]]
    ) / SAMPLE_RATE

    flatness_real = (
        np.exp(np.mean(np.log(psd_real))) / np.mean(psd_real)
    )

    real_values = [
        np.mean(power),
        np.sqrt(np.mean(power)),
        np.max(magnitude),
        np.var(magnitude),
        freq_real[peak_real] / (SAMPLE_RATE / 2),
        centroid_real / (SAMPLE_RATE / 2),
        bandwidth_real / (SAMPLE_RATE / 2),
        occupied_real,
        flatness_real,
        np.max(power) / (np.mean(power) + 1e-12),
    ]

    # Synthetic feature-extraction implementation, applied to the same samples
    freq_syn, psd_syn = periodogram(
        x, fs=SAMPLE_RATE, return_onesided=False
    )
    freq_syn = np.fft.fftshift(freq_syn)
    psd_syn = np.abs(np.fft.fftshift(psd_syn))
    total = np.sum(psd_syn) + 1e-12

    peak_syn = np.argmax(psd_syn)
    centroid_syn = np.sum(np.abs(freq_syn) * psd_syn) / total
    bandwidth_syn = np.sqrt(
        np.sum((np.abs(freq_syn) - centroid_syn) ** 2 * psd_syn) / total
    )

    cumulative = np.cumsum(psd_syn)
    lower = min(np.searchsorted(cumulative, 0.05 * total), len(freq_syn) - 1)
    upper = min(np.searchsorted(cumulative, 0.95 * total), len(freq_syn) - 1)
    occupied_syn = abs(freq_syn[upper] - freq_syn[lower]) / SAMPLE_RATE

    flatness_syn = (
        np.exp(np.mean(np.log(psd_syn + 1e-12)))
        / (np.mean(psd_syn) + 1e-12)
    )

    synthetic_values = [
        np.mean(power),
        np.sqrt(np.mean(power)),
        np.max(magnitude),
        np.var(magnitude),
        freq_syn[peak_syn] / (SAMPLE_RATE / 2),
        centroid_syn / (SAMPLE_RATE / 2),
        bandwidth_syn / (SAMPLE_RATE / 2),
        occupied_syn,
        flatness_syn,
        np.max(psd_syn) / (np.mean(psd_syn) + 1e-12),
    ]

    rows.append({
        "window": i,
        **{f"{name}_real": value for name, value in zip(FEATURES, real_values)},
        **{f"{name}_synthetic_formula": value for name, value in zip(FEATURES, synthetic_values)},
    })

df = pd.DataFrame(rows)

print("=" * 75)
print("RFRecon - FEATURE FORMULA AUDIT ON IDENTICAL REAL WINDOWS")
print("=" * 75)
print(f"Windows compared: {count}\n")

for feature in FEATURES:
    a = df[f"{feature}_real"].to_numpy()
    b = df[f"{feature}_synthetic_formula"].to_numpy()

    corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
    mean_abs_diff = np.mean(np.abs(a - b))
    print(
        f"{feature:<35} "
        f"corr={corr:>8.3f}  "
        f"mean_abs_diff={mean_abs_diff:.6g}"
    )

output = "real_dataset/feature_formula_audit.csv"
df.to_csv(output, index=False)
print(f"\nSaved per-window audit: {output}")
print("No source datasets or trained models were modified.")
