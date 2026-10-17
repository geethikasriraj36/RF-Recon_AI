import numpy as np
import pandas as pd
from pathlib import Path

CAPTURES = {
    "noise": r"real_dataset\real_noise_01.dat",
    "wifi_intended": r"real_dataset\real_wifi_01.dat",
    "bluetooth_intended": r"real_dataset\real_bluetooth_01.dat",
}

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096
NUM_WINDOWS = 500

def measure(x):
    window = np.hanning(len(x))
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    power = np.abs(spectrum) ** 2

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
    )

    total = np.sum(power)
    if total <= 0:
        return np.nan, np.nan, np.nan

    cumulative = np.cumsum(power) / total
    low = min(np.searchsorted(cumulative, 0.025), len(power) - 1)
    high = min(np.searchsorted(cumulative, 0.975), len(power) - 1)

    bandwidth = frequencies[high] - frequencies[low]

    # Fraction of spectral energy within the central 10 MHz.
    central = np.abs(frequencies) <= 5e6
    central_fraction = np.sum(power[central]) / total

    # Fraction within the outer 5 MHz at either edge.
    edges = np.abs(frequencies) >= 7.5e6
    edge_fraction = np.sum(power[edges]) / total

    return bandwidth / SAMPLE_RATE, central_fraction, edge_fraction

rows = []

for label, filename in CAPTURES.items():
    path = Path(filename)
    if not path.exists():
        print(f"Missing capture: {filename}")
        continue

    data = np.memmap(path, dtype=np.complex64, mode="r")
    count = min(NUM_WINDOWS, len(data) // WINDOW_SIZE)

    for i in range(count):
        start = i * WINDOW_SIZE
        x = np.asarray(data[start:start + WINDOW_SIZE])
        bw, central, edges = measure(x)

        rows.append({
            "capture": label,
            "window": i,
            "occupied_bandwidth_normalized": bw,
            "central_10mhz_energy_fraction": central,
            "outer_edge_energy_fraction": edges,
        })

df = pd.DataFrame(rows)
out = Path("real_dataset/capture_bandwidth_comparison.csv")
df.to_csv(out, index=False)

print("=" * 70)
print("RFRecon - CAPTURE BANDWIDTH COMPARISON")
print("=" * 70)
print(df.groupby("capture").agg(
    windows=("window", "count"),
    bandwidth_median=("occupied_bandwidth_normalized", "median"),
    bandwidth_mean=("occupied_bandwidth_normalized", "mean"),
    central_energy_mean=("central_10mhz_energy_fraction", "mean"),
    edge_energy_mean=("outer_edge_energy_fraction", "mean"),
).to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\nSaved: {out}")
print("No source captures or models were modified.")
