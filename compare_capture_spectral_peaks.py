
import numpy as np
import pandas as pd
from pathlib import Path

CAPTURES = {
    "noise_intended": r"real_dataset\real_noise_01.dat",
    "wifi_intended": r"real_dataset\real_wifi_01.dat",
    "bluetooth_intended": r"real_dataset\real_bluetooth_01.dat",
}

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096
NUM_WINDOWS = 500
TOP_PEAKS = 5

rows = []

for label, filename in CAPTURES.items():
    path = Path(filename)
    if not path.exists():
        print(f"Missing capture: {filename}")
        continue

    data = np.memmap(path, dtype=np.complex64, mode="r")
    count = min(NUM_WINDOWS, len(data) // WINDOW_SIZE)

    for i in range(count):
        x = np.asarray(data[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE])
        x = x * np.hanning(len(x))

        spectrum = np.fft.fftshift(np.fft.fft(x))
        power = np.abs(spectrum) ** 2

        frequencies = np.fft.fftshift(
            np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
        )

        # Rank bins by power, then report their frequency offsets.
        indices = np.argsort(power)[-TOP_PEAKS:][::-1]

        for rank, idx in enumerate(indices, start=1):
            rows.append({
                "capture": label,
                "window": i,
                "peak_rank": rank,
                "frequency_offset_mhz": frequencies[idx] / 1e6,
                "relative_power_db": 10 * np.log10(
                    power[idx] / (np.max(power) + 1e-20) + 1e-20
                ),
            })

df = pd.DataFrame(rows)
output = Path("real_dataset/capture_spectral_peaks.csv")
df.to_csv(output, index=False)

print("=" * 70)
print("RFRecon - TOP SPECTRAL PEAKS")
print("=" * 70)

print("\nPeak frequency-offset summary (MHz):")
print(df.groupby(["capture", "peak_rank"])["frequency_offset_mhz"].agg(
    ["median", "mean", "std"]
).to_string(float_format=lambda v: f"{v:.4f}"))

print(f"\nSaved: {output}")
print("No source captures or models were modified.")
