
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
SECTIONS = 5

rows = []

for label, filename in CAPTURES.items():
    path = Path(filename)
    if not path.exists():
        print(f"Missing capture: {filename}")
        continue

    data = np.memmap(path, dtype=np.complex64, mode="r")
    total_windows = len(data) // WINDOW_SIZE
    section_size = total_windows // SECTIONS

    for section in range(SECTIONS):
        start_window = section * section_size
        end_window = (
            (section + 1) * section_size
            if section < SECTIONS - 1
            else total_windows
        )

        peak_offsets = []

        for i in range(start_window, end_window):
            start = i * WINDOW_SIZE
            x = np.asarray(data[start:start + WINDOW_SIZE])

            x = x * np.hanning(WINDOW_SIZE)
            spectrum = np.fft.fftshift(np.fft.fft(x))
            power = np.abs(spectrum) ** 2

            frequencies = np.fft.fftshift(
                np.fft.fftfreq(
                    WINDOW_SIZE, d=1 / SAMPLE_RATE
                )
            )

            peak_offsets.append(
                frequencies[np.argmax(power)] / 1e6
            )

        rows.append({
            "capture": label,
            "section": section + 1,
            "windows": len(peak_offsets),
            "median_peak_offset_mhz": np.median(peak_offsets),
            "mean_peak_offset_mhz": np.mean(peak_offsets),
            "std_peak_offset_mhz": np.std(peak_offsets),
        })

df = pd.DataFrame(rows)
output = Path("real_dataset/peak_stability.csv")
df.to_csv(output, index=False)

print("=" * 75)
print("RFRecon - CHRONOLOGICAL PEAK STABILITY")
print("=" * 75)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nSaved: {output}")
print("Read-only diagnostic; no captures or models were modified.")

