import numpy as np
import pandas as pd

FILE = r"real_dataset\real_wifi_01.dat"
SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096
NUM_WINDOWS = 200

data = np.memmap(FILE, dtype=np.complex64, mode="r")
available = len(data) // WINDOW_SIZE
count = min(NUM_WINDOWS, available)

def calculate_ratios(x):
    power = np.abs(x) ** 2

    # Formula used by the real Wi-Fi extraction script
    time_domain_ratio = np.max(power) / (np.mean(power) + 1e-12)

    # Formula used by the synthetic feature extraction script
    frequencies, psd = __import__(
        "scipy.signal", fromlist=["periodogram"]
    ).periodogram(
        x,
        fs=SAMPLE_RATE,
        return_onesided=False
    )

    psd = np.abs(np.fft.fftshift(psd))
    spectral_ratio = np.max(psd) / (np.mean(psd) + 1e-12)

    return time_domain_ratio, spectral_ratio

rows = []

for i in range(count):
    start = i * WINDOW_SIZE
    x = data[start:start + WINDOW_SIZE]

    time_ratio, spectral_ratio = calculate_ratios(x)

    rows.append({
        "window": i,
        "time_domain_ratio": time_ratio,
        "spectral_ratio": spectral_ratio,
    })

df = pd.DataFrame(rows)

print("=" * 70)
print("RFRecon - PEAK-TO-AVERAGE FORMULA CHECK")
print("=" * 70)
print(f"Capture: {FILE}")
print(f"Windows tested: {count}")

print("\nSummary:")
print(df[["time_domain_ratio", "spectral_ratio"]].describe().round(4))

print("\nCorrelation between the two definitions:")
print(
    df["time_domain_ratio"].corr(df["spectral_ratio"])
)

print("\nFirst 10 windows:")
print(df.head(10).round(4).to_string(index=False))

print("\nNo model or dataset was modified.")
