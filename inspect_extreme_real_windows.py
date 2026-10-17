
import pandas as pd

FILE = "real_dataset/real_capture_features_consistent.csv"

df = pd.read_csv(FILE)

columns = [
    "class_name",
    "source_file",
    "window_index",
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_to_average_ratio",
    "spectral_flatness",
]

print("TOP 20 WINDOWS BY PEAK-TO-AVERAGE RATIO")
print(
    df.nlargest(20, "peak_to_average_ratio")[columns]
    .to_string(index=False)
)

print("\nTOP 20 WINDOWS BY MEAN POWER")
print(
    df.nlargest(20, "mean_power")[columns]
    .to_string(index=False)
)

print("\nTOP 20 WINDOWS BY PEAK AMPLITUDE")
print(
    df.nlargest(20, "peak_amplitude")[columns]
    .to_string(index=False)
)

print("\nRead-only inspection complete.")
