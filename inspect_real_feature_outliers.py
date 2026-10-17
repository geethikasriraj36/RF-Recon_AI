
import pandas as pd

FILE = "real_dataset/real_capture_features_consistent.csv"

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

df = pd.read_csv(FILE)

print("Dataset shape:", df.shape)

for feature in FEATURES:
    print(f"\n{'=' * 60}")
    print("FEATURE:", feature)

    for label, group in df.groupby("class_name"):
        values = group[feature].dropna()

        q1 = values.quantile(0.25)
        median = values.median()
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr

        print(
            f"{label:16s} "
            f"median={median:.6g} "
            f"Q1={q1:.6g} "
            f"Q3={q3:.6g} "
            f"max={values.max():.6g} "
            f"above_3IQR={int((values > upper).sum())}"
        )

print("\nRead-only analysis complete.")
print("No datasets, captures, or models were modified.")
