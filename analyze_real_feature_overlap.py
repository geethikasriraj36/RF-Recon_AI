
import pandas as pd

FILE = "real_dataset/real_capture_features_consistent.csv"

df = pd.read_csv(FILE)

features = [
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

classes = ["noise", "wifi_like", "bluetooth_like"]

for feature in features:
    print(f"\n{'=' * 65}")
    print(feature)

    summary = df.groupby("class_name")[feature].agg(
        ["mean", "std", "median"]
    )

    print(summary.round(6).to_string())

    noise = df.loc[df["class_name"] == "noise", feature]
    wifi = df.loc[df["class_name"] == "wifi_like", feature]

    # Fraction of values in the overlapping interval.
    low = max(noise.min(), wifi.min())
    high = min(noise.max(), wifi.max())

    if low <= high:
        noise_overlap = noise.between(low, high).mean()
        wifi_overlap = wifi.between(low, high).mean()

        print(f"Range intersection: [{low:.6f}, {high:.6f}]")
        print(
            "Values inside shared range — "
            f"noise: {noise_overlap:.1%}, "
            f"Wi-Fi-like: {wifi_overlap:.1%}"
        )
    else:
        print("No observed range intersection.")

print("\nAnalysis complete. No files modified.")
