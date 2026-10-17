import numpy as np
import pandas as pd

OLD_FILE = r"real_dataset\real_wifi_features.csv"
NEW_FILE = r"real_dataset\real_wifi_consistent_test.csv"
OUTPUT_FILE = r"real_dataset\wifi_feature_extractor_comparison.csv"

old = pd.read_csv(OLD_FILE).head(200)
new = pd.read_csv(NEW_FILE).head(200)

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

results = []

for feature in features:
    a = old[feature].to_numpy(dtype=float)
    b = new[feature].to_numpy(dtype=float)

    corr = np.corrcoef(a, b)[0, 1] if (
        np.std(a) > 0 and np.std(b) > 0
    ) else np.nan

    results.append({
        "feature": feature,
        "old_mean": np.mean(a),
        "new_mean": np.mean(b),
        "mean_absolute_difference": np.mean(np.abs(a - b)),
        "correlation": corr,
    })

result = pd.DataFrame(results)
result.to_csv(OUTPUT_FILE, index=False)

print("=" * 75)
print("RFRecon - OLD VS CONSISTENT FEATURE EXTRACTION")
print("=" * 75)
print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"\nSaved comparison: {OUTPUT_FILE}")
print("\nNo model or original dataset was modified.")
