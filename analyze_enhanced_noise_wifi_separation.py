import pandas as pd
import numpy as np

noise_file = r"real_dataset\real_noise_enhanced_features.csv"
wifi_file = r"real_dataset\real_wifi_enhanced_features.csv"

noise = pd.read_csv(noise_file)
wifi = pd.read_csv(wifi_file)

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
    "zero_crossing_rate",
    "amplitude_std",
    "amplitude_kurtosis",
    "power_std",
    "power_variation"
]

print("=" * 75)
print("ENHANCED REAL NOISE vs REAL WIFI FEATURE SEPARATION")
print("=" * 75)

results = []

for feature in features:

    n = noise[feature].dropna().values
    w = wifi[feature].dropna().values

    noise_mean = np.mean(n)
    wifi_mean = np.mean(w)

    noise_std = np.std(n)
    wifi_std = np.std(w)

    pooled_std = np.sqrt((noise_std**2 + wifi_std**2) / 2)

    if pooled_std > 0:
        effect_size = abs(noise_mean - wifi_mean) / pooled_std
    else:
        effect_size = 0

    min_max = max(np.min(n), np.min(w))
    max_min = min(np.max(n), np.max(w))

    if min_max <= max_min:
        overlap = (max_min - min_max) / (
            max(np.max(n), np.max(w)) -
            min(np.min(n), np.min(w))
        ) * 100
    else:
        overlap = 0

    results.append([
        feature,
        noise_mean,
        wifi_mean,
        effect_size,
        overlap
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Feature",
        "Noise Mean",
        "WiFi Mean",
        "Effect Size",
        "Range Overlap (%)"
    ]
)

results_df = results_df.sort_values(
    "Effect Size",
    ascending=False
)

print("\nRESULTS")
print("-" * 75)

for _, row in results_df.iterrows():

    print(
        f"{row['Feature']:32s} "
        f"Effect={row['Effect Size']:.4f}   "
        f"Overlap={row['Range Overlap (%)']:.2f}%"
    )

output = r"real_dataset\enhanced_noise_wifi_separation.csv"
results_df.to_csv(output, index=False)

print("\nSaved:")
print(output)

print("\nTOP FEATURES")
print("-" * 75)

print(
    results_df[
        ["Feature", "Effect Size", "Range Overlap (%)"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)