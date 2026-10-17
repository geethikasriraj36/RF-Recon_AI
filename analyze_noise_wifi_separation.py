import pandas as pd
import numpy as np

NOISE_FILE = r"real_dataset\real_noise_features.csv"
WIFI_FILE = r"real_dataset\real_wifi_features.csv"

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
    "peak_to_average_ratio"
]

noise = pd.read_csv(NOISE_FILE)
wifi = pd.read_csv(WIFI_FILE)

print("=" * 70)
print("RFRecon - REAL NOISE vs REAL WIFI FEATURE SEPARATION")
print("=" * 70)

print(f"\nNoise windows : {len(noise):,}")
print(f"WiFi windows  : {len(wifi):,}")

print("\nFEATURE STATISTICS")
print("-" * 70)

results = []

for feature in FEATURES:

    noise_values = noise[feature].values
    wifi_values = wifi[feature].values

    noise_mean = np.mean(noise_values)
    wifi_mean = np.mean(wifi_values)

    noise_std = np.std(noise_values)
    wifi_std = np.std(wifi_values)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        (noise_std**2 + wifi_std**2) / 2
    )

    if pooled_std > 0:

        effect_size = (
            abs(noise_mean - wifi_mean)
            / pooled_std
        )

    else:

        effect_size = 0

    # Overlap using combined range
    low = max(
        np.min(noise_values),
        np.min(wifi_values)
    )

    high = min(
        np.max(noise_values),
        np.max(wifi_values)
    )

    if high > low:

        overlap = high - low

        total_range = max(
            np.max(noise_values),
            np.max(wifi_values)
        ) - min(
            np.min(noise_values),
            np.min(wifi_values)
        )

        overlap_ratio = (
            overlap / total_range
            if total_range > 0
            else 1
        )

    else:

        overlap_ratio = 0

    results.append({
        "feature": feature,
        "noise_mean": noise_mean,
        "wifi_mean": wifi_mean,
        "noise_std": noise_std,
        "wifi_std": wifi_std,
        "effect_size": effect_size,
        "overlap_ratio": overlap_ratio
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "effect_size",
    ascending=False
)

for _, row in results_df.iterrows():

    print(
        f"\n{row['feature']}"
    )

    print(
        f"  Noise mean : "
        f"{row['noise_mean']:.6f}"
    )

    print(
        f"  WiFi mean  : "
        f"{row['wifi_mean']:.6f}"
    )

    print(
        f"  Noise std  : "
        f"{row['noise_std']:.6f}"
    )

    print(
        f"  WiFi std   : "
        f"{row['wifi_std']:.6f}"
    )

    print(
        f"  Effect size: "
        f"{row['effect_size']:.4f}"
    )

    print(
        f"  Range overlap: "
        f"{row['overlap_ratio']*100:.2f}%"
    )


print("\n" + "=" * 70)
print("FEATURE RANKING")
print("=" * 70)

print(
    results_df[
        [
            "feature",
            "effect_size",
            "overlap_ratio"
        ]
    ].to_string(index=False)
)


results_df.to_csv(
    "real_noise_wifi_feature_separation.csv",
    index=False
)

print("\nSaved:")
print("real_noise_wifi_feature_separation.csv")