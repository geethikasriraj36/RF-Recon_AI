import pandas as pd
import numpy as np

noise_file = r"real_dataset\real_noise_spectral_features.csv"
wifi_file = r"real_dataset\real_wifi_spectral_features.csv"

noise = pd.read_csv(noise_file)
wifi = pd.read_csv(wifi_file)

features = [
    "spectral_entropy",
    "spectral_peak_prominence",
    "number_of_spectral_peaks",
    "spectral_rolloff",
    "spectral_flatness"
]

print("=" * 75)
print("SPECTRAL FEATURE SEPARATION")
print("REAL NOISE vs REAL WIFI")
print("=" * 75)

results = []

for feature in features:

    n = noise[feature].dropna().values
    w = wifi[feature].dropna().values

    noise_mean = np.mean(n)
    wifi_mean = np.mean(w)

    noise_std = np.std(n)
    wifi_std = np.std(w)

    pooled_std = np.sqrt(
        (noise_std ** 2 + wifi_std ** 2) / 2
    )

    if pooled_std > 0:
        effect_size = abs(
            noise_mean - wifi_mean
        ) / pooled_std
    else:
        effect_size = 0

    overall_min = min(
        np.min(n),
        np.min(w)
    )

    overall_max = max(
        np.max(n),
        np.max(w)
    )

    overlap_low = max(
        np.min(n),
        np.min(w)
    )

    overlap_high = min(
        np.max(n),
        np.max(w)
    )

    if overlap_low <= overlap_high:
        denominator = overall_max - overall_min

        if denominator > 0:
            overlap = (
                (overlap_high - overlap_low)
                / denominator
                * 100
            )
        else:
            overlap = 0
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

output = r"real_dataset\spectral_noise_wifi_separation.csv"
results_df.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\nTOP SPECTRAL FEATURES")
print("-" * 75)

print(
    results_df[
        [
            "Feature",
            "Effect Size",
            "Range Overlap (%)"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("SPECTRAL SEPARATION ANALYSIS COMPLETE")
print("=" * 75)