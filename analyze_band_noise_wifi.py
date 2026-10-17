import pandas as pd
import numpy as np

noise_file = r"real_dataset\real_noise_band_features.csv"
wifi_file = r"real_dataset\real_wifi_band_features.csv"

noise = pd.read_csv(noise_file)
wifi = pd.read_csv(wifi_file)

features = [
    f"band_power_{i:02d}"
    for i in range(1, 11)
]

print("=" * 75)
print("SPECTRAL BAND-POWER SEPARATION")
print("REAL NOISE vs REAL WIFI")
print("=" * 75)

results = []

for feature in features:

    n = noise[feature].values
    w = wifi[feature].values

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

    results.append([
        feature,
        noise_mean,
        wifi_mean,
        effect_size
    ])


results_df = pd.DataFrame(
    results,
    columns=[
        "Feature",
        "Noise Mean",
        "WiFi Mean",
        "Effect Size"
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
        f"{row['Feature']:20s} "
        f"Noise={row['Noise Mean']:.6f}   "
        f"WiFi={row['WiFi Mean']:.6f}   "
        f"Effect={row['Effect Size']:.4f}"
    )


output = (
    r"real_dataset\band_noise_wifi_separation.csv"
)

results_df.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\n" + "=" * 75)
print("BAND SEPARATION ANALYSIS COMPLETE")
print("=" * 75)