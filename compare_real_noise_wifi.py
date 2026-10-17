import pandas as pd
import numpy as np

REAL_NOISE = r"real_dataset\real_noise_features.csv"
REAL_WIFI = r"real_dataset\real_wifi_features.csv"
HYBRID = r"hybrid_training_dataset.csv"

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

noise = pd.read_csv(REAL_NOISE)
wifi = pd.read_csv(REAL_WIFI)
hybrid = pd.read_csv(HYBRID)

print("=" * 70)
print("RFRecon - REAL NOISE vs REAL WIFI FEATURE COMPARISON")
print("=" * 70)

print("\nREAL NOISE MEANS")
print("-" * 70)

for f in FEATURES:
    print(f"{f:35s}{noise[f].mean():.6f}")

print("\nREAL WIFI MEANS")
print("-" * 70)

for f in FEATURES:
    print(f"{f:35s}{wifi[f].mean():.6f}")

print("\nSYNTHETIC NOISE MEANS")
print("-" * 70)

synthetic_noise = hybrid[
    hybrid["label"] == "noise"
]

for f in FEATURES:
    print(f"{f:35s}{synthetic_noise[f].mean():.6f}")

print("\nSYNTHETIC WIFI MEANS")
print("-" * 70)

synthetic_wifi = hybrid[
    hybrid["label"] == "wifi_like"
]

for f in FEATURES:
    print(f"{f:35s}{synthetic_wifi[f].mean():.6f}")


# ---------------------------------------------------------
# NORMALIZED FEATURE DISTANCES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE DISTANCE ANALYSIS")
print("=" * 70)

# Use real WiFi scale to avoid absolute-power features
# dominating everything.

combined = pd.concat([
    noise[FEATURES],
    wifi[FEATURES],
    synthetic_noise[FEATURES],
    synthetic_wifi[FEATURES]
])

feature_min = combined.min()
feature_max = combined.max()

scale = feature_max - feature_min
scale[scale == 0] = 1

def normalize(series):
    return (series - feature_min) / scale


noise_mean = normalize(
    noise[FEATURES].mean()
)

wifi_mean = normalize(
    wifi[FEATURES].mean()
)

synthetic_noise_mean = normalize(
    synthetic_noise[FEATURES].mean()
)

synthetic_wifi_mean = normalize(
    synthetic_wifi[FEATURES].mean()
)

print("\nReal Noise → Real WiFi distance:")

real_noise_wifi_distance = np.linalg.norm(
    noise_mean.values -
    wifi_mean.values
)

print(
    f"{real_noise_wifi_distance:.6f}"
)

print("\nReal Noise → Synthetic Noise distance:")

real_noise_synthetic_noise = np.linalg.norm(
    noise_mean.values -
    synthetic_noise_mean.values
)

print(
    f"{real_noise_synthetic_noise:.6f}"
)

print("\nReal WiFi → Synthetic WiFi distance:")

real_wifi_synthetic_wifi = np.linalg.norm(
    wifi_mean.values -
    synthetic_wifi_mean.values
)

print(
    f"{real_wifi_synthetic_wifi:.6f}"
)

print("\nReal Noise → Synthetic WiFi distance:")

real_noise_synthetic_wifi = np.linalg.norm(
    noise_mean.values -
    synthetic_wifi_mean.values
)

print(
    f"{real_noise_synthetic_wifi:.6f}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)