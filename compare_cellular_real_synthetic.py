import pandas as pd
import numpy as np

REAL_FILE = r"real_dataset\real_cellular_features.csv"
SYNTHETIC_FILE = r"hybrid_training_dataset.csv"

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

real = pd.read_csv(REAL_FILE)
synthetic_all = pd.read_csv(SYNTHETIC_FILE)

synthetic = synthetic_all[
    synthetic_all["label"] == "cellular_like"
]

print("=" * 70)
print("RFRecon - REAL vs SYNTHETIC CELLULAR COMPARISON")
print("=" * 70)

print(f"Real cellular windows      : {len(real)}")
print(f"Synthetic cellular windows : {len(synthetic)}")

print()
print("-" * 70)
print(f"{'Feature':<35} {'Real':>12} {'Synthetic':>12}")
print("-" * 70)

for feature in FEATURES:

    real_mean = real[feature].mean()
    synthetic_mean = synthetic[feature].mean()

    print(
        f"{feature:<35} "
        f"{real_mean:>12.6f} "
        f"{synthetic_mean:>12.6f}"
    )

print("-" * 70)

# Calculate ratio
print()
print("Relative difference:")

for feature in FEATURES:

    r = real[feature].mean()
    s = synthetic[feature].mean()

    if abs(s) > 1e-12:
        difference = ((r - s) / abs(s)) * 100
    else:
        difference = np.nan

    print(
        f"{feature:<35}: {difference:+.2f}%"
    )

print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)