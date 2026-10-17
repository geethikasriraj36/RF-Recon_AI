import pandas as pd
import numpy as np

REAL_FILE = r"real_dataset\real_noise_dataset.csv"
HYBRID_FILE = r"hybrid_training_dataset.csv"

print("=" * 70)
print("RFRecon - REAL NOISE vs TRAINING DATA FEATURE ANALYSIS")
print("=" * 70)

real = pd.read_csv(REAL_FILE)
hybrid = pd.read_csv(HYBRID_FILE)

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
    "peak_to_average_ratio"
]

print("\nREAL NOISE DATASET")
print("-" * 70)

for f in features:
    if f in real.columns:
        print(
            f"{f:35s} "
            f"{real[f].mean():.6f}"
        )

print("\nTRAINING DATA BY CLASS")
print("-" * 70)

for cls in sorted(hybrid["label"].unique()):

    subset = hybrid[
        hybrid["label"] == cls
    ]

    print(f"\n{cls}")

    for f in features:

        if f in subset.columns:

            print(
                f"  {f:31s}"
                f"{subset[f].mean():.6f}"
            )

print("\n" + "=" * 70)
print("NEAREST CLASS BASED ON FEATURE MEANS")
print("=" * 70)

real_mean = real[features].mean()

distances = {}

for cls in hybrid["label"].unique():

    subset = hybrid[
        hybrid["label"] == cls
    ]

    class_mean = subset[features].mean()

    distance = np.linalg.norm(
        real_mean.values -
        class_mean.values
    )

    distances[cls] = distance

for cls, distance in sorted(
    distances.items(),
    key=lambda x: x[1]
):

    print(
        f"{cls:20s} "
        f"{distance:.6f}"
    )

print("\nAnalysis complete.")