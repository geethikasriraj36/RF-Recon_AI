import pandas as pd
import numpy as np

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

SOURCES = {
    "Synthetic training": "hybrid_training_dataset.csv",
    "Real noise": "real_dataset/real_noise_features.csv",
    "Real WiFi": "real_dataset/real_wifi_features.csv",
    "Real Bluetooth-intended": "real_dataset/real_bluetooth_features.csv",
}

print("=" * 90)
print("RFRecon - FEATURE SCALE INSPECTION")
print("=" * 90)

for name, path in SOURCES.items():
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"\nMissing file: {path}")
        continue

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\n{name}: missing columns {missing}")
        continue

    print(f"\n{name} | rows={len(df):,}")
    print(f"{'Feature':<36} {'Mean':>13} {'Median':>13} {'Min':>13} {'Max':>13}")
    print("-" * 90)

    for feature in FEATURES:
        x = pd.to_numeric(df[feature], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).dropna()

        if x.empty:
            continue

        print(
            f"{feature:<36} "
            f"{x.mean():>13.6g} "
            f"{x.median():>13.6g} "
            f"{x.min():>13.6g} "
            f"{x.max():>13.6g}"
        )

print("\nInspection complete. No files or models were modified.")
