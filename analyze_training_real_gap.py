import os
import numpy as np
import pandas as pd

TRAIN_FILE = "hybrid_training_dataset.csv"

REAL_FILES = {
    "noise": "real_dataset/real_noise_features.csv",
    "wifi_like": "real_dataset/real_wifi_features.csv",
    "bluetooth_intended": "real_dataset/real_bluetooth_features.csv",
}

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

print("=" * 75)
print("RFRecon - TRAINING VS REAL DATA GAP")
print("=" * 75)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training CSV not found: {TRAIN_FILE}\n"
        "Check the filename before continuing."
    )

train = pd.read_csv(TRAIN_FILE)
missing = [f for f in FEATURES if f not in train.columns]
if missing:
    raise ValueError(f"Training CSV is missing features: {missing}")

label_col = "label"
if label_col not in train.columns:
    raise ValueError("Training CSV has no 'label' column.")

for feature in FEATURES:
    train[feature] = pd.to_numeric(train[feature], errors="coerce")

train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=FEATURES)

print(f"\nTraining rows available: {len(train):,}")
print("Training label counts:")
print(train[label_col].value_counts().to_string())

results = []

for real_label, path in REAL_FILES.items():
    if not os.path.exists(path):
        print(f"\nSkipping missing file: {path}")
        continue

    real = pd.read_csv(path)
    real = real.replace([np.inf, -np.inf], np.nan)

    for feature in FEATURES:
        real[feature] = pd.to_numeric(real[feature], errors="coerce")

    real = real.dropna(subset=FEATURES)

    print(f"\n{'-' * 75}")
    print(f"REAL CAPTURE: {real_label} | windows={len(real):,}")

    for train_label, group in train.groupby(label_col):
        rows = []

        for feature in FEATURES:
            a = group[feature].dropna().to_numpy()
            b = real[feature].dropna().to_numpy()

            if len(a) < 2 or len(b) < 2:
                continue

            mean_a, mean_b = np.mean(a), np.mean(b)
            std_a, std_b = np.std(a), np.std(b)

            pooled = np.sqrt((std_a**2 + std_b**2) / 2)
            effect = (mean_b - mean_a) / pooled if pooled > 1e-12 else 0.0

            lo = max(np.quantile(a, 0.25), np.quantile(b, 0.25))
            hi = min(np.quantile(a, 0.75), np.quantile(b, 0.75))
            overlap = max(0.0, hi - lo)

            scale = max(
                np.quantile(a, 0.75) - np.quantile(a, 0.25),
                np.quantile(b, 0.75) - np.quantile(b, 0.25),
                1e-12
            )

            rows.append({
                "real_capture": real_label,
                "training_class": train_label,
                "feature": feature,
                "training_mean": mean_a,
                "real_mean": mean_b,
                "standardized_mean_difference": effect,
                "absolute_effect_size": abs(effect),
                "training_q25": np.quantile(a, 0.25),
                "training_q75": np.quantile(a, 0.75),
                "real_q25": np.quantile(b, 0.25),
                "real_q75": np.quantile(b, 0.75),
                "middle_50pct_overlap_width": overlap,
                "relative_overlap_width": overlap / scale,
            })

        comparison = pd.DataFrame(rows)
        if comparison.empty:
            continue

        comparison = comparison.sort_values(
            "absolute_effect_size", ascending=False
        )

        print(f"\nClosest comparison by training class: {train_label}")
        for _, row in comparison.head(3).iterrows():
            print(
                f"  {row['feature']:<35} "
                f"effect={row['standardized_mean_difference']:+.3f}"
            )

        results.extend(rows)

if not results:
    raise SystemExit("No comparisons completed. Check the CSV files and columns.")

out = pd.DataFrame(results)
out_path = "real_dataset/training_real_gap.csv"
out.to_csv(out_path, index=False)

print("\n" + "=" * 75)
print(f"Saved detailed comparison: {out_path}")
print("This is a distribution comparison, not an accuracy measurement.")
print("No models or datasets were modified.")
