import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_FILE = Path("hybrid_training_dataset.csv")
REAL_FILES = {
    "noise": Path("real_dataset/real_noise_features.csv"),
    "wifi_like": Path("real_dataset/real_wifi_features.csv"),
    "bluetooth_like": Path("real_dataset/real_bluetooth_features.csv"),
    "cellular_like": Path("real_dataset/real_cellular_features.csv"),
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

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {TRAIN_FILE}. Check the project directory."
    )

train = pd.read_csv(TRAIN_FILE)

if "class_name" not in train.columns:
    raise ValueError(
        "Training CSV has no 'class_name' column. "
        f"Available columns: {list(train.columns)}"
    )

missing = [f for f in FEATURES if f not in train.columns]
if missing:
    raise ValueError(f"Missing training features: {missing}")

print("=" * 80)
print("RFRecon - TRAINING VS REAL FEATURE SCALE AUDIT")
print("=" * 80)

print("\nTRAINING DATASET")
print("Rows:", len(train))
print("\nClass counts:")
print(train["class_name"].value_counts().to_string())

results = []

for label, path in REAL_FILES.items():
    if not path.exists():
        print(f"\nSkipping {label}: file not found: {path}")
        continue

    real = pd.read_csv(path)
    available = [f for f in FEATURES if f in real.columns]

    print(f"\n{'-' * 80}")
    print(f"REAL CAPTURE: {label} | rows={len(real)}")

    for feature in available:
        train_values = pd.to_numeric(
            train[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        real_values = pd.to_numeric(
            real[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(train_values) == 0 or len(real_values) == 0:
            continue

        train_median = train_values.median()
        real_median = real_values.median()

        results.append({
            "real_class": label,
            "feature": feature,
            "training_median": train_median,
            "real_median": real_median,
            "training_mean": train_values.mean(),
            "real_mean": real_values.mean(),
            "median_abs_difference": abs(
                train_median - real_median
            ),
            "training_q25": train_values.quantile(0.25),
            "training_q75": train_values.quantile(0.75),
            "real_q25": real_values.quantile(0.25),
            "real_q75": real_values.quantile(0.75),
        })

    subset = [
        r for r in results if r["real_class"] == label
    ]
    for r in subset:
        print(
            f"{r['feature']:<36} "
            f"train median={r['training_median']:.6g} | "
            f"real median={r['real_median']:.6g}"
        )

output = Path("real_dataset/feature_scale_audit.csv")
pd.DataFrame(results).to_csv(output, index=False)

print(f"\nSaved detailed audit: {output}")
print("\nRead-only audit: no datasets or models were modified.")
