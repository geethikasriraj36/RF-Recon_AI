
from pathlib import Path
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

ROOT = Path("real_dataset")
OUTPUT = ROOT / "four_class_baseline_features.csv"

CAPTURES = {
    "noise": ROOT / "real_noise_01.dat",
    "wifi_like": ROOT / "real_wifi_01.dat",
    "bluetooth_like": ROOT / "real_bluetooth_01.dat",
    "cellular_like": ROOT / "real_cellular_01.dat",
}

SAMPLE_RATE = 20_000_000
WINDOW_SIZE = 4096
WINDOWS_PER_CAPTURE = 500
BYTES_PER_COMPLEX_SAMPLE = 8

rows = []

for label, path in CAPTURES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing capture: {path.resolve()}"
        )

    size = path.stat().st_size

    if size % BYTES_PER_COMPLEX_SAMPLE:
        raise ValueError(
            f"File size is not divisible by 8 bytes: {path}"
        )

    total_samples = size // BYTES_PER_COMPLEX_SAMPLE
    total_windows = total_samples // WINDOW_SIZE

    if total_windows < 2:
        raise ValueError(f"Not enough complete windows: {path}")

    count = min(WINDOWS_PER_CAPTURE, total_windows)

    # Keep selection deterministic and spread across the capture.
    selected = np.linspace(
        0, total_windows - 1, num=count, dtype=np.int64
    )

    print(
        f"\n{label}: {path.name} | "
        f"{total_samples:,} samples | "
        f"{count} windows"
    )

    with path.open("rb") as f:
        for index in selected:
            f.seek(int(index) * WINDOW_SIZE * BYTES_PER_COMPLEX_SAMPLE)

            iq = np.fromfile(
                f, dtype=np.complex64, count=WINDOW_SIZE
            )

            if iq.size != WINDOW_SIZE:
                raise ValueError(
                    f"Incomplete window: {path}, index {index}"
                )

            features = extract_features(iq, SAMPLE_RATE)

            row = dict(features)
            row["class_name"] = label
            row["source_file"] = path.name
            row["window_index"] = int(index)
            row["sample_rate_hz"] = SAMPLE_RATE

            # This is the configured frequency found in trial.py.
            # It is not independently verified RF calibration.
            row["configured_center_frequency_hz"] = (
                1_872_000_000
                if label == "cellular_like"
                else np.nan
            )

            rows.append(row)

df = pd.DataFrame(rows)

# Validate feature schema and values.
if list(df[FEATURE_NAMES].columns) != FEATURE_NAMES:
    raise RuntimeError("Feature column order mismatch.")

if not np.isfinite(df[FEATURE_NAMES].to_numpy()).all():
    raise ValueError("Dataset contains non-finite features.")

if df.duplicated(
    subset=["class_name", "source_file", "window_index"]
).any():
    raise ValueError("Duplicate capture windows found.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n========== FOUR-CLASS DATASET ==========")
print("Saved:", OUTPUT.resolve())
print("Rows, columns:", df.shape)
print("\nClass counts:")
print(df["class_name"].value_counts().to_string())
print("\nFeature columns:")
print(", ".join(FEATURE_NAMES))
print("\nConfigured cellular frequency: 1.872 GHz")
print("Note: labels reflect intended capture classes, not")
print("independent proof of protocol identity.")
print("\nExisting datasets and models were not overwritten.")
