
from pathlib import Path
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

CAPTURES = {
    "noise": Path("real_dataset/real_noise_01.dat"),
    "wifi_like": Path("real_dataset/real_wifi_01.dat"),
    "bluetooth_like": Path("real_dataset/real_bluetooth_01.dat"),
}

SAMPLE_RATE = 20_000_000
WINDOW_SIZE = 4096
WINDOWS_PER_CAPTURE = 500

rows = []

for label, path in CAPTURES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Capture not found: {path.resolve()}"
        )

    print(f"\nProcessing {label}: {path}")

    # Complex64 IQ samples occupy 8 bytes per sample.
    sample_count = path.stat().st_size // 8

    if path.stat().st_size % 8 != 0:
        raise ValueError(f"Unexpected file size: {path}")

    total_windows = sample_count // WINDOW_SIZE

    if total_windows == 0:
        raise ValueError(f"Capture is too short: {path}")

    # Spread selected windows across the entire recording.
    selected = np.linspace(
        0,
        total_windows - 1,
        num=min(WINDOWS_PER_CAPTURE, total_windows),
        dtype=int,
    )

    with path.open("rb") as file:
        for window_index in selected:
            file.seek(
                int(window_index) * WINDOW_SIZE * 8
            )

            raw = np.fromfile(
                file,
                dtype=np.complex64,
                count=WINDOW_SIZE,
            )

            if raw.size != WINDOW_SIZE:
                raise ValueError(
                    f"Incomplete window in {path}"
                )

            features = extract_features(
                raw,
                SAMPLE_RATE,
            )

            features["class_name"] = label
            features["source_file"] = path.name
            features["window_index"] = int(window_index)
            rows.append(features)

    print("Available windows:", total_windows)
    print("Selected windows:", len(selected))

df = pd.DataFrame(rows)

output = Path(
    "real_dataset/real_capture_features_consistent.csv"
)
df.to_csv(output, index=False)

print("\n========== DATASET SUMMARY ==========")
print("Output:", output.resolve())
print("Shape:", df.shape)
print("\nClass counts:")
print(df["class_name"].value_counts().to_string())

print("\nFeature means by capture:")
print(
    df.groupby("class_name")[FEATURE_NAMES]
    .mean()
    .round(5)
    .T.to_string()
)

print("\nSaved successfully. Original captures and models were not changed.")
