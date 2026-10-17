
from pathlib import Path

import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

DATA_FILE = Path("real_dataset/four_class_baseline_features.csv")
CAPTURE_FILES = [
    Path("real_dataset/pluto_rx_smoke_test.dat"),
    Path("real_dataset/pluto_fresh_20261014_202446.dat"),
]

SAMPLE_RATE = 30_720_000
WINDOW_SIZE = 4096

df = pd.read_csv(DATA_FILE)

print("TRAINING DATA CLASS SUMMARY")
print("=" * 65)

for label, group in df.groupby("class_name"):
    print(f"\n{label}: {len(group)} windows")
    print(
        group[FEATURE_NAMES]
        .median()
        .to_string(float_format=lambda x: f"{x:.6g}")
    )

for capture in CAPTURE_FILES:
    if not capture.exists():
        print(f"\nSkipping missing capture: {capture}")
        continue

    iq = np.fromfile(capture, dtype=np.complex64)
    count = len(iq) // WINDOW_SIZE

    if count == 0:
        print(f"\nSkipping short capture: {capture}")
        continue

    rows = []
    for i in range(count):
        window = iq[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
        rows.append(extract_features(window, SAMPLE_RATE))

    live = pd.DataFrame(rows, columns=FEATURE_NAMES)

    print("\n" + "=" * 65)
    print(f"CAPTURE: {capture}")
    print(f"Complete windows: {count}")
    print("\nMedian feature comparison")

    comparison = pd.DataFrame({
        "Pluto median": live[FEATURE_NAMES].median()
    })

    for label, group in df.groupby("class_name"):
        comparison[f"{label} median"] = (
            group[FEATURE_NAMES].median()
        )

    print(comparison.to_string(
        float_format=lambda x: f"{x:.6g}"
    ))

    print("\nPower/amplitude range check")
    for feature in [
        "mean_power",
        "rms",
        "peak_amplitude",
        "variance",
    ]:
        live_median = live[feature].median()

        print(
            f"{feature}: Pluto median={live_median:.6g}"
        )

print("\nDiagnostic complete.")
print("Feature similarity does not prove protocol identity.")
print("No models or recordings were modified.")
