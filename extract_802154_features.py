
import os
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

IQ_PATH = r"real_dataset\6lowpan_ping.dat"
OUTPUT_CSV = r"real_dataset\ieee802154_6lowpan_features.csv"

# Flowgraph setting only; the recording's actual sample rate
# must be verified independently before final model training.
SAMPLE_RATE_HZ = 4_000_000

WINDOW_SIZE = 4096
MAX_WINDOWS = 500

file_size = os.path.getsize(IQ_PATH)
if file_size % 8 != 0:
    raise ValueError("File size is not aligned to complex64 IQ samples.")

total_samples = file_size // 8
available_windows = total_samples // WINDOW_SIZE
windows_to_use = min(MAX_WINDOWS, available_windows)

print("File size:", file_size, "bytes")
print("Total complex64 samples:", total_samples)
print("Available full windows:", available_windows)
print("Windows to process:", windows_to_use)

rows = []

with open(IQ_PATH, "rb") as f:
    # Select windows spread across the recording.
    indices = np.linspace(
        0,
        available_windows - 1,
        num=windows_to_use,
        dtype=int,
    )

    for window_index in indices:
        f.seek(int(window_index) * WINDOW_SIZE * 8)
        iq = np.fromfile(f, dtype=np.complex64, count=WINDOW_SIZE)

        if len(iq) != WINDOW_SIZE:
            continue

        features = extract_features(iq, SAMPLE_RATE_HZ)

        row = {name: features[name] for name in FEATURE_NAMES}
        row.update({
            "class_name": "ieee802154_6lowpan",
            "source_file": os.path.basename(IQ_PATH),
            "capture_id": "6lowpan_ping",
            "window_index": int(window_index),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_rate_verified": False,
            "configured_center_frequency_hz": 2_415_000_000,
            "center_frequency_verified_for_capture": False,
            "rf_bandwidth_hz": np.nan,
            "rf_bandwidth_verified": False,
            "capture_gain": "unknown",
            "data_origin": "external_real_iq",
            "protocol_label_status": "IEEE 802.15.4 / 6LoWPAN source; Zigbee unverified",
        })
        rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No feature windows were extracted.")

if df[FEATURE_NAMES].isna().any().any():
    raise RuntimeError("Missing feature values detected.")

if not np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all():
    raise RuntimeError("Non-finite feature values detected.")

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print("Rows:", len(df))
print("Feature columns in order:", list(df[FEATURE_NAMES].columns))
print("Missing feature values:", int(df[FEATURE_NAMES].isna().sum().sum()))
print("All feature values finite:",
      bool(np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all()))
print("Class label:", df["class_name"].iloc[0])
print("Capture settings marked verified: False")
