
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

# --------------------------------------------------
# RFRecon four-class prediction configuration
# --------------------------------------------------
MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
FEATURE_FILE = "rfrecon_four_class_best_features.pkl"

import sys

CAPTURE_FILE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("real_dataset/pluto_rx_smoke_test.dat")
)

OUTPUT_FILE = Path("real_dataset/pluto_four_class_predictions.csv")
SAMPLE_RATE = 30_720_000
WINDOW_SIZE = 4096

# --------------------------------------------------
# Load and validate model
# --------------------------------------------------
print("=" * 60)
print("RFRecon - FOUR-CLASS RF PREDICTION")
print("=" * 60)

model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

if feature_names != FEATURE_NAMES:
    raise ValueError("Model and feature extractor do not match.")

expected_classes = {
    "noise",
    "wifi_like",
    "bluetooth_like",
    "cellular_like",
}

if set(model.classes_) != expected_classes:
    raise ValueError(
        f"Unexpected model classes: {list(model.classes_)}"
    )

# --------------------------------------------------
# Load complex IQ capture
# --------------------------------------------------
if not CAPTURE_FILE.exists():
    raise FileNotFoundError(
        f"Capture not found: {CAPTURE_FILE}"
    )

iq = np.fromfile(CAPTURE_FILE, dtype=np.complex64)

if not np.isfinite(iq).all():
    raise ValueError("Capture contains non-finite IQ samples.")

num_windows = len(iq) // WINDOW_SIZE

if num_windows == 0:
    raise ValueError("Capture is too short for one complete window.")

print("\nCapture:", CAPTURE_FILE)
print("Samples:", f"{len(iq):,}")
print("Sample rate:", f"{SAMPLE_RATE / 1e6:.2f} MS/s")
print("Window size:", WINDOW_SIZE)
print("Windows:", num_windows)

# --------------------------------------------------
# Consistent feature extraction and prediction
# --------------------------------------------------
feature_rows = []

for index in range(num_windows):
    start = index * WINDOW_SIZE
    window = iq[start:start + WINDOW_SIZE]

    feature_rows.append(
        extract_features(window, SAMPLE_RATE)
    )

X = pd.DataFrame(feature_rows, columns=feature_names)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

# --------------------------------------------------
# Save window-level results
# --------------------------------------------------
results = pd.DataFrame({
    "window": np.arange(num_windows),
    "predicted_class": predictions,
    "max_model_probability": probabilities.max(axis=1),
})

for index, class_name in enumerate(model.classes_):
    results[f"prob_{class_name}"] = probabilities[:, index]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Display summary
# --------------------------------------------------
print("\nPrediction counts:")
print(results["predicted_class"].value_counts())

print("\nAverage model probability by predicted class:")
print(
    results.groupby("predicted_class")["max_model_probability"]
    .mean()
    .round(4)
)

print("\nSaved results:", OUTPUT_FILE)
print("\nIMPORTANT:")
print("Predictions are model outputs, not verified protocol identities.")
print("This smoke-test capture is not labelled ground-truth data.")
print("=" * 60)
