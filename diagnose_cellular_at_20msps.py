
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

CAPTURE = Path("real_dataset/real_cellular_01.dat")
MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
FEATURE_FILE = "rfrecon_four_class_best_features.pkl"

SAMPLE_RATE = 20_000_000
WINDOW_SIZE = 4096
MAX_WINDOWS = 2000

print("Loading model...")
model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

if feature_names != FEATURE_NAMES:
    raise ValueError("Feature extractor and model feature order differ.")

if not CAPTURE.exists():
    raise FileNotFoundError(CAPTURE)

total_samples = CAPTURE.stat().st_size // 8
total_windows = total_samples // WINDOW_SIZE

# Sample windows across the whole recording without loading
# the entire multi-gigabyte capture into RAM.
indices = np.linspace(
    0,
    total_windows - 1,
    num=min(MAX_WINDOWS, total_windows),
    dtype=np.int64,
)

rows = []

print(f"Capture: {CAPTURE}")
print(f"Samples: {total_samples:,}")
print(f"Assumed sample rate: {SAMPLE_RATE / 1e6:.2f} MS/s")
print(f"Windows analyzed: {len(indices):,}")

with CAPTURE.open("rb") as f:
    for number, index in enumerate(indices, start=1):
        f.seek(int(index) * WINDOW_SIZE * 8)
        iq = np.fromfile(f, dtype=np.complex64, count=WINDOW_SIZE)

        if len(iq) != WINDOW_SIZE:
            continue

        features = extract_features(iq, SAMPLE_RATE)
        rows.append(features)

        if number % 250 == 0:
            print(f"Processed {number}/{len(indices)} windows")

X = pd.DataFrame(rows, columns=feature_names)

if X.empty:
    raise RuntimeError("No complete windows were processed.")

predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("\nPrediction counts:")
print(pd.Series(predictions).value_counts().to_string())

print("\nMean model probabilities:")
mean_probabilities = probabilities.mean(axis=0)

for name, probability in zip(model.classes_, mean_probabilities):
    print(f"{name:20s}: {probability:.4f}")

print("\nMean maximum model probability:")
print(f"{probabilities.max(axis=1).mean():.4f}")

print("\nDiagnostic complete.")
print("The 20 MS/s rate is assumed from trial.py, not independently verified.")
print("Predictions do not prove protocol identity.")
print("No model or capture was modified.")
