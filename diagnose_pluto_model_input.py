
import joblib
import numpy as np
import pandas as pd

from rfrecon_consistent_features import extract_features, FEATURE_NAMES

DATA_FILE = "real_dataset/four_class_baseline_features.csv"
CAPTURE_FILE = "real_dataset/pluto_rx_smoke_test.dat"
MODEL_FILE = "rfrecon_four_class_best_rf.pkl"

SAMPLE_RATE = 30_720_000
WINDOW_SIZE = 4096

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

iq = np.fromfile(CAPTURE_FILE, dtype=np.complex64)
n = len(iq) // WINDOW_SIZE

rows = [
    extract_features(iq[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE], SAMPLE_RATE)
    for i in range(n)
]
pluto = pd.DataFrame(rows, columns=FEATURE_NAMES)

print("\n=== Training feature ranges ===")
print(df[FEATURE_NAMES].describe().loc[["min", "50%", "max"]].T.round(6))

print("\n=== Pluto feature ranges ===")
print(pluto.describe().loc[["min", "50%", "max"]].T.round(6))

print("\n=== Pluto medians outside training min/max ===")
train_min = df[FEATURE_NAMES].min()
train_max = df[FEATURE_NAMES].max()
pluto_median = pluto.median()

for feature in FEATURE_NAMES:
    outside = (
        pluto_median[feature] < train_min[feature]
        or pluto_median[feature] > train_max[feature]
    )
    print(
        f"{feature}: median={pluto_median[feature]:.6g}, "
        f"training=[{train_min[feature]:.6g}, {train_max[feature]:.6g}], "
        f"outside={outside}"
    )

print("\n=== Predicted class probabilities (first 5 windows) ===")
probs = model.predict_proba(pluto)
for i in range(min(5, n)):
    order = np.argsort(probs[i])[::-1]
    print(
        f"Window {i}: "
        + ", ".join(
            f"{model.classes_[j]}={probs[i, j]:.3f}"
            for j in order
        )
    )
