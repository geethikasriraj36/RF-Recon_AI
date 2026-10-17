
import joblib
import numpy as np
import pandas as pd

from rfrecon_consistent_features import extract_features, FEATURE_NAMES

MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
CAPTURE_FILE = "real_dataset/pluto_rx_smoke_test.dat"

SAMPLE_RATE = 30_720_000
WINDOW_SIZE = 4096

model = joblib.load(MODEL_FILE)
features = joblib.load("rfrecon_four_class_best_features.pkl")
iq = np.fromfile(CAPTURE_FILE, dtype=np.complex64)

count = len(iq) // WINDOW_SIZE
raw_rows = []
normalized_rows = []

for i in range(count):
    window = iq[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]

    raw_rows.append(extract_features(window, SAMPLE_RATE))

    rms = np.sqrt(np.mean(np.abs(window) ** 2))
    if rms > 0:
        normalized_window = window / rms
    else:
        normalized_window = window

    normalized_rows.append(
        extract_features(normalized_window, SAMPLE_RATE)
    )

raw_X = pd.DataFrame(raw_rows, columns=features)
norm_X = pd.DataFrame(normalized_rows, columns=features)

raw_pred = model.predict(raw_X)
norm_pred = model.predict(norm_X)

raw_prob = model.predict_proba(raw_X).max(axis=1)
norm_prob = model.predict_proba(norm_X).max(axis=1)

print("\nRAW Pluto predictions:")
print(pd.Series(raw_pred).value_counts())
print("Mean maximum probability:", round(float(raw_prob.mean()), 4))

print("\nUNIT-RMS Pluto predictions (diagnostic only):")
print(pd.Series(norm_pred).value_counts())
print("Mean maximum probability:", round(float(norm_prob.mean()), 4))

print("\nPrediction agreement:", round(float(np.mean(raw_pred == norm_pred)), 4))
print("\nNo models or datasets were modified.")
print("Normalization does not establish protocol identity.")
