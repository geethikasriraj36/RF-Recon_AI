
import joblib
import numpy as np
import pandas as pd

from rfrecon_consistent_features import extract_features, FEATURE_NAMES

MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
FEATURE_FILE = "rfrecon_four_class_best_features.pkl"
CAPTURE_FILE = "real_dataset/pluto_rx_smoke_test.dat"

SAMPLE_RATE = 30_720_000
WINDOW_SIZE = 4096

print("Loading four-class model...")
model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

if features != FEATURE_NAMES:
    raise ValueError("Model feature order does not match extractor.")

print("Loading Pluto IQ capture...")
iq = np.fromfile(CAPTURE_FILE, dtype=np.complex64)

if len(iq) < WINDOW_SIZE:
    raise ValueError("Capture is too short.")

num_windows = len(iq) // WINDOW_SIZE
print("IQ samples:", len(iq))
print("Complete windows:", num_windows)
print("Sample rate:", SAMPLE_RATE / 1e6, "MS/s")

rows = []

for i in range(num_windows):
    window = iq[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
    rows.append(extract_features(window, SAMPLE_RATE))

X = pd.DataFrame(rows, columns=features)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

results = pd.DataFrame({
    "window": np.arange(num_windows),
    "predicted_class": predictions,
    "model_probability": probabilities.max(axis=1),
})

print("\nPrediction distribution:")
print(results["predicted_class"].value_counts())

print("\nMean model probability by predicted class:")
print(
    results.groupby("predicted_class")["model_probability"]
    .mean()
    .sort_values(ascending=False)
)

output_file = "real_dataset/pluto_smoke_test_predictions.csv"
results.to_csv(output_file, index=False)

print("\nSaved predictions:", output_file)
print("Test completed. Predictions are not verified protocol identities.")
