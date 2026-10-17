
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix

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

MODEL_FILE = "rfrecon_hybrid_model.pkl"

FILES = {
    "noise": "real_dataset/real_noise_features.csv",
    "wifi_like": "real_dataset/real_wifi_features.csv",
    "bluetooth_like": "real_dataset/real_bluetooth_features.csv",
}

model = joblib.load(MODEL_FILE)

all_results = []

print("=" * 75)
print("RFRecon REAL-DATA BASELINE")
print("=" * 75)

for expected_class, path in FILES.items():

    df = pd.read_csv(path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\nSkipping {path}; missing columns: {missing}")
        continue

    X = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    valid = X.notna().all(axis=1)
    X = X.loc[valid]

    if X.empty:
        print(f"\nSkipping {path}; no valid feature rows")
        continue

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    confidence = probabilities.max(axis=1)

    distribution = pd.Series(predictions).value_counts()

    print(f"\nExpected capture label: {expected_class}")
    print(f"File: {path}")
    print(f"Valid windows: {len(X)}")
    print("Prediction distribution:")

    for cls in classes:
        count = int((predictions == cls).sum())
        print(f"  {cls:20s} {count:6d} ({100*count/len(X):6.2f}%)")

    dominant = distribution.index[0]

    print(f"Dominant prediction: {dominant}")
    print(f"Mean max probability: {confidence.mean()*100:.2f}%")

    all_results.append({
        "capture_label": expected_class,
        "file": path,
        "valid_windows": len(X),
        "dominant_prediction": dominant,
        "dominant_share_percent":
            100 * distribution.iloc[0] / len(X),
        "mean_max_probability_percent":
            100 * confidence.mean(),
    })

out = "real_dataset/real_data_baseline.csv"
pd.DataFrame(all_results).to_csv(out, index=False)

print("\nSaved:", out)
print("\nBaseline experiment complete.")
