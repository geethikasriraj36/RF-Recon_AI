import os
import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "rfrecon_hybrid_model.pkl"

FILES = {
    "noise": "real_dataset/real_noise_features.csv",
    "wifi_like": "real_dataset/real_wifi_features.csv",
    "bluetooth_like": "real_dataset/real_bluetooth_features.csv",
}

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

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)

print("=" * 72)
print("RFRecon - TEMPORAL STABILITY TEST")
print("=" * 72)
print("This evaluates the existing model only. No training is performed.")

results = []

for expected_label, path in FILES.items():
    if not os.path.exists(path):
        print(f"\nSKIPPED: missing file {path}")
        continue

    df = pd.read_csv(path)
    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        print(f"\nSKIPPED: {path}; missing features: {missing}")
        continue

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)

    if len(X) < 10:
        print(f"\nSKIPPED: too few valid rows in {path}")
        continue

    midpoint = len(X) // 2
    segments = {
        "First half": X.iloc[:midpoint],
        "Second half": X.iloc[midpoint:],
    }

    print(f"\nCapture: {expected_label}")
    print(f"File: {path}")
    print(f"Valid windows: {len(X):,}")

    for segment_name, segment in segments.items():
        predictions = model.predict(segment[FEATURES])
        counts = pd.Series(predictions).value_counts(normalize=True) * 100
        dominant = counts.index[0]

        print(f"\n  {segment_name} ({len(segment):,} windows)")
        print(f"  Dominant prediction: {dominant}")
        print(f"  Dominant share: {counts.iloc[0]:.2f}%")

        for label, percentage in counts.items():
            print(f"    {label:<20} {percentage:6.2f}%")

        results.append({
            "expected_capture_label": expected_label,
            "segment": segment_name,
            "window_count": len(segment),
            "dominant_prediction": dominant,
            "dominant_share_percent": counts.iloc[0],
        })

output = "real_dataset/real_temporal_stability.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("\n" + "=" * 72)
print(f"Saved results: {output}")
print("Reminder: windows from one capture are not independent sessions.")
print("A stable prediction is not proof that the prediction is correct.")
