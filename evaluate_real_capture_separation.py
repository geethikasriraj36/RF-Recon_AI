
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FILES = {
    "noise_intended": Path("real_dataset/real_noise_features.csv"),
    "wifi_intended": Path("real_dataset/real_wifi_features.csv"),
    "bluetooth_intended": Path("real_dataset/real_bluetooth_features.csv"),
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

frames = []

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing features: {missing}")

    df = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    df["capture_label"] = label

    # Chronological split within each recording.
    split = int(len(df) * 0.70)
    df["partition"] = "test"
    df.iloc[:split, df.columns.get_loc("partition")] = "train"

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

train = data[data["partition"] == "train"]
test = data[data["partition"] == "test"]

X_train = train[FEATURES]
y_train = train["capture_label"]
X_test = test[FEATURES]
y_test = test["capture_label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

labels = list(FILES.keys())

print("=" * 75)
print("RFRecon - REAL CAPTURE SEPARATION DIAGNOSTIC")
print("=" * 75)
print(f"Training windows: {len(train)}")
print(f"Test windows: {len(test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, predictions, labels=labels, zero_division=0
))
print("Confusion matrix (rows=true capture, columns=predicted capture):")
print(labels)
print(confusion_matrix(y_test, predictions, labels=labels))

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    "real_dataset/capture_separation_feature_importance.csv",
    index=False,
)

print("\nFeature importance:")
print(importance.to_string(index=False))
print("\nSaved feature importance CSV.")
print("Only a temporary diagnostic model was trained; existing RFRecon models were not changed.")
print("Labels identify recordings, not independently verified RF protocols.")
