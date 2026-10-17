
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

FILES = {
    "noise_intended": Path("real_dataset/real_noise_features.csv"),
    "wifi_intended": Path("real_dataset/real_wifi_features.csv"),
    "bluetooth_intended": Path("real_dataset/real_bluetooth_features.csv"),
}

FEATURES = [
    "mean_power", "rms", "peak_amplitude", "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness", "peak_to_average_ratio",
]

frames = []

for label, path in FILES.items():
    df = pd.read_csv(path)
    df = df.replace([float("inf"), -float("inf")], float("nan"))
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    split = int(len(df) * 0.70)
    df["capture_label"] = label
    df["partition"] = "test"
    df.loc[:split - 1, "partition"] = "train"
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
train = data[data["partition"] == "train"]
test = data[data["partition"] == "test"]

X_train, y_train = train[FEATURES], train["capture_label"]
X_test, y_test = test[FEATURES], test["capture_label"]

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro"
        ),
    })

result = pd.DataFrame(results)
result.to_csv("real_dataset/baseline_model_comparison.csv", index=False)

print("=" * 65)
print("RFRecon - BASELINE MODEL COMPARISON")
print("=" * 65)
print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved: real_dataset/baseline_model_comparison.csv")
print("Existing RFRecon models were not changed.")
print("These are recording-label diagnostics, not verified protocol accuracy.")
