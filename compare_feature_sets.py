
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

FILES = {
    "noise_intended": Path("real_dataset/real_noise_features.csv"),
    "wifi_intended": Path("real_dataset/real_wifi_features.csv"),
    "bluetooth_intended": Path("real_dataset/real_bluetooth_features.csv"),
}

ALL_FEATURES = [
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

FEATURE_SETS = {
    "all_10": ALL_FEATURES,
    "without_occupied_bandwidth": [
        f for f in ALL_FEATURES
        if f != "occupied_bandwidth_normalized"
    ],
    "spectral_only": [
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "spectral_flatness",
        "peak_to_average_ratio",
    ],
}

frames = []

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=ALL_FEATURES
    ).reset_index(drop=True)

    split = int(len(df) * 0.70)

    df["capture_label"] = label
    df["partition"] = "test"
    df.loc[:split - 1, "partition"] = "train"
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
train = data[data["partition"] == "train"]
test = data[data["partition"] == "test"]

results = []

for set_name, features in FEATURE_SETS.items():
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(train[features], train["capture_label"])
    pred = model.predict(test[features])

    results.append({
        "feature_set": set_name,
        "feature_count": len(features),
        "accuracy": accuracy_score(test["capture_label"], pred),
        "macro_f1": f1_score(
            test["capture_label"], pred, average="macro"
        ),
    })

result = pd.DataFrame(results)
result.to_csv(
    "real_dataset/feature_set_comparison.csv", index=False
)

print("=" * 65)
print("RFRecon - FEATURE SET COMPARISON")
print("=" * 65)
print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved: real_dataset/feature_set_comparison.csv")
print("Diagnostic only; existing RFRecon models were not modified.")
print("Labels refer to recordings, not verified RF protocols.")
