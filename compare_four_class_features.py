
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATASET = Path("real_dataset/four_class_baseline_features.csv")

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

SPECTRAL_FEATURES = [
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
]

df = pd.read_csv(DATASET)
train_parts, test_parts = [], []

for label, group in df.groupby("class_name"):
    group = group.sort_values("window_index").reset_index(drop=True)
    split_at = int(len(group) * 0.70)
    train_parts.append(group.iloc[:split_at])
    test_parts.append(group.iloc[split_at:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

y_train = train["class_name"]
y_test = test["class_name"]
labels = sorted(df["class_name"].unique())

experiments = {
    "All 10 features": ALL_FEATURES,
    "Spectral 6 features": SPECTRAL_FEATURES,
}

summary = []

for name, features in experiments.items():
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(train[features], y_train)
    pred = model.predict(test[features])

    accuracy = accuracy_score(y_test, pred)
    balanced = balanced_accuracy_score(y_test, pred)
    macro_f1 = f1_score(
        y_test, pred, average="macro", zero_division=0
    )

    summary.append({
        "experiment": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
    })

    print(f"\n========== {name} ==========")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, pred, labels=labels, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels),
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    ).to_string())

out = pd.DataFrame(summary)
out.to_csv("four_class_feature_comparison.csv", index=False)

print("\n========== COMPARISON ==========")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved: four_class_feature_comparison.csv")
print("No existing model or dataset was overwritten.")
