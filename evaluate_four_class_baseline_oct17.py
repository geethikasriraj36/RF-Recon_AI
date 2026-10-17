
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATASET = Path("real_dataset/four_class_baseline_features.csv")
MODEL_PATH = Path("rfrecon_four_class_baseline_oct17_rf.pkl")
FEATURES_PATH = Path("rfrecon_four_class_baseline_oct17_features.pkl")
RESULTS_PATH = Path("four_class_baseline_oct17_results.csv")
CONFUSION_PATH = Path("four_class_baseline_oct17_confusion_matrix.csv")
FEATURE_NAMES = [
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

df = pd.read_csv(DATASET)

required = FEATURE_NAMES + [
    "class_name", "source_file", "window_index"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isnull().any().any():
    raise ValueError("Required columns contain missing values.")

if not np.isfinite(df[FEATURE_NAMES].to_numpy()).all():
    raise ValueError("Features contain non-finite values.")

# Sort by capture time, then hold out the final 30% of each
# capture. Do not randomly split windows from the same recording.
train_parts = []
test_parts = []

for label, group in df.groupby("class_name"):
    group = group.sort_values("window_index").reset_index(drop=True)

    if group["source_file"].nunique() != 1:
        raise ValueError(
            f"{label}: expected one source capture for this baseline."
        )

    split_at = int(len(group) * 0.70)
    if split_at < 1 or split_at >= len(group):
        raise ValueError(f"Invalid split for class {label}")

    train_parts.append(group.iloc[:split_at])
    test_parts.append(group.iloc[split_at:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

X_train = train[FEATURE_NAMES]
y_train = train["class_name"]
X_test = test[FEATURE_NAMES]
y_test = test["class_name"]

print("Training rows:", len(train))
print("Held-out rows:", len(test))
print("\nTraining class counts:")
print(y_train.value_counts().to_string())
print("\nHeld-out class counts:")
print(y_test.value_counts().to_string())

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
labels = sorted(df["class_name"].unique())

accuracy = accuracy_score(y_test, predictions)
balanced = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

print("\n========== HELD-OUT RESULTS ==========")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, predictions, labels=labels, zero_division=0
))

cm = confusion_matrix(y_test, predictions, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels],
)
cm_df.to_csv(CONFUSION_PATH)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(RESULTS_PATH)

# Save as new files; do not overwrite existing models.
with MODEL_PATH.open("wb") as f:
    pickle.dump(model, f)

with FEATURES_PATH.open("wb") as f:
    pickle.dump(FEATURE_NAMES, f)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(cm_df.to_string())

print("\nSaved new baseline model:", MODEL_PATH)
print("Saved feature order:", FEATURES_PATH)
print("Saved metrics:", RESULTS_PATH)
print("Saved confusion matrix:", CONFUSION_PATH)
print("\nImportant: these are held-out windows from the same")
print("four captures, not independent recording sessions.")
print("Labels are intended capture classes, not independently")
print("verified protocol identities.")
