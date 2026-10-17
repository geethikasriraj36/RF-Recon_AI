
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_FILE = Path(
    "real_dataset/real_capture_features_consistent.csv"
)

FEATURES = [
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
]

df = pd.read_csv(DATA_FILE)

# Sort windows in their original chronological order.
df = df.sort_values(
    ["class_name", "window_index"]
).reset_index(drop=True)

train_parts = []
test_parts = []

for label, group in df.groupby("class_name"):
    group = group.sort_values("window_index")
    split = int(len(group) * 0.70)

    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

X_train = train_df[FEATURES]
y_train = train_df["class_name"]
X_test = test_df[FEATURES]
y_test = test_df["class_name"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Features:", len(FEATURES))

print(
    "\nAccuracy:",
    f"{accuracy_score(y_test, predictions) * 100:.2f}%"
)
print(
    "Macro F1:",
    f"{f1_score(y_test, predictions, average='macro'):.4f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

labels = sorted(df["class_name"].unique())

print("Confusion matrix (rows=true, columns=predicted):")
print(
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    )
)

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance.round(4).to_string())

importance.to_csv(
    "real_dataset/consistent_real_feature_importance.csv",
    header=["importance"],
)

print(
    "\nSaved feature importance to "
    "real_dataset/consistent_real_feature_importance.csv"
)
print("Existing trained models were not modified.")
