import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

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

print("=" * 72)
print("RFRecon - REAL CAPTURE SEPARATION DIAGNOSTIC")
print("=" * 72)
print("Diagnostic models only; existing RFRecon models are untouched.")

train_parts = []
test_parts = []
labels = list(FILES.keys())

for label, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing features: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(X) < 20:
        raise ValueError(f"Too few valid windows in {path}")

    split = int(len(X) * 0.70)
    train_parts.append((X.iloc[:split].copy(), label))
    test_parts.append((X.iloc[split:].copy(), label))

    print(f"{label:<18} total={len(X):>7,}  train={split:>7,}  test={len(X)-split:>7,}")

X_train = pd.concat([x for x, y in train_parts], ignore_index=True)
y_train = np.concatenate([[y] * len(x) for x, y in train_parts])

X_test = pd.concat([x for x, y in test_parts], ignore_index=True)
y_test = np.concatenate([[y] * len(x) for x, y in test_parts])

# Equalize training class counts without duplicating samples.
rng = np.random.default_rng(42)
train_df = X_train.copy()
train_df["label"] = y_train
smallest = train_df["label"].value_counts().min()

balanced_parts = []
for label in labels:
    part = train_df[train_df["label"] == label]
    selected = rng.choice(part.index.to_numpy(), size=smallest, replace=False)
    balanced_parts.append(train_df.loc[selected])

balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42)
X_train_bal = balanced[FEATURES]
y_train_bal = balanced["label"].to_numpy()

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
}

summary_rows = []

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'=' * 72}")
    print(name)
    print(f"Chronological holdout accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{x}" for x in labels],
        columns=[f"predicted_{x}" for x in labels]
    )

    safe_name = name.lower().replace(" ", "_")
    cm_path = f"real_dataset/diagnostic_confusion_{safe_name}.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved: {cm_path}")

    summary_rows.append({
        "model": name,
        "chronological_holdout_accuracy": accuracy,
    })

summary_path = "real_dataset/diagnostic_separation_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print("\n" + "=" * 72)
print(f"Summary saved: {summary_path}")
print("These results apply only to the three existing captures.")
print("They do not establish performance on independent sessions.")