
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_FILE = "robust_training_dataset_consistent.csv"

df = pd.read_csv(DATA_FILE)

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

# Remove only features with effectively zero variation.
USEFUL_FEATURES = [
    feature for feature in ALL_FEATURES
    if df[feature].std() > 1e-6
]

FEATURE_SETS = {
    "All 10 features": ALL_FEATURES,
    "Remove constant power features": USEFUL_FEATURES,
    "Time-domain features only": [
        "peak_amplitude",
        "variance",
    ],
    "Spectral features only": [
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "occupied_bandwidth_normalized",
        "spectral_flatness",
        "peak_to_average_ratio",
    ],
}

X_labels = df["class_name"]

# Reuse exactly the same split for every experiment.
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.25,
    random_state=42,
    stratify=X_labels,
)

y_train = X_labels.loc[train_idx]
y_test = X_labels.loc[test_idx]

results = []

for set_name, features in FEATURE_SETS.items():
    X_train = df.loc[train_idx, features]
    X_test = df.loc[test_idx, features]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    results.append({
        "feature_set": set_name,
        "feature_count": len(features),
        "accuracy_percent": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1, 4),
    })

    print(f"\n{set_name}")
    print("Features:", features)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(
    "consistent_feature_comparison.csv",
    index=False,
)

print("\n========== COMPARISON SUMMARY ==========")
print(results_df.to_string(index=False))
print("\nSaved: consistent_feature_comparison.csv")
