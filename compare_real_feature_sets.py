
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

DATA_FILE = "real_dataset/real_capture_features_consistent.csv"

df = pd.read_csv(DATA_FILE)
df = df.sort_values(
    ["class_name", "window_index"]
).reset_index(drop=True)

feature_sets = {
    "All 8 features": [
        "peak_amplitude",
        "variance",
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "occupied_bandwidth_normalized",
        "spectral_flatness",
        "peak_to_average_ratio",
    ],
    "Spectral shape only": [
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "occupied_bandwidth_normalized",
        "spectral_flatness",
    ],
    "Without amplitude and variance": [
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "occupied_bandwidth_normalized",
        "spectral_flatness",
        "peak_to_average_ratio",
    ],
    "Frequency and shape only": [
        "peak_frequency_normalized",
        "spectral_centroid_normalized",
        "spectral_bandwidth_normalized",
        "occupied_bandwidth_normalized",
    ],
}

train_parts = []
test_parts = []

for label, group in df.groupby("class_name"):
    group = group.sort_values("window_index")
    split = int(len(group) * 0.70)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

y_train = train_df["class_name"]
y_test = test_df["class_name"]
labels = sorted(df["class_name"].unique())

results = []

for name, features in feature_sets.items():
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(train_df[features], y_train)
    predictions = model.predict(test_df[features])

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    results.append({
        "feature_set": name,
        "features": len(features),
        "accuracy_percent": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1, 4),
    })

    print(f"\n===== {name} =====")
    print("Features:", features)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    ))
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

result_df = pd.DataFrame(results)
result_df.to_csv(
    "real_dataset/real_feature_set_comparison.csv",
    index=False,
)

print("\n===== SUMMARY =====")
print(result_df.to_string(index=False))
print("\nSaved comparison CSV.")
