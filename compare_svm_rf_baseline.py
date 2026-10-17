
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_csv("real_dataset/four_class_baseline_features.csv")

features = [
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

train_parts, test_parts = [], []

for _, group in df.groupby("class_name"):
    group = group.sort_values("window_index").reset_index(drop=True)
    split = int(len(group) * 0.70)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

X_train, y_train = train[features], train["class_name"]
X_test, y_test = test[features], test["class_name"]
labels = sorted(df["class_name"].unique())

models = {
    "Random Forest - larger leaves": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
    ),
    "RBF SVM - scaled": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
        ),
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(
        y_test, pred, average="macro", zero_division=0
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    print(f"\n========== {name} ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(
        y_test, pred, labels=labels, zero_division=0
    ))
    print("Confusion matrix (rows=true):")
    print(pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels),
        index=labels,
        columns=labels,
    ).to_string())

out = pd.DataFrame(results)
out.to_csv("four_class_svm_rf_comparison.csv", index=False)

print("\n========== SUMMARY ==========")
print(out.to_string(index=False))
print("\nSaved: four_class_svm_rf_comparison.csv")
print("No existing models or datasets were overwritten.")
