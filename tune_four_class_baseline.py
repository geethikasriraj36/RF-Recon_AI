
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv(
    "real_dataset/four_class_baseline_features.csv"
)

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

configs = [
    {
        "name": "Baseline balanced",
        "class_weight": "balanced",
        "max_depth": None,
        "min_samples_leaf": 1,
    },
    {
        "name": "Shallower trees",
        "class_weight": "balanced",
        "max_depth": 12,
        "min_samples_leaf": 2,
    },
    {
        "name": "Larger leaves",
        "class_weight": "balanced",
        "max_depth": None,
        "min_samples_leaf": 5,
    },
]

results = []

for config in configs:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight=config["class_weight"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = {
        "configuration": config["name"],
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(
            y_test, pred, average="macro", zero_division=0
        ),
    }
    results.append(result)

    print(f"\n=== {config['name']} ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(pd.crosstab(
        pd.Series(y_test, name="True"),
        pd.Series(pred, name="Predicted"),
    ).to_string())

out = pd.DataFrame(results)
out.to_csv("four_class_rf_tuning_results.csv", index=False)

print("\n=== SUMMARY ===")
print(out.to_string(index=False))
print("\nSaved: four_class_rf_tuning_results.csv")
print("Existing models and datasets were not overwritten.")
