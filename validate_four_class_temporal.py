
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

# Evaluate three forward-in-time test blocks.
# Each test block is later than its corresponding training data.
fractions = [
    (0.50, 0.70),
    (0.60, 0.80),
    (0.70, 0.90),
]

results = []

for train_end, test_end in fractions:
    train_parts, test_parts = [], []

    for _, group in df.groupby("class_name"):
        group = group.sort_values("window_index").reset_index(drop=True)
        n = len(group)
        train_cut = int(n * train_end)
        test_cut = int(n * test_end)

        train_parts.append(group.iloc[:train_cut])
        test_parts.append(group.iloc[train_cut:test_cut])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
    )

    model.fit(train[features], train["class_name"])
    pred = model.predict(test[features])

    accuracy = accuracy_score(test["class_name"], pred)
    macro_f1 = f1_score(
        test["class_name"], pred,
        average="macro", zero_division=0
    )

    results.append({
        "train_fraction": train_end,
        "test_start_fraction": train_end,
        "test_end_fraction": test_end,
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    print(
        f"\nTrain 0-{train_end:.0%}; "
        f"test {train_end:.0%}-{test_end:.0%}"
    )
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

out = pd.DataFrame(results)
out.to_csv("four_class_temporal_validation.csv", index=False)

print("\n========== TEMPORAL VALIDATION SUMMARY ==========")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved: four_class_temporal_validation.csv")
print("This is same-recording validation, not independent-session testing.")
