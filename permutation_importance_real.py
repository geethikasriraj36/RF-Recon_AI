
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

FILE = "real_dataset/real_capture_features_consistent.csv"

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

df = pd.read_csv(FILE).sort_values(
    ["class_name", "window_index"]
).reset_index(drop=True)

train_parts = []
test_parts = []

for _, group in df.groupby("class_name"):
    group = group.sort_values("window_index")
    split = int(len(group) * 0.70)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(train_df[FEATURES], train_df["class_name"])

X_test = test_df[FEATURES]
y_test = test_df["class_name"]

pred = model.predict(X_test)

print(f"Baseline accuracy: {accuracy_score(y_test, pred):.4f}")
print(
    "Baseline macro F1:",
    round(f1_score(y_test, pred, average="macro"), 4),
)

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "mean_macro_f1_drop": result.importances_mean,
    "std": result.importances_std,
}).sort_values("mean_macro_f1_drop", ascending=False)

print("\nPermutation importance (macro F1 decrease):")
print(importance_df.round(5).to_string(index=False))

importance_df.to_csv(
    "real_dataset/real_permutation_importance.csv",
    index=False,
)

print("\nSaved: real_dataset/real_permutation_importance.csv")
print("No saved models or capture files were modified.")
