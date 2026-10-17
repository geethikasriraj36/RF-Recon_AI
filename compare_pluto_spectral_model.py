
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_FILE = "real_dataset/four_class_baseline_features.csv"

df = pd.read_csv(DATA_FILE)

spectral_features = [
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
]

X = df[spectral_features]
y = df["class_name"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Spectral-only model evaluation")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(f1_score(y_test, predictions, average="macro"), 4))
print("\nFeature importance:")
for name, importance in sorted(
    zip(spectral_features, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{name}: {importance:.4f}")

print("\nDiagnostic only: this does not validate Pluto protocol identity.")
print("No existing model or dataset was modified.")
