import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


noise_file = r"real_dataset\real_noise_large_window_features.csv"
wifi_file = r"real_dataset\real_wifi_large_window_features.csv"

noise = pd.read_csv(noise_file)
wifi = pd.read_csv(wifi_file)

noise["label"] = "noise"
wifi["label"] = "wifi_like"

# Balance the classes
n = min(len(noise), len(wifi))

noise = noise.sample(
    n=n,
    random_state=42
)

wifi = wifi.sample(
    n=n,
    random_state=42
)

data = pd.concat(
    [noise, wifi],
    ignore_index=True
)

features = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "spectral_entropy",
    "spectral_flatness",
    "peak_to_average_ratio"
]

X = data[features]
y = data["label"]

print("=" * 75)
print("LARGE-WINDOW REAL NOISE vs REAL WIFI CLASSIFICATION")
print("=" * 75)

print("\nDataset:")
print(y.value_counts())

print("\nWindow size: 32768 samples")
print("Duration per window: 1.6384 ms")
print("Features:", len(features))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    importance.to_string(
        index=False
    )
)

print("\n" + "=" * 75)
print("TEST COMPLETE")
print("=" * 75)