import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


noise_file = r"real_dataset\real_noise_enhanced_features.csv"
wifi_file = r"real_dataset\real_wifi_enhanced_features.csv"

noise_spectral_file = r"real_dataset\real_noise_spectral_features.csv"
wifi_spectral_file = r"real_dataset\real_wifi_spectral_features.csv"


# ---------------------------------------------------------
# Load enhanced features
# ---------------------------------------------------------

noise = pd.read_csv(noise_file)
wifi = pd.read_csv(wifi_file)

noise_spectral = pd.read_csv(noise_spectral_file)
wifi_spectral = pd.read_csv(wifi_spectral_file)


# ---------------------------------------------------------
# Add spectral features
# ---------------------------------------------------------

spectral_features = [
    "spectral_entropy",
    "spectral_peak_prominence",
    "number_of_spectral_peaks",
    "spectral_rolloff",
    "spectral_flatness"
]

for feature in spectral_features:

    noise[feature] = noise_spectral[feature]
    wifi[feature] = wifi_spectral[feature]


# Remove labels if present
noise["label"] = "noise"
wifi["label"] = "wifi_like"


# ---------------------------------------------------------
# Feature list
# ---------------------------------------------------------

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
    "zero_crossing_rate",
    "amplitude_std",
    "amplitude_kurtosis",
    "power_std",
    "power_variation",
    "spectral_entropy",
    "spectral_peak_prominence",
    "number_of_spectral_peaks",
    "spectral_rolloff"
]


# Use spectral_flatness only once
features = list(dict.fromkeys(features))


# ---------------------------------------------------------
# Balance classes
# ---------------------------------------------------------

n_samples = min(
    len(noise),
    len(wifi)
)

noise = noise.sample(
    n=n_samples,
    random_state=42
)

wifi = wifi.sample(
    n=n_samples,
    random_state=42
)


data = pd.concat(
    [noise, wifi],
    ignore_index=True
)


X = data[features]
y = data["label"]


print("=" * 75)
print("REAL NOISE vs REAL WIFI CLASSIFICATION")
print("=" * 75)

print("\nDataset:")
print(data["label"].value_counts())

print("\nFeatures:", len(features))


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Features:")
print(
    importance.head(15).to_string(
        index=False
    )
)


importance.to_csv(
    r"real_dataset\real_noise_wifi_classifier_feature_importance.csv",
    index=False
)


print("\nSaved:")
print(
    r"real_dataset\real_noise_wifi_classifier_feature_importance.csv"
)

print("\n" + "=" * 75)
print("TEST COMPLETE")
print("=" * 75)