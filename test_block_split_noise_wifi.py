import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


noise_file = r"real_dataset\real_noise_enhanced_features.csv"
wifi_file = r"real_dataset\real_wifi_enhanced_features.csv"

noise_spectral_file = r"real_dataset\real_noise_spectral_features.csv"
wifi_spectral_file = r"real_dataset\real_wifi_spectral_features.csv"


# ---------------------------------------------------------
# Load data
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


noise["label"] = "noise"
wifi["label"] = "wifi_like"


# ---------------------------------------------------------
# Balance the two classes
# ---------------------------------------------------------

n = min(
    len(noise),
    len(wifi)
)

noise = noise.iloc[:n].copy()
wifi = wifi.iloc[:n].copy()


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


# ---------------------------------------------------------
# Chronological 70/30 split
# ---------------------------------------------------------

split_noise = int(
    len(noise) * 0.70
)

split_wifi = int(
    len(wifi) * 0.70
)


noise_train = noise.iloc[:split_noise]
noise_test = noise.iloc[split_noise:]

wifi_train = wifi.iloc[:split_wifi]
wifi_test = wifi.iloc[split_wifi:]


train = pd.concat(
    [noise_train, wifi_train],
    ignore_index=True
)

test = pd.concat(
    [noise_test, wifi_test],
    ignore_index=True
)


X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]


print("=" * 75)
print("BLOCK / CHRONOLOGICAL REAL NOISE vs REAL WIFI TEST")
print("=" * 75)

print("\nTraining:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nWindow size: 4096 samples")
print("Feature count:", len(features))


# ---------------------------------------------------------
# Train
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
# Test
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
    f"\nChronological Test Accuracy: "
    f"{accuracy * 100:.2f}%"
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


print("\n" + "=" * 75)
print("BLOCK SPLIT TEST COMPLETE")
print("=" * 75)