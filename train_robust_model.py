import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD ROBUST DATASET
# ============================================================

df = pd.read_csv("robust_training_dataset.csv")

print("=" * 65)
print("RFRecon AI - Robust Training")
print("=" * 65)

print("\nTotal examples:", len(df))


# ============================================================
# 2. SELECT FEATURES
# ============================================================

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
    "peak_to_average_ratio"
]


X = df[features]
y = df["label"]


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_test))


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining robust RFRecon AI model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 5. VALIDATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 6. RESULTS
# ============================================================

print("\n" + "=" * 65)
print("ROBUST MODEL RESULTS")
print("=" * 65)

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


class_names = [
    "Noise",
    "WiFi-like",
    "Bluetooth-like",
    "Zigbee-like",
    "Cellular-like"
]


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)