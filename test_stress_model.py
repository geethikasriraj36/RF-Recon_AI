import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(
    "realistic_rf_dataset.csv"
)

stress_df = pd.read_csv(
    "stress_test_dataset.csv"
)

print("=" * 65)
print("RFRecon - Low-SNR Robustness Test")
print("=" * 65)

print("\nTraining dataset:", len(train_df))
print("Stress-test dataset:", len(stress_df))


# ============================================================
# 2. FEATURES
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


X_train = train_df[features]
y_train = train_df["label"]

X_test = stress_df[features]
y_test = stress_df["label"]


# ============================================================
# 3. TRAIN ONLY ON NORMAL TRAINING DATA
# ============================================================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 4. TEST ON LOW-SNR DATA
# ============================================================

print("Testing on low-SNR stress-test data...")

y_pred = model.predict(
    X_test
)


# ============================================================
# 5. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 65)
print("LOW-SNR STRESS-TEST RESULTS")
print("=" * 65)

print(
    f"\nStress-Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 8. PLOT CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "RFRecon AI - Low-SNR Stress Test"
)

plt.tight_layout()

plt.savefig(
    "stress_test_confusion_matrix.png",
    dpi=300
)

plt.show()