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

train_df = pd.read_csv("robust_training_dataset.csv")
stress_df = pd.read_csv("stress_test_dataset.csv")

print("=" * 65)
print("RFRecon AI - Robust Model vs Low-SNR Stress Test")
print("=" * 65)

print("\nRobust training samples:", len(train_df))
print("Stress-test samples:", len(stress_df))


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

X_stress = stress_df[features]
y_stress = stress_df["label"]


# ============================================================
# 3. TRAIN ROBUST MODEL
# ============================================================

print("\nTraining robust model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 4. TEST ON LOW-SNR DATA
# ============================================================

print("Testing robust model on 0-6 dB SNR data...")

y_pred = model.predict(X_stress)


# ============================================================
# 5. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_stress,
    y_pred
)

print("\n" + "=" * 65)
print("ROBUST MODEL - STRESS TEST RESULTS")
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
        y_stress,
        y_pred,
        target_names=class_names
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_stress,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 8. DISPLAY CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "RFRecon AI - Robust Model Low-SNR Stress Test"
)

plt.tight_layout()

plt.savefig(
    "robust_stress_confusion_matrix.png",
    dpi=300
)

plt.show()