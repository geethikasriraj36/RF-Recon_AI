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
# 1. LOAD TRAINING DATA
# ============================================================

train_df = pd.read_csv(
    "realistic_rf_dataset.csv"
)

test_df = pd.read_csv(
    "external_test_dataset.csv"
)

print("=" * 65)
print("RFRecon - Independent Generalization Test")
print("=" * 65)

print("\nTraining dataset:", len(train_df))
print("Independent test dataset:", len(test_df))


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

X_test = test_df[features]
y_test = test_df["label"]


# ============================================================
# 3. TRAIN MODEL ONLY ON ORIGINAL DATA
# ============================================================

print("\nTraining model on original dataset...")

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
# 4. PREDICT COMPLETELY NEW DATA
# ============================================================

print("Testing on independent dataset...")

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
print("INDEPENDENT TEST RESULTS")
print("=" * 65)

print(
    f"\nIndependent Test Accuracy: "
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
# 8. SAVE CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "RFRecon AI - Independent Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "independent_confusion_matrix.png",
    dpi=300
)

plt.show()