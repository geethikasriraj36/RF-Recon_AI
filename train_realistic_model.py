import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("realistic_rf_dataset.csv")

print("=" * 60)
print("RFRecon - 5 Class AI Classification")
print("=" * 60)

print("\nDataset loaded!")
print("Total examples:", len(df))


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
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining RFRecon 5-class model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 7. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 8. CLASSIFICATION REPORT
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
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 10. PLOT CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "RFRecon AI - 5 Class Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "realistic_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    feature_importance.to_string(
        index=False
    )
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)