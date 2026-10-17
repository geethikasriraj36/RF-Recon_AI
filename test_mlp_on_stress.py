import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "rfrecon_mlp_model.pkl"
FEATURE_FILE = "rfrecon_mlp_features.pkl"
STRESS_FILE = "stress_test_dataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

print("=" * 60)
print("RFRecon - MLP Low-SNR Stress Test")
print("=" * 60)


# ============================================================
# LOAD STRESS DATA
# ============================================================

df = pd.read_csv(STRESS_FILE)

print("\nStress-test dataset shape:", df.shape)

print("\nOriginal class distribution:")
print(df["label"].value_counts())


# ============================================================
# CONVERT NUMERIC LABELS TO CLASS NAMES
# ============================================================

label_mapping = {
    0: "noise",
    1: "wifi_like",
    2: "bluetooth_like",
    3: "zigbee_like",
    4: "cellular_like"
}

df["label"] = df["label"].map(label_mapping)

print("\nConverted class distribution:")
print(df["label"].value_counts())


# ============================================================
# CHECK FOR INVALID LABELS
# ============================================================

if df["label"].isna().any():
    print("\nERROR: Unknown label found in stress dataset.")
    print(df[df["label"].isna()])
    raise ValueError("Invalid stress-test label detected.")


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_test = df[features]
y_test = df["label"]


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning MLP predictions...")

y_pred = model.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MLP LOW-SNR STRESS TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "noise",
    "wifi_like",
    "bluetooth_like",
    "zigbee_like",
    "cellular_like"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

results.to_csv(
    "mlp_stress_test_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to:"
    " mlp_stress_test_predictions.csv"
)

print("\nMLP low-SNR stress test completed!")