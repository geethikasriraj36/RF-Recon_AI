import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD MLP MODEL
# ============================================================

MODEL_FILE = "rfrecon_mlp_model.pkl"
FEATURE_FILE = "rfrecon_mlp_features.pkl"
TEST_FILE = "real_wifi_test_dataset.csv"

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

print("=" * 60)
print("RFRecon - MLP Real PlutoSDR Test")
print("=" * 60)

# ============================================================
# 2. LOAD UNSEEN REAL WIFI DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

print("\nTest dataset shape:", df.shape)

print("\nActual class distribution:")
print(df["label"].value_counts())

# ============================================================
# 3. EXTRACT FEATURES
# ============================================================

X_test = df[features]
y_test = df["label"]

# ============================================================
# 4. PREDICT
# ============================================================

print("\nRunning MLP predictions...")

y_pred = model.predict(X_test)

# ============================================================
# 5. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MLP REAL WIFI TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ============================================================
# 6. CLASSIFICATION REPORT
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
# 7. PREDICTION DISTRIBUTION
# ============================================================

print("\nPrediction distribution:")

prediction_counts = pd.Series(y_pred).value_counts()

print(prediction_counts)

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

labels = [
    "bluetooth_like",
    "cellular_like",
    "noise",
    "wifi_like",
    "zigbee_like"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# 9. WIFI-SPECIFIC RESULT
# ============================================================

wifi_predictions = sum(y_pred == "wifi_like")

print("\nActual Wi-Fi samples:", len(y_test))
print("Predicted Wi-Fi samples:", wifi_predictions)

print(
    f"Wi-Fi agreement: "
    f"{wifi_predictions / len(y_test) * 100:.2f}%"
)

print("\nMLP real-data test completed!")