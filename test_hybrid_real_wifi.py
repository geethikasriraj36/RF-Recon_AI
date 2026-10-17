import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_FILE = "rfrecon_hybrid_model.pkl"
DATA_FILE = "real_wifi_test_dataset.csv"

print("=" * 60)
print("RFRecon - HYBRID MODEL REAL WI-FI TEST")
print("=" * 60)

# Load model
model = joblib.load(MODEL_FILE)

# Load unseen real Wi-Fi test data
df = pd.read_csv(DATA_FILE)

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

print("\nUnseen real Wi-Fi samples:", len(df))

# Predict
predictions = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, predictions)

print("\n" + "=" * 60)
print("REAL WI-FI HYBRID TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nPredicted classes:")
print(pd.Series(predictions).value_counts())

print("=" * 60)