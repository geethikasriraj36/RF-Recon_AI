import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# RFRecon - Hybrid Model Training
# ==========================================

DATA_FILE = "hybrid_training_dataset.csv"

FEATURES = [
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

print("=" * 60)
print("RFRecon - HYBRID MODEL TRAINING")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\nTotal samples:", len(df))

X = df[FEATURES]
y = df["label"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# Train Random Forest
print("\nTraining hybrid Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Validation prediction
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\n" + "=" * 60)
print("HYBRID MODEL VALIDATION RESULTS")
print("=" * 60)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Save model
joblib.dump(
    model,
    "rfrecon_hybrid_model.pkl"
)

joblib.dump(
    FEATURES,
    "rfrecon_hybrid_features.pkl"
)

print("\n" + "=" * 60)
print("HYBRID MODEL SAVED")
print("=" * 60)

print("\nModel:")
print("rfrecon_hybrid_model.pkl")

print("\nFeatures:")
print("rfrecon_hybrid_features.pkl")