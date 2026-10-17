import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD ROBUST DATASET
# ============================================================

df = pd.read_csv("robust_training_dataset.csv")

print("=" * 65)
print("RFRecon AI - Saving Trained Model")
print("=" * 65)

print("\nTraining examples:", len(df))


# ============================================================
# FEATURES
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
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final RFRecon AI model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "rfrecon_rf_model.pkl"
)


# ============================================================
# SAVE FEATURE ORDER
# ============================================================

joblib.dump(
    features,
    "rfrecon_features.pkl"
)


print("\n" + "=" * 65)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 65)

print("\nModel file:")
print("rfrecon_rf_model.pkl")

print("\nFeature file:")
print("rfrecon_features.pkl")