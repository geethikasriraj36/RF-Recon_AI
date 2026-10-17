import pandas as pd
import numpy as np
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD HYBRID DATASET
# ============================================================

DATASET = "hybrid_training_dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 60)
print("RFRecon - MLP Neural Network Training")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


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

print("\nNumber of features:", len(features))
print("Number of samples:", len(X))


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# 4. BUILD MLP NEURAL NETWORK
# ============================================================

mlp_model = Pipeline([
    
    # Neural networks work better with scaled features
    ("scaler", StandardScaler()),

    # MLP Neural Network
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    ))
])


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining MLP Neural Network...")

start_time = time.time()

mlp_model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\nTraining completed!")
print(f"Training time: {training_time:.2f} seconds")


# ============================================================
# 6. VALIDATION PREDICTION
# ============================================================

y_pred = mlp_model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\n" + "=" * 60)
print("MLP VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        digits=4
    )
)


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
    y_val,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    mlp_model,
    "rfrecon_mlp_model.pkl"
)

joblib.dump(
    features,
    "rfrecon_mlp_features.pkl"
)

print("\nModel saved as:")
print("rfrecon_mlp_model.pkl")

print("\nFeature list saved as:")
print("rfrecon_mlp_features.pkl")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL MLP SUMMARY")
print("=" * 60)

print(f"Dataset samples       : {len(df)}")
print(f"Training samples      : {len(X_train)}")
print(f"Validation samples    : {len(X_val)}")
print(f"Number of features    : {len(features)}")
print(f"Number of classes     : {len(y.unique())}")
print(f"Validation accuracy   : {accuracy * 100:.2f}%")
print(f"Training time         : {training_time:.2f} seconds")

print("\nClasses:")
for c in sorted(y.unique()):
    print(" -", c)

print("\nMLP training finished successfully!")