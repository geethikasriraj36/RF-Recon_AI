import pandas as pd
import numpy as np

# ==========================================
# RFRecon - Hybrid Synthetic + Real Dataset
# ==========================================

SYNTHETIC_FILE = "robust_training_dataset.csv"
REAL_FILE = "real_wifi_dataset.csv"

HYBRID_TRAIN_FILE = "hybrid_training_dataset.csv"
REAL_TEST_FILE = "real_wifi_test_dataset.csv"

# ------------------------------------------
# Load datasets
# ------------------------------------------

print("=" * 60)
print("RFRecon - HYBRID DATASET CREATION")
print("=" * 60)

synthetic = pd.read_csv(SYNTHETIC_FILE)
real = pd.read_csv(REAL_FILE)

print("\nSynthetic samples:", len(synthetic))
print("Real Wi-Fi samples:", len(real))

# ------------------------------------------
# Feature columns
# ------------------------------------------

feature_columns = [
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

# ------------------------------------------
# Convert synthetic numeric labels
# ------------------------------------------

class_names = {
    0: "noise",
    1: "wifi_like",
    2: "bluetooth_like",
    3: "zigbee_like",
    4: "cellular_like"
}

synthetic["label"] = synthetic["label"].map(class_names)

# ------------------------------------------
# Split real Wi-Fi data
# 400 -> training
# 100 -> independent testing
# ------------------------------------------

real = real.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

real_train = real.iloc[:400].copy()
real_test = real.iloc[400:500].copy()

# ------------------------------------------
# Remove synthetic Wi-Fi
#
# We want the model to learn Wi-Fi from
# actual PlutoSDR measurements.
# ------------------------------------------

synthetic_non_wifi = synthetic[
    synthetic["label"] != "wifi_like"
].copy()

# ------------------------------------------
# Combine:
#
# 500 Noise
# 400 Real Wi-Fi
# 500 Bluetooth-like
# 500 Zigbee-like
# 500 Cellular-like
#
# Total = 2400 training samples
# ------------------------------------------

hybrid_train = pd.concat(
    [
        synthetic_non_wifi,
        real_train
    ],
    ignore_index=True
)

# Shuffle training data
hybrid_train = hybrid_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ------------------------------------------
# Save datasets
# ------------------------------------------

hybrid_train.to_csv(
    HYBRID_TRAIN_FILE,
    index=False
)

real_test.to_csv(
    REAL_TEST_FILE,
    index=False
)

# ------------------------------------------
# Display results
# ------------------------------------------

print("\n" + "=" * 60)
print("HYBRID DATASET CREATED")
print("=" * 60)

print("\nTraining samples:", len(hybrid_train))
print("Real Wi-Fi training:", len(real_train))
print("Real Wi-Fi test:", len(real_test))

print("\nTraining class distribution:")
print(hybrid_train["label"].value_counts().sort_index())

print("\nIndependent real Wi-Fi test:")
print(real_test["label"].value_counts())

print("\nSaved:")
print("-", HYBRID_TRAIN_FILE)
print("-", REAL_TEST_FILE)

print("=" * 60)