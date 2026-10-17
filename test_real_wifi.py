import os
import numpy as np
import pandas as pd
import joblib

# ==============================
# RFRecon Real Wi-Fi Test
# ==============================

MODEL_FILE = "rfrecon_rf_model.pkl"
FEATURE_FILE = "rfrecon_features.pkl"
DATA_FOLDER = "real_wifi_data"

print("=" * 50)
print("RFRecon - REAL PLUTO SDR WI-FI TEST")
print("=" * 50)

# Load model and feature list
model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

print("\nModel loaded successfully.")

# Feature extraction
def extract_features(signal, sample_rate=20_000_000):

    power = np.abs(signal) ** 2

    mean_power = np.mean(power)
    rms = np.sqrt(mean_power)
    peak_amplitude = np.max(np.abs(signal))
    variance = np.var(np.abs(signal))

    spectrum = np.fft.fftshift(np.fft.fft(signal))
    magnitude = np.abs(spectrum)

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(signal), 1 / sample_rate)
    )

    peak_index = np.argmax(magnitude)

    peak_frequency_normalized = (
        abs(freqs[peak_index]) / (sample_rate / 2)
    )

    weights = magnitude + 1e-12

    spectral_centroid = (
        np.sum(np.abs(freqs) * weights) /
        np.sum(weights)
    )

    spectral_centroid_normalized = (
        spectral_centroid / (sample_rate / 2)
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((np.abs(freqs) - spectral_centroid) ** 2) * weights
        ) / np.sum(weights)
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (sample_rate / 2)
    )

    threshold = 0.10 * np.max(magnitude)

    occupied = np.abs(freqs)[magnitude > threshold]

    if len(occupied) > 0:
        occupied_bandwidth_normalized = (
            (np.max(occupied) - np.min(occupied))
            / (sample_rate / 2)
        )
    else:
        occupied_bandwidth_normalized = 0

    normalized_mag = magnitude / (np.max(magnitude) + 1e-12)

    spectral_flatness = (
        np.exp(np.mean(np.log(normalized_mag + 1e-12))) /
        (np.mean(normalized_mag) + 1e-12)
    )

    peak_to_average_ratio = (
        peak_amplitude / (rms + 1e-12)
    )

    return [
        mean_power,
        rms,
        peak_amplitude,
        variance,
        peak_frequency_normalized,
        spectral_centroid_normalized,
        spectral_bandwidth_normalized,
        occupied_bandwidth_normalized,
        spectral_flatness,
        peak_to_average_ratio
    ]


# ==============================
# Test all real samples
# ==============================

files = sorted([
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".npy")
])

print(f"Real samples found: {len(files)}")

predictions = []
confidences = []

for i, filename in enumerate(files):

    path = os.path.join(DATA_FOLDER, filename)

    signal = np.load(path)

    features = extract_features(signal)

    X = pd.DataFrame(
        [features],
        columns=feature_names
    )

    probabilities = model.predict_proba(X)[0]

    prediction = model.predict(X)[0]

    confidence = np.max(probabilities)

    predictions.append(prediction)
    confidences.append(confidence)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(files)}")


# ==============================
# Results
# ==============================

print("\n" + "=" * 50)
print("REAL PLUTO SDR WI-FI RESULTS")
print("=" * 50)

unique, counts = np.unique(
    predictions,
    return_counts=True
)

class_names = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}

for label, count in zip(unique, counts):

    percentage = (
        count / len(predictions)
    ) * 100

    name = class_names.get(int(label), f"Class {label}")

    print(
        f"{name:20s}: "
        f"{count:3d} ({percentage:.2f}%)"
    )

print(
    f"\nAverage confidence: "
    f"{np.mean(confidences) * 100:.2f}%"
)

print("=" * 50)