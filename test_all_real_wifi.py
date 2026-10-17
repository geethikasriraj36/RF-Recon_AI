import numpy as np
import joblib
import pandas as pd
from collections import Counter

MODEL_FILE = "rfrecon_rf_model.pkl"
FEATURE_FILE = "rfrecon_features.pkl"

FS = 2_000_000

class_names = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}


def extract_features(x):
    power = np.abs(x) ** 2
    amplitude = np.abs(x)

    mean_power = np.mean(power)
    rms = np.sqrt(np.mean(power))
    peak_amplitude = np.max(amplitude)
    variance = np.var(power)

    spectrum = np.fft.fftshift(np.fft.fft(x))
    magnitude = np.abs(spectrum)

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / FS)
    )

    peak_index = np.argmax(magnitude)
    peak_frequency = abs(freqs[peak_index])

    peak_frequency_normalized = peak_frequency / (FS / 2)

    total_mag = np.sum(magnitude)

    spectral_centroid = (
        np.sum(np.abs(freqs) * magnitude) / total_mag
        if total_mag > 0 else 0
    )

    spectral_centroid_normalized = (
        spectral_centroid / (FS / 2)
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((np.abs(freqs) - spectral_centroid) ** 2)
            * magnitude
        ) / total_mag
    ) if total_mag > 0 else 0

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (FS / 2)
    )

    threshold = 0.10 * np.max(magnitude)
    active = np.abs(freqs[magnitude >= threshold])

    occupied_bandwidth = (
        np.max(active) - np.min(active)
        if len(active) > 0 else 0
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / (FS / 2)
    )

    magnitude_positive = magnitude + 1e-12

    spectral_flatness = (
        np.exp(np.mean(np.log(magnitude_positive)))
        / np.mean(magnitude_positive)
    )

    peak_to_average_ratio = (
        peak_amplitude / rms
        if rms > 0 else 0
    )

    return {
        "mean_power": mean_power,
        "rms": rms,
        "peak_amplitude": peak_amplitude,
        "variance": variance,
        "peak_frequency_normalized": peak_frequency_normalized,
        "spectral_centroid_normalized": spectral_centroid_normalized,
        "spectral_bandwidth_normalized": spectral_bandwidth_normalized,
        "occupied_bandwidth_normalized": occupied_bandwidth_normalized,
        "spectral_flatness": spectral_flatness,
        "peak_to_average_ratio": peak_to_average_ratio
    }


model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

predictions = []
confidences = []

print()
print("==========================================")
print(" RFRecon - ALL REAL Wi-Fi SAMPLE TEST")
print("==========================================")
print()
print("Testing 500 real PlutoSDR samples...")
print()

for i in range(500):

    filename = f"real_wifi_data\\real_wifi_{i:04d}.npy"

    x = np.load(filename)

    features = extract_features(x)

    X = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(X)[0]
    probability = np.max(model.predict_proba(X)[0])

    predictions.append(int(prediction))
    confidences.append(probability)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/500")


counts = Counter(predictions)

print()
print("==========================================")
print(" REAL WI-FI DATASET RESULTS")
print("==========================================")

for class_id in range(5):

    count = counts.get(class_id, 0)
    percentage = count / 500 * 100

    print(
        f"{class_names[class_id]:<20}: "
        f"{count} ({percentage:.2f}%)"
    )

print()
print(
    "Average confidence: "
    f"{np.mean(confidences) * 100:.2f}%"
)

print()
print("==========================================")