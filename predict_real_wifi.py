import numpy as np
import joblib
import pandas as pd

# ==============================
# RFRecon REAL Wi-Fi PREDICTION
# ==============================

MODEL_FILE = "rfrecon_rf_model.pkl"
FEATURE_FILE = "rfrecon_features.pkl"
SAMPLE_FILE = r"real_wifi_data\real_wifi_0000.npy"

FS = 2_000_000


def extract_features(x):
    x = np.asarray(x, dtype=np.complex64)

    # Time-domain features
    power = np.abs(x) ** 2
    amplitude = np.abs(x)

    mean_power = np.mean(power)
    rms = np.sqrt(np.mean(power))
    peak_amplitude = np.max(amplitude)
    variance = np.var(power)

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(x))
    magnitude = np.abs(spectrum)

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(x), d=1 / FS)
    )

    # Peak frequency
    peak_index = np.argmax(magnitude)
    peak_frequency = abs(freqs[peak_index])
    peak_frequency_normalized = peak_frequency / (FS / 2)

    # Spectral centroid
    total_mag = np.sum(magnitude)

    if total_mag > 0:
        spectral_centroid = np.sum(
            np.abs(freqs) * magnitude
        ) / total_mag
    else:
        spectral_centroid = 0

    spectral_centroid_normalized = (
        spectral_centroid / (FS / 2)
    )

    # Spectral bandwidth
    if total_mag > 0:
        spectral_bandwidth = np.sqrt(
            np.sum(
                ((np.abs(freqs) - spectral_centroid) ** 2)
                * magnitude
            ) / total_mag
        )
    else:
        spectral_bandwidth = 0

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (FS / 2)
    )

    # Occupied bandwidth
    threshold = 0.10 * np.max(magnitude)
    active = np.abs(freqs[magnitude >= threshold])

    if len(active) > 0:
        occupied_bandwidth = (
            np.max(active) - np.min(active)
        )
    else:
        occupied_bandwidth = 0

    occupied_bandwidth_normalized = (
        occupied_bandwidth / (FS / 2)
    )

    # Spectral flatness
    magnitude_positive = magnitude + 1e-12

    geometric_mean = np.exp(
        np.mean(np.log(magnitude_positive))
    )

    arithmetic_mean = np.mean(magnitude_positive)

    spectral_flatness = (
        geometric_mean / arithmetic_mean
        if arithmetic_mean > 0 else 0
    )

    # Peak-to-average ratio
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


# Load model
model = joblib.load(MODEL_FILE)

# Load expected feature names
feature_names = joblib.load(FEATURE_FILE)

# Load real PlutoSDR Wi-Fi sample
x = np.load(SAMPLE_FILE)

# Extract RF features
features = extract_features(x)

# Arrange features exactly as expected by model
X = pd.DataFrame(
    [[features[name] for name in feature_names]],
    columns=feature_names
)

# Prediction
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

class_names = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}

predicted_name = class_names.get(
    int(prediction),
    str(prediction)
)

confidence = np.max(probabilities) * 100

print()
print("======================================")
print(" RFRecon REAL Wi-Fi AI CLASSIFICATION")
print("======================================")
print()
print("Sample              : real_wifi_0000.npy")
print("Samples              :", len(x))
print("Sample Rate          : 2 MHz")
print()
print("Predicted Class      :", predicted_name)
print("Confidence           : {:.2f}%".format(confidence))
print()
print("Class Probabilities:")
print("--------------------------------------")

for class_id, probability in zip(
    model.classes_, probabilities
):
    name = class_names.get(
        int(class_id),
        str(class_id)
    )

    print(
        "{:<20}: {:.2f}%".format(
            name,
            probability * 100
        )
    )

print()
print("======================================")