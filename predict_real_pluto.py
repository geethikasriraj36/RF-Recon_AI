import numpy as np
import pandas as pd
import joblib

# Load trained RFRecon model
model = joblib.load("rfrecon_rf_model.pkl")

# Load feature names used during training
feature_names = joblib.load("rfrecon_features.pkl")

# Load real Pluto sample
x = np.load(r"real_pluto_data\real_pluto_0000.npy")

fs = 2_000_000

# Magnitude
magnitude = np.abs(x)

mean_power = np.mean(magnitude ** 2)
rms = np.sqrt(np.mean(magnitude ** 2))
peak_amplitude = np.max(magnitude)
variance = np.var(magnitude)

# FFT
spectrum = np.fft.fftshift(np.fft.fft(x))
power_spectrum = np.abs(spectrum) ** 2

freqs = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / fs)
)

total_power = np.sum(power_spectrum)

normalized_spectrum = (
    power_spectrum / total_power
    if total_power > 0
    else power_spectrum
)

# Peak frequency
peak_frequency = freqs[np.argmax(power_spectrum)]
peak_frequency_normalized = abs(peak_frequency) / (fs / 2)

# Spectral centroid
spectral_centroid = np.sum(
    np.abs(freqs) * normalized_spectrum
)

spectral_centroid_normalized = (
    spectral_centroid / (fs / 2)
)

# Spectral bandwidth
spectral_bandwidth = np.sqrt(
    np.sum(
        ((np.abs(freqs) - spectral_centroid) ** 2)
        * normalized_spectrum
    )
)

spectral_bandwidth_normalized = (
    spectral_bandwidth / (fs / 2)
)

# Occupied bandwidth
sorted_indices = np.argsort(np.abs(freqs))
sorted_power = power_spectrum[sorted_indices]
cumulative_power = np.cumsum(sorted_power)

low_index = np.searchsorted(
    cumulative_power,
    0.05 * total_power
)

high_index = np.searchsorted(
    cumulative_power,
    0.95 * total_power
)

low_freq = np.abs(freqs)[
    sorted_indices[low_index]
]

high_freq = np.abs(freqs)[
    sorted_indices[
        min(high_index, len(freqs) - 1)
    ]
]

occupied_bandwidth = high_freq - low_freq

occupied_bandwidth_normalized = (
    occupied_bandwidth / (fs / 2)
)

# Spectral flatness
epsilon = 1e-12

spectral_flatness = (
    np.exp(
        np.mean(
            np.log(power_spectrum + epsilon)
        )
    )
    /
    np.mean(power_spectrum + epsilon)
)

# Peak-to-average ratio
peak_to_average_ratio = (
    peak_amplitude / (rms + epsilon)
)

# Create feature dictionary
features = {
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

# Arrange features in exactly the training order
X = pd.DataFrame(
    [[features[name] for name in feature_names]],
    columns=feature_names
)

# Prediction
prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

confidence = np.max(probabilities) * 100

print("\n===== RFRecon REAL PLUTO PREDICTION =====")
print("Predicted Class :", prediction)
print(f"Confidence      : {confidence:.2f}%")

class_names = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}

print("\nClass Probabilities:")

for class_id, probability in zip(
    model.classes_,
    probabilities
):
    print(
        f"{class_names[int(class_id)]:20s}: "
        f"{probability * 100:.2f}%"
    )