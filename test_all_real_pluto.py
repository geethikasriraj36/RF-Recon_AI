import numpy as np
import pandas as pd
import joblib
from collections import Counter

# Load model
model = joblib.load("rfrecon_rf_model.pkl")
feature_names = joblib.load("rfrecon_features.pkl")

class_names = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}

fs = 2_000_000


def extract_features(x):

    magnitude = np.abs(x)

    mean_power = np.mean(magnitude ** 2)
    rms = np.sqrt(np.mean(magnitude ** 2))
    peak_amplitude = np.max(magnitude)
    variance = np.var(magnitude)

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

    peak_frequency = freqs[np.argmax(power_spectrum)]

    peak_frequency_normalized = (
        abs(peak_frequency) / (fs / 2)
    )

    spectral_centroid = np.sum(
        np.abs(freqs) * normalized_spectrum
    )

    spectral_centroid_normalized = (
        spectral_centroid / (fs / 2)
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((np.abs(freqs) - spectral_centroid) ** 2)
            * normalized_spectrum
        )
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (fs / 2)
    )

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

    peak_to_average_ratio = (
        peak_amplitude / (rms + epsilon)
    )

    return {
        "mean_power": mean_power,
        "rms": rms,
        "peak_amplitude": peak_amplitude,
        "variance": variance,
        "peak_frequency_normalized":
            peak_frequency_normalized,
        "spectral_centroid_normalized":
            spectral_centroid_normalized,
        "spectral_bandwidth_normalized":
            spectral_bandwidth_normalized,
        "occupied_bandwidth_normalized":
            occupied_bandwidth_normalized,
        "spectral_flatness":
            spectral_flatness,
        "peak_to_average_ratio":
            peak_to_average_ratio
    }


predictions = []
confidences = []

print("Testing 500 real PlutoSDR samples...\n")

for i in range(500):

    filename = (
        f"real_pluto_data/real_pluto_{i:04d}.npy"
    )

    x = np.load(filename)

    features = extract_features(x)

    X = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    confidence = np.max(probabilities)

    predictions.append(int(prediction))
    confidences.append(confidence)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/500")


# Count predictions
counts = Counter(predictions)

print("\n===== REAL PLUTO DATASET RESULTS =====")

for class_id in sorted(counts):
    percentage = (
        counts[class_id] / len(predictions) * 100
    )

    print(
        f"{class_names[class_id]:20s}: "
        f"{counts[class_id]:3d} "
        f"({percentage:.2f}%)"
    )


average_confidence = (
    np.mean(confidences) * 100
)

print(
    f"\nAverage confidence: "
    f"{average_confidence:.2f}%"
)