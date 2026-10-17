import numpy as np
import pandas as pd
import joblib

from scipy.signal import periodogram


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("rfrecon_rf_model.pkl")
feature_names = joblib.load("rfrecon_features.pkl")

SAMPLE_RATE = 2_000_000

CLASS_NAMES = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    magnitude = np.abs(signal)

    mean_power = np.mean(magnitude ** 2)

    rms = np.sqrt(mean_power)

    peak_amplitude = np.max(magnitude)

    variance = np.var(magnitude)

    frequencies, psd = periodogram(
        signal,
        fs=SAMPLE_RATE,
        return_onesided=False
    )

    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(np.abs(psd))

    total_power = np.sum(psd) + 1e-12

    peak_index = np.argmax(psd)
    peak_frequency = frequencies[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    spectral_centroid = (
        np.sum(np.abs(frequencies) * psd)
        / total_power
    )

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(frequencies)
                - spectral_centroid
            ) ** 2 * psd
        ) / total_power
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    cumulative_power = np.cumsum(psd)

    lower_index = np.searchsorted(
        cumulative_power,
        0.05 * total_power
    )

    upper_index = np.searchsorted(
        cumulative_power,
        0.95 * total_power
    )

    lower_index = min(
        lower_index,
        len(frequencies) - 1
    )

    upper_index = min(
        upper_index,
        len(frequencies) - 1
    )

    occupied_bandwidth = abs(
        frequencies[upper_index]
        - frequencies[lower_index]
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    geometric_mean = np.exp(
        np.mean(np.log(psd + 1e-12))
    )

    arithmetic_mean = np.mean(psd) + 1e-12

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    peak_to_average_ratio = (
        np.max(psd)
        / (np.mean(psd) + 1e-12)
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


# ============================================================
# TEST ONE SAMPLE FROM EACH CLASS
# ============================================================

test_samples = [
    ("Noise", "external_test_data/noise/noise_0000.npy"),
    ("WiFi-like", "external_test_data/wifi_like/wifi_like_0000.npy"),
    ("Bluetooth-like", "external_test_data/bluetooth_like/bluetooth_like_0000.npy"),
    ("Zigbee-like", "external_test_data/zigbee_like/zigbee_like_0000.npy"),
    ("Cellular-like", "external_test_data/cellular_like/cellular_like_0000.npy")
]


print("\n" + "=" * 65)
print("RFRecon AI - ALL CLASS PREDICTION TEST")
print("=" * 65)


for true_class, file_path in test_samples:

    signal = np.load(file_path)

    feature_vector = extract_features(signal)

    X = pd.DataFrame(
        [feature_vector],
        columns=feature_names
    )

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    confidence = probabilities[prediction] * 100

    predicted_class = CLASS_NAMES[prediction]

    print("\n---------------------------------------------")
    print("True Class      :", true_class)
    print("Predicted Class :", predicted_class)
    print(f"Confidence      : {confidence:.2f}%")

    print("\nProbabilities:")

    for class_id, probability in enumerate(probabilities):

        print(
            f"  {CLASS_NAMES[class_id]:<18}"
            f"{probability * 100:6.2f}%"
        )


print("\n" + "=" * 65)
print("TEST COMPLETE")
print("=" * 65)