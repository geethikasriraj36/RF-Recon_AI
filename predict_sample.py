import numpy as np
import joblib

from scipy.signal import periodogram


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load(
    "rfrecon_rf_model.pkl"
)

features = joblib.load(
    "rfrecon_features.pkl"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SAMPLE_RATE = 2_000_000

CLASS_NAMES = {
    0: "Noise",
    1: "WiFi-like",
    2: "Bluetooth-like",
    3: "Zigbee-like",
    4: "Cellular-like"
}


# ============================================================
# 3. FEATURE EXTRACTION
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

    psd = np.fft.fftshift(psd)

    psd = np.abs(psd)

    total_power = np.sum(psd) + 1e-12

    peak_index = np.argmax(psd)

    peak_frequency = frequencies[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    spectral_centroid = (
        np.sum(
            np.abs(frequencies) * psd
        ) / total_power
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
        -
        frequencies[lower_index]
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(psd + 1e-12)
        )
    )

    arithmetic_mean = (
        np.mean(psd) + 1e-12
    )

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    peak_power = np.max(psd)

    average_spectral_power = (
        np.mean(psd) + 1e-12
    )

    peak_to_average_ratio = (
        peak_power / average_spectral_power
    )

    return np.array([
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
    ])


# ============================================================
# 4. SELECT ONE SAMPLE
# ============================================================

sample_path = (
    "external_test_data/wifi_like/wifi_like_0000.npy"
)

signal = np.load(sample_path)


# ============================================================
# 5. EXTRACT FEATURES
# ============================================================

feature_vector = extract_features(signal)

X = feature_vector.reshape(1, -1)


# ============================================================
# 6. PREDICT
# ============================================================

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

confidence = probabilities[prediction] * 100


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 55)
print("RFRecon AI - Single Sample Prediction")
print("=" * 55)

print("\nInput:")
print(sample_path)

print("\nPredicted Class:")
print(CLASS_NAMES[prediction])

print(
    f"\nConfidence: {confidence:.2f}%"
)

print("\nClass probabilities:")

for class_id, probability in enumerate(probabilities):

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"{probability * 100:.2f}%"
    )

print("\n" + "=" * 55)