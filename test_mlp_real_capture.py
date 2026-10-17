import numpy as np
import pandas as pd
import joblib

# ============================================================
# CONFIGURATION
# ============================================================

CAPTURE_FILE = "real_wifi_capture_02.dat"
SAMPLE_RATE = 20_000_000
CENTER_FREQUENCY = 2_400_000_000
WINDOW_SIZE = 4096

MODEL_FILE = "rfrecon_mlp_model.pkl"
FEATURE_FILE = "rfrecon_mlp_features.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

print("=" * 60)
print("RFRecon - MLP Real PlutoSDR Capture Test")
print("=" * 60)


# ============================================================
# LOAD REAL I/Q DATA
# ============================================================

iq = np.fromfile(CAPTURE_FILE, dtype=np.complex64)

print("\nCapture file:", CAPTURE_FILE)
print("Total samples:", len(iq))
print("Sample rate:", SAMPLE_RATE / 1e6, "MS/s")
print("Center frequency:", CENTER_FREQUENCY / 1e9, "GHz")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    power = np.abs(signal) ** 2
    amplitude = np.abs(signal)

    mean_power = np.mean(power)

    rms = np.sqrt(np.mean(power))

    peak_amplitude = np.max(amplitude)

    variance = np.var(power)

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(signal))

    magnitude = np.abs(spectrum)

    power_spectrum = magnitude ** 2

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            len(signal),
            d=1 / SAMPLE_RATE
        )
    )

    # Avoid division problems
    total_power = np.sum(power_spectrum) + 1e-12

    # Peak frequency
    peak_index = np.argmax(power_spectrum)

    peak_frequency = freqs[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    # Spectral centroid
    spectral_centroid = np.sum(
        np.abs(freqs) * power_spectrum
    ) / total_power

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            ((np.abs(freqs) - spectral_centroid) ** 2)
            * power_spectrum
        )
        / total_power
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # Occupied bandwidth
    threshold = 0.01 * np.max(power_spectrum)

    occupied = freqs[
        power_spectrum >= threshold
    ]

    if len(occupied) > 0:
        occupied_bandwidth = (
            np.max(occupied) - np.min(occupied)
        )
    else:
        occupied_bandwidth = 0

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(
            np.log(power_spectrum + 1e-12)
        )
    )

    arithmetic_mean = np.mean(power_spectrum) + 1e-12

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    # Peak-to-average ratio
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


# ============================================================
# PROCESS WINDOWS
# ============================================================

num_windows = len(iq) // WINDOW_SIZE

print("\nWindow size:", WINDOW_SIZE)
print("Number of windows:", num_windows)

results = []

print("\nRunning MLP classification...")

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window = iq[start:end]

    feature_vector = extract_features(window)

    feature_df = pd.DataFrame(
        [feature_vector],
        columns=features
    )

    prediction = model.predict(feature_df)[0]

    probabilities = model.predict_proba(feature_df)[0]

    confidence = np.max(probabilities)

    results.append(
        (
            prediction,
            confidence
        )
    )

# ============================================================
# RESULTS
# ============================================================

predictions = [x[0] for x in results]
confidences = [x[1] for x in results]

prediction_counts = pd.Series(
    predictions
).value_counts()

print("\n" + "=" * 60)
print("MLP REAL CAPTURE RESULTS")
print("=" * 60)

print("\nPrediction distribution:")
print(prediction_counts)

print(
    "\nAverage confidence: "
    f"{np.mean(confidences) * 100:.2f}%"
)

print(
    "Maximum confidence: "
    f"{np.max(confidences) * 100:.2f}%"
)

print(
    "Minimum confidence: "
    f"{np.min(confidences) * 100:.2f}%"
)


# ============================================================
# DOMINANT CLASS
# ============================================================

dominant_class = prediction_counts.index[0]

dominant_count = prediction_counts.iloc[0]

agreement = (
    dominant_count / num_windows
) * 100

print("\nDominant classification:", dominant_class)

print(
    f"Window agreement: {agreement:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = pd.DataFrame({
    "window": range(num_windows),
    "prediction": predictions,
    "confidence": confidences
})

output.to_csv(
    "mlp_real_capture_predictions.csv",
    index=False
)

print(
    "\nResults saved to:"
    " mlp_real_capture_predictions.csv"
)

print("\nMLP real capture test completed!")