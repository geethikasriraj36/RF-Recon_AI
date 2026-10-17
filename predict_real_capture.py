import numpy as np
import pandas as pd
import joblib

MODEL_FILE = "rfrecon_hybrid_model.pkl"
FEATURE_FILE = "rfrecon_hybrid_features.pkl"
CAPTURE_FILE = "real_wifi_capture_03.dat"

SAMPLE_RATE = 20_000_000
WINDOW_SIZE = 4096

print("=" * 65)
print("RFRecon - REAL PLUTO SDR CAPTURE CLASSIFICATION")
print("=" * 65)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_FILE)
model.n_jobs = 1
feature_names = joblib.load(FEATURE_FILE)

print("\nModel loaded successfully.")
print("Capture:", CAPTURE_FILE)

# --------------------------------------------------
# LOAD PLUTO CAPTURE
# --------------------------------------------------

samples = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

print("Total samples:", len(samples))
print("Sample rate:", SAMPLE_RATE / 1e6, "MHz")

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(signal):

    power = np.abs(signal) ** 2
    magnitude = np.abs(signal)

    mean_power = np.mean(power)
    rms = np.sqrt(np.mean(power))
    peak_amplitude = np.max(magnitude)
    variance = np.var(magnitude)

    # FFT
    spectrum = np.fft.fftshift(
        np.fft.fft(signal)
    )

    magnitude_spectrum = np.abs(spectrum)

    # Frequency axis
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(signal),
            1 / SAMPLE_RATE
        )
    )

    power_spectrum = magnitude_spectrum ** 2

    # Avoid division problems
    total_power = np.sum(power_spectrum) + 1e-12

    # Peak frequency
    peak_index = np.argmax(magnitude_spectrum)

    peak_frequency = abs(
        frequencies[peak_index]
    )

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    # Spectral centroid
    spectral_centroid = (
        np.sum(
            np.abs(frequencies) *
            power_spectrum
        ) / total_power
    )

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(frequencies)
                - spectral_centroid
            ) ** 2
            * power_spectrum
        )
        / total_power
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # Occupied bandwidth: 90% power
    sorted_indices = np.argsort(
        np.abs(frequencies)
    )

    cumulative_power = np.cumsum(
        power_spectrum[sorted_indices]
    )

    lower_index = np.searchsorted(
        cumulative_power,
        0.05 * total_power
    )

    upper_index = np.searchsorted(
        cumulative_power,
        0.95 * total_power
    )

    lower_freq = np.abs(
        frequencies[
            sorted_indices[
                min(lower_index, len(frequencies)-1)
            ]
        ]
    )

    upper_freq = np.abs(
        frequencies[
            sorted_indices[
                min(upper_index, len(frequencies)-1)
            ]
        ]
    )

    occupied_bandwidth = abs(
        upper_freq - lower_freq
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(
            np.log(power_spectrum + 1e-12)
        )
    )

    arithmetic_mean = np.mean(
        power_spectrum
    ) + 1e-12

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    # Peak-to-average ratio
    peak_to_average_ratio = (
        np.max(power)
        / (np.mean(power) + 1e-12)
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


# --------------------------------------------------
# PROCESS CAPTURE
# --------------------------------------------------

num_windows = len(samples) // WINDOW_SIZE

print("Windows available:", num_windows)
print("\nExtracting features from capture...")

all_features = []

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window = samples[start:end]

    features = extract_features(window)

    all_features.append(features)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{num_windows} windows")

# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

print("\nRunning AI classification in one batch...")

X = pd.DataFrame(
    all_features,
    columns=feature_names
)

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidences = np.max(
    probabilities,
    axis=1
) * 100

results_df = pd.DataFrame({
    "window": np.arange(num_windows),
    "prediction": predictions,
    "confidence": confidences
})

print("AI classification complete.")
# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("RFRecon REAL CAPTURE RESULTS")
print("=" * 65)

print("\nPrediction distribution:")

print(
    results_df["prediction"]
    .value_counts()
)

print(
    "\nAverage confidence: "
    f"{results_df['confidence'].mean():.2f}%"
)

print(
    "Maximum confidence: "
    f"{results_df['confidence'].max():.2f}%"
)

print(
    "Minimum confidence: "
    f"{results_df['confidence'].min():.2f}%"
)

# --------------------------------------------------
# DOMINANT CLASS
# --------------------------------------------------

dominant_class = (
    results_df["prediction"]
    .value_counts()
    .idxmax()
)

dominant_percentage = (
    results_df["prediction"]
    .value_counts(normalize=True)
    .iloc[0]
    * 100
)

print("\n" + "=" * 65)
print("FINAL RFRecon DECISION")
print("=" * 65)

print(
    f"\nDetected RF class : {dominant_class}"
)

print(
    f"Window agreement  : {dominant_percentage:.2f}%"
)

print(
    f"Average confidence: "
    f"{results_df['confidence'].mean():.2f}%"
)

print("=" * 65)