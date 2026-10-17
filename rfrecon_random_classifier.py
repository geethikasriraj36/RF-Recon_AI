import os
import numpy as np
import joblib

# =========================================================
# RFRecon - RANDOM RF CAPTURE CLASSIFIER
# =========================================================

RF_MODEL_FILE = "rfrecon_hybrid_model.pkl"
MLP_MODEL_FILE = "rfrecon_mlp_model.pkl"

RF_FEATURE_FILE = "rfrecon_hybrid_features.pkl"
MLP_FEATURE_FILE = "rfrecon_mlp_features.pkl"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096


# =========================================================
# LOAD MODELS
# =========================================================

print("=" * 70)
print("RFRecon - RANDOM RF SIGNAL CLASSIFIER")
print("=" * 70)

rf_model = joblib.load(RF_MODEL_FILE)
mlp_model = joblib.load(MLP_MODEL_FILE)

rf_features = joblib.load(RF_FEATURE_FILE)
mlp_features = joblib.load(MLP_FEATURE_FILE)

print("Models loaded successfully.")


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(samples):

    x = samples.astype(np.complex64)

    amplitude = np.abs(x)

    power = amplitude ** 2

    mean_power = np.mean(power)

    rms = np.sqrt(mean_power)

    peak_amplitude = np.max(amplitude)

    variance = np.var(amplitude)

    # -----------------------------------------------------
    # FFT
    # -----------------------------------------------------

    window = np.hanning(len(x))

    spectrum = np.fft.fftshift(
        np.fft.fft(x * window)
    )

    power_spectrum = np.abs(spectrum) ** 2

    power_spectrum += 1e-12

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(x),
            d=1 / SAMPLE_RATE
        )
    )

    # -----------------------------------------------------
    # Peak frequency
    # -----------------------------------------------------

    peak_index = np.argmax(power_spectrum)

    peak_frequency = frequencies[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    # -----------------------------------------------------
    # Spectral centroid
    # -----------------------------------------------------

    spectral_centroid = (
        np.sum(
            np.abs(frequencies)
            * power_spectrum
        )
        /
        np.sum(power_spectrum)
    )

    spectral_centroid_normalized = (
        spectral_centroid / (SAMPLE_RATE / 2)
    )

    # -----------------------------------------------------
    # Spectral bandwidth
    # -----------------------------------------------------

    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(frequencies)
                - spectral_centroid
            ) ** 2
            * power_spectrum
        )
        /
        np.sum(power_spectrum)
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth / (SAMPLE_RATE / 2)
    )

    # -----------------------------------------------------
    # Occupied bandwidth - 95% power
    # -----------------------------------------------------

    sorted_indices = np.argsort(power_spectrum)

    sorted_power = power_spectrum[sorted_indices]

    cumulative_power = np.cumsum(sorted_power)

    total_power = cumulative_power[-1]

    lower_index = np.searchsorted(
        cumulative_power,
        0.025 * total_power
    )

    upper_index = np.searchsorted(
        cumulative_power,
        0.975 * total_power
    )

    lower_frequency = frequencies[
        sorted_indices[lower_index]
    ]

    upper_frequency = frequencies[
        sorted_indices[upper_index]
    ]

    occupied_bandwidth = (
        upper_frequency - lower_frequency
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    # -----------------------------------------------------
    # Spectral flatness
    # -----------------------------------------------------

    geometric_mean = np.exp(
        np.mean(np.log(power_spectrum))
    )

    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness = (
        geometric_mean / arithmetic_mean
    )

    # -----------------------------------------------------
    # Peak-to-average ratio
    # -----------------------------------------------------

    peak_to_average_ratio = (
        np.max(power) /
        (np.mean(power) + 1e-12)
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


# =========================================================
# ASK FOR CAPTURE
# =========================================================

capture_file = input(
    "\nEnter .dat capture path: "
).strip().strip('"')


if not os.path.exists(capture_file):

    print("\nERROR: File not found.")

    raise SystemExit


# =========================================================
# LOAD CAPTURE
# =========================================================

data = np.memmap(
    capture_file,
    dtype=np.complex64,
    mode="r"
)

total_samples = len(data)

total_windows = (
    total_samples // WINDOW_SIZE
)

print()
print("CAPTURE INFORMATION")
print("-" * 70)

print(f"File           : {capture_file}")
print(f"Samples        : {total_samples:,}")
print(f"Sample rate    : {SAMPLE_RATE/1e6:.1f} MS/s")
print(f"Window size    : {WINDOW_SIZE}")
print(f"Windows        : {total_windows:,}")


# =========================================================
# PROCESS WINDOWS
# =========================================================

features = []

for i in range(total_windows):

    start = i * WINDOW_SIZE

    end = start + WINDOW_SIZE

    samples = data[start:end]

    feature_dict = extract_features(samples)

    features.append(feature_dict)


# =========================================================
# CONVERT TO FEATURE MATRIX
# =========================================================

feature_names = [
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


X = np.array([
    [
        row[name]
        for name in feature_names
    ]
    for row in features
])


# =========================================================
# CHECK FEATURE COMPATIBILITY
# =========================================================

if hasattr(rf_model, "n_features_in_"):

    if X.shape[1] != rf_model.n_features_in_:

        print(
            "\nERROR: Feature count does not match RF model."
        )

        print(
            "Capture features:",
            X.shape[1]
        )

        print(
            "RF model expects:",
            rf_model.n_features_in_
        )

        raise SystemExit


# =========================================================
# RANDOM FOREST
# =========================================================

rf_predictions = rf_model.predict(X)

rf_probabilities = rf_model.predict_proba(X)

rf_confidence = np.max(
    rf_probabilities,
    axis=1
)

rf_classes = rf_model.classes_

rf_counts = {
    cls: np.sum(rf_predictions == cls)
    for cls in rf_classes
}

rf_final_index = np.argmax(
    [
        np.sum(rf_predictions == cls)
        for cls in rf_classes
    ]
)

rf_final_class = rf_classes[
    rf_final_index
]

rf_final_confidence = np.mean(
    rf_confidence[
        rf_predictions == rf_final_class
    ]
)


# =========================================================
# MLP
# =========================================================

mlp_predictions = mlp_model.predict(X)

mlp_probabilities = mlp_model.predict_proba(X)

mlp_confidence = np.max(
    mlp_probabilities,
    axis=1
)

mlp_classes = mlp_model.classes_

mlp_counts = {
    cls: np.sum(mlp_predictions == cls)
    for cls in mlp_classes
}

mlp_final_index = np.argmax(
    [
        np.sum(mlp_predictions == cls)
        for cls in mlp_classes
    ]
)

mlp_final_class = mlp_classes[
    mlp_final_index
]

mlp_final_confidence = np.mean(
    mlp_confidence[
        mlp_predictions == mlp_final_class
    ]
)


# =========================================================
# AGREEMENT
# =========================================================

agreement = np.mean(
    rf_predictions == mlp_predictions
) * 100


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 70)
print("RANDOM FOREST RESULT")
print("=" * 70)

print("Prediction distribution:")

for cls, count in rf_counts.items():

    percentage = (
        count / total_windows * 100
    )

    print(
        f"{cls:20s} "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )

print(
    f"\nFinal RF class       : "
    f"{rf_final_class}"
)

print(
    f"Final RF confidence  : "
    f"{rf_final_confidence*100:.2f}%"
)


print()
print("=" * 70)
print("MLP NEURAL NETWORK RESULT")
print("=" * 70)

print("Prediction distribution:")

for cls, count in mlp_counts.items():

    percentage = (
        count / total_windows * 100
    )

    print(
        f"{cls:20s} "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )

print(
    f"\nFinal MLP class       : "
    f"{mlp_final_class}"
)

print(
    f"Final MLP confidence  : "
    f"{mlp_final_confidence*100:.2f}%"
)


print()
print("=" * 70)
print("RF ↔ MLP AGREEMENT")
print("=" * 70)

print(
    f"Agreement : {agreement:.2f}%"
)


# =========================================================
# FINAL DECISION
# =========================================================

print()
print("=" * 70)
print("RFRecon FINAL DECISION")
print("=" * 70)

if rf_final_class == mlp_final_class:

    final_class = rf_final_class

    final_confidence = (
        rf_final_confidence +
        mlp_final_confidence
    ) / 2

    print(
        f"Final classification : "
        f"{final_class}"
    )

    print(
        f"Combined confidence  : "
        f"{final_confidence*100:.2f}%"
    )

    if agreement >= 95:

        print(
            "Status               : "
            "KNOWN / HIGH AGREEMENT"
        )

    else:

        print(
            "Status               : "
            "UNCERTAIN / LOW AGREEMENT"
        )

else:

    print(
        "Final classification : UNKNOWN"
    )

    print(
        "Status               : "
        "RF AND MLP DISAGREE"
    )


print()
print("=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)