import numpy as np
import pandas as pd
import joblib
import warnings


# ============================================================
# RFRecon FINAL AI SYSTEM
# Random Forest + MLP Neural Network
# Optimized Batch Prediction Version
# ============================================================

# Suppress the repeated scikit-learn parallel warning
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*"
)


# ============================================================
# CONFIGURATION
# ============================================================

CAPTURE_FILE = "real_wifi_capture_03.dat"

SAMPLE_RATE = 20_000_000
CENTER_FREQUENCY = 2_400_000_000
WINDOW_SIZE = 4096

# Random Forest
RF_MODEL_FILE = "rfrecon_hybrid_model.pkl"
RF_FEATURE_FILE = "rfrecon_hybrid_features.pkl"

# MLP
MLP_MODEL_FILE = "rfrecon_mlp_model.pkl"
MLP_FEATURE_FILE = "rfrecon_mlp_features.pkl"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("RFRecon FINAL AI SYSTEM")
print("Random Forest + MLP Neural Network")
print("=" * 60)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\nLoading Random Forest model...")

rf_model = joblib.load(
    RF_MODEL_FILE
)

rf_features = joblib.load(
    RF_FEATURE_FILE
)

print("Random Forest loaded successfully.")


# ============================================================
# LOAD MLP
# ============================================================

print("\nLoading MLP Neural Network model...")

mlp_model = joblib.load(
    MLP_MODEL_FILE
)

mlp_features = joblib.load(
    MLP_FEATURE_FILE
)

print("MLP model loaded successfully.")


# ============================================================
# NORMALIZE FEATURE LISTS
# ============================================================

rf_features = list(rf_features)
mlp_features = list(mlp_features)


# ============================================================
# CHECK FEATURE COMPATIBILITY
# ============================================================

print("\nChecking feature compatibility...")

if rf_features != mlp_features:

    print("\nERROR:")
    print("Random Forest and MLP use different features.")

    print("\nRandom Forest features:")
    print(rf_features)

    print("\nMLP features:")
    print(mlp_features)

    raise ValueError(
        "RF and MLP feature lists do not match."
    )


features = rf_features

print("Feature lists match successfully.")

print("\nFeatures used:")

for i, feature in enumerate(
    features,
    start=1
):
    print(
        f"{i}. {feature}"
    )


# ============================================================
# LOAD REAL PLUTO SDR CAPTURE
# ============================================================

print("\n" + "=" * 60)
print("LOADING REAL PLUTO SDR CAPTURE")
print("=" * 60)

iq = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

print("\nCapture file:")
print(CAPTURE_FILE)

print("\nTotal I/Q samples:")
print(f"{len(iq):,}")

print("\nSample rate:")
print(
    f"{SAMPLE_RATE / 1e6:.1f} MS/s"
)

print("\nCenter frequency:")
print(
    f"{CENTER_FREQUENCY / 1e9:.3f} GHz"
)


# ============================================================
# CHECK CAPTURE
# ============================================================

if len(iq) < WINDOW_SIZE:

    raise ValueError(
        "Capture contains fewer samples "
        "than one analysis window."
    )


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(signal):

    # --------------------------------------------------------
    # Time-domain quantities
    # --------------------------------------------------------

    power = np.abs(signal) ** 2

    amplitude = np.abs(signal)

    mean_power = np.mean(
        power
    )

    rms = np.sqrt(
        np.mean(power)
    )

    peak_amplitude = np.max(
        amplitude
    )

    variance = np.var(
        power
    )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    spectrum = np.fft.fftshift(
        np.fft.fft(signal)
    )

    magnitude = np.abs(
        spectrum
    )

    power_spectrum = (
        magnitude ** 2
    )


    # --------------------------------------------------------
    # Frequency axis
    # --------------------------------------------------------

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            len(signal),
            d=1 / SAMPLE_RATE
        )
    )


    # --------------------------------------------------------
    # Total spectral power
    # --------------------------------------------------------

    total_power = (
        np.sum(power_spectrum)
        + 1e-12
    )


    # --------------------------------------------------------
    # Peak frequency
    # --------------------------------------------------------

    peak_index = np.argmax(
        power_spectrum
    )

    peak_frequency = (
        freqs[peak_index]
    )

    peak_frequency_normalized = (
        peak_frequency
        / (SAMPLE_RATE / 2)
    )


    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(
            np.abs(freqs)
            * power_spectrum
        )
        / total_power
    )

    spectral_centroid_normalized = (
        spectral_centroid
        / (SAMPLE_RATE / 2)
    )


    # --------------------------------------------------------
    # Spectral bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(freqs)
                - spectral_centroid
            ) ** 2
            * power_spectrum
        )
        / total_power
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth
        / (SAMPLE_RATE / 2)
    )


    # --------------------------------------------------------
    # Occupied bandwidth
    # --------------------------------------------------------

    threshold = (
        0.01
        * np.max(power_spectrum)
    )

    occupied = freqs[
        power_spectrum >= threshold
    ]

    if len(occupied) > 0:

        occupied_bandwidth = (
            np.max(occupied)
            - np.min(occupied)
        )

    else:

        occupied_bandwidth = 0


    occupied_bandwidth_normalized = (
        occupied_bandwidth
        / SAMPLE_RATE
    )


    # --------------------------------------------------------
    # Spectral flatness
    # --------------------------------------------------------

    geometric_mean = np.exp(
        np.mean(
            np.log(
                power_spectrum
                + 1e-12
            )
        )
    )

    arithmetic_mean = (
        np.mean(power_spectrum)
        + 1e-12
    )

    spectral_flatness = (
        geometric_mean
        / arithmetic_mean
    )


    # --------------------------------------------------------
    # Peak-to-average ratio
    # --------------------------------------------------------

    peak_to_average_ratio = (
        peak_amplitude
        / (rms + 1e-12)
    )


    # --------------------------------------------------------
    # Return 10 features
    # --------------------------------------------------------

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
# NUMBER OF WINDOWS
# ============================================================

num_windows = (
    len(iq)
    // WINDOW_SIZE
)

print("\n" + "=" * 60)
print("RF WINDOW PROCESSING")
print("=" * 60)

print("\nWindow size:")
print(WINDOW_SIZE)

print("\nNumber of complete windows:")
print(
    f"{num_windows:,}"
)


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

print("\nExtracting features from all windows...")

feature_matrix = []

for i in range(num_windows):

    start = (
        i * WINDOW_SIZE
    )

    end = (
        start + WINDOW_SIZE
    )

    window = iq[
        start:end
    ]

    feature_vector = extract_features(
        window
    )

    feature_matrix.append(
        feature_vector
    )

    # Progress display
    if (i + 1) % 500 == 0:

        print(
            f"Processed "
            f"{i + 1:,}/{num_windows:,} windows"
        )


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

X = pd.DataFrame(
    feature_matrix,
    columns=features
)

print("\nFeature extraction completed.")

print("\nFeature matrix shape:")

print(
    X.shape
)


# ============================================================
# RANDOM FOREST BATCH PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST PREDICTION")
print("=" * 60)

print("\nRunning Random Forest on all windows...")

rf_predictions = rf_model.predict(
    X
)

rf_probabilities = rf_model.predict_proba(
    X
)

rf_confidences = np.max(
    rf_probabilities,
    axis=1
)

print(
    "Random Forest prediction completed."
)


# ============================================================
# MLP BATCH PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("MLP NEURAL NETWORK PREDICTION")
print("=" * 60)

print("\nRunning MLP on all windows...")

mlp_predictions = mlp_model.predict(
    X
)

mlp_probabilities = mlp_model.predict_proba(
    X
)

mlp_confidences = np.max(
    mlp_probabilities,
    axis=1
)

print(
    "MLP prediction completed."
)


# ============================================================
# RANDOM FOREST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

rf_counts = pd.Series(
    rf_predictions
).value_counts()

print("\nPrediction distribution:")

print(
    rf_counts
)


rf_dominant = (
    rf_counts.index[0]
)

rf_dominant_count = (
    rf_counts.iloc[0]
)

rf_agreement = (
    rf_dominant_count
    / num_windows
) * 100

rf_average_confidence = (
    np.mean(
        rf_confidences
    )
    * 100
)

rf_max_confidence = (
    np.max(
        rf_confidences
    )
    * 100
)

rf_min_confidence = (
    np.min(
        rf_confidences
    )
    * 100
)


print(
    "\nDominant classification:"
)

print(
    rf_dominant
)

print(
    f"\nWindow agreement: "
    f"{rf_agreement:.2f}%"
)

print(
    f"Average confidence: "
    f"{rf_average_confidence:.2f}%"
)

print(
    f"Maximum confidence: "
    f"{rf_max_confidence:.2f}%"
)

print(
    f"Minimum confidence: "
    f"{rf_min_confidence:.2f}%"
)


# ============================================================
# MLP RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MLP NEURAL NETWORK RESULTS")
print("=" * 60)

mlp_counts = pd.Series(
    mlp_predictions
).value_counts()

print("\nPrediction distribution:")

print(
    mlp_counts
)


mlp_dominant = (
    mlp_counts.index[0]
)

mlp_dominant_count = (
    mlp_counts.iloc[0]
)

mlp_agreement = (
    mlp_dominant_count
    / num_windows
) * 100

mlp_average_confidence = (
    np.mean(
        mlp_confidences
    )
    * 100
)

mlp_max_confidence = (
    np.max(
        mlp_confidences
    )
    * 100
)

mlp_min_confidence = (
    np.min(
        mlp_confidences
    )
    * 100
)


print(
    "\nDominant classification:"
)

print(
    mlp_dominant
)

print(
    f"\nWindow agreement: "
    f"{mlp_agreement:.2f}%"
)

print(
    f"Average confidence: "
    f"{mlp_average_confidence:.2f}%"
)

print(
    f"Maximum confidence: "
    f"{mlp_max_confidence:.2f}%"
)

print(
    f"Minimum confidence: "
    f"{mlp_min_confidence:.2f}%"
)


# ============================================================
# RF vs MLP AGREEMENT
# ============================================================

same_prediction = (
    rf_predictions
    == mlp_predictions
)

model_agreement_count = np.sum(
    same_prediction
)

model_agreement = (
    model_agreement_count
    / num_windows
) * 100


print("\n" + "=" * 60)
print("MODEL AGREEMENT")
print("=" * 60)

print(
    "\nWindows where RF and MLP agree:"
)

print(
    f"{model_agreement_count:,}"
)

print(
    "\nTotal windows:"
)

print(
    f"{num_windows:,}"
)

print(
    f"\nRF vs MLP prediction agreement:"
    f" {model_agreement:.2f}%"
)


# ============================================================
# FINAL RFRecon RESULT
# ============================================================

print("\n" + "=" * 60)
print("RFRecon FINAL AI RESULT")
print("=" * 60)

print("\nPrimary classifier:")
print(
    "Random Forest"
)

print("\nComparative classifier:")
print(
    "MLP Neural Network"
)

print("\nDetected RF class:")
print(
    rf_dominant
)

print(
    "\nRandom Forest confidence:"
    f" {rf_average_confidence:.2f}%"
)

print(
    "MLP confidence:"
    f" {mlp_average_confidence:.2f}%"
)

print(
    "\nRandom Forest window agreement:"
    f" {rf_agreement:.2f}%"
)

print(
    "MLP window agreement:"
    f" {mlp_agreement:.2f}%"
)

print(
    "\nRF vs MLP prediction agreement:"
    f" {model_agreement:.2f}%"
)


# ============================================================
# DOMINANT CLASS AGREEMENT
# ============================================================

if rf_dominant == mlp_dominant:

    print(
        "\nBoth models agree on the "
        "dominant signal class."
    )

else:

    print(
        "\nWARNING: Models disagree on "
        "the dominant signal class."
    )


# ============================================================
# SAVE WINDOW-LEVEL RESULTS
# ============================================================

comparison = pd.DataFrame({

    "window":
        np.arange(num_windows),

    "rf_prediction":
        rf_predictions,

    "rf_confidence":
        rf_confidences,

    "mlp_prediction":
        mlp_predictions,

    "mlp_confidence":
        mlp_confidences

})


comparison.to_csv(
    "rfrecon_final_model_comparison.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "Capture file",

        "Total samples",

        "Sample rate",

        "Center frequency",

        "Window size",

        "Number of windows",

        "RF dominant class",

        "RF window agreement",

        "RF average confidence",

        "RF maximum confidence",

        "RF minimum confidence",

        "MLP dominant class",

        "MLP window agreement",

        "MLP average confidence",

        "MLP maximum confidence",

        "MLP minimum confidence",

        "RF vs MLP agreement"

    ],

    "value": [

        CAPTURE_FILE,

        len(iq),

        f"{SAMPLE_RATE / 1e6:.2f} MS/s",

        f"{CENTER_FREQUENCY / 1e9:.3f} GHz",

        WINDOW_SIZE,

        num_windows,

        rf_dominant,

        f"{rf_agreement:.2f}%",

        f"{rf_average_confidence:.2f}%",

        f"{rf_max_confidence:.2f}%",

        f"{rf_min_confidence:.2f}%",

        mlp_dominant,

        f"{mlp_agreement:.2f}%",

        f"{mlp_average_confidence:.2f}%",

        f"{mlp_max_confidence:.2f}%",

        f"{mlp_min_confidence:.2f}%",

        f"{model_agreement:.2f}%"

    ]

})


summary.to_csv(
    "rfrecon_final_summary.csv",
    index=False
)


# ============================================================
# SAVE FEATURE MATRIX
# ============================================================

X.to_csv(
    "rfrecon_final_features.csv",
    index=False
)


# ============================================================
# FINAL FILES
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    "\n1. rfrecon_final_model_comparison.csv"
)

print(
    "2. rfrecon_final_summary.csv"
)

print(
    "3. rfrecon_final_features.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("RFRecon FINAL SYSTEM COMPLETED")
print("=" * 60)