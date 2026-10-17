import numpy as np
import pandas as pd
import joblib
import warnings


# ============================================================
# RFRecon COMPLETE SYSTEM
# ADALM-PLUTO + RF ANALYSIS + AI CLASSIFICATION
# Random Forest + MLP Neural Network
# ============================================================

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

# AI models
RF_MODEL_FILE = "rfrecon_hybrid_model.pkl"
RF_FEATURE_FILE = "rfrecon_hybrid_features.pkl"

MLP_MODEL_FILE = "rfrecon_mlp_model.pkl"
MLP_FEATURE_FILE = "rfrecon_mlp_features.pkl"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RFRecon - COMPLETE RF RECONNAISSANCE SYSTEM")
print("=" * 70)

print("\nADALM-PLUTO RF Analysis")
print("Random Forest + MLP Neural Network")


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\nLoading Random Forest model...")

rf_model = joblib.load(
    RF_MODEL_FILE
)

rf_features = list(
    joblib.load(RF_FEATURE_FILE)
)

print("Random Forest loaded successfully.")


# ============================================================
# LOAD MLP
# ============================================================

print("\nLoading MLP Neural Network model...")

mlp_model = joblib.load(
    MLP_MODEL_FILE
)

mlp_features = list(
    joblib.load(MLP_FEATURE_FILE)
)

print("MLP loaded successfully.")


# ============================================================
# CHECK FEATURES
# ============================================================

if rf_features != mlp_features:

    raise ValueError(
        "Random Forest and MLP feature lists do not match."
    )

features = rf_features

print("\nFeature compatibility: OK")

print("\nFeatures:")

for i, feature in enumerate(
    features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# LOAD PLUTO SDR CAPTURE
# ============================================================

print("\n" + "=" * 70)
print("LOADING PLUTO SDR CAPTURE")
print("=" * 70)

iq = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

if len(iq) < WINDOW_SIZE:

    raise ValueError(
        "Capture is too small for analysis."
    )


print("\nCapture file:")
print(CAPTURE_FILE)

print("\nTotal I/Q samples:")
print(
    f"{len(iq):,}"
)

print("\nSample rate:")
print(
    f"{SAMPLE_RATE / 1e6:.2f} MS/s"
)

print("\nCenter frequency:")
print(
    f"{CENTER_FREQUENCY / 1e9:.6f} GHz"
)


# ============================================================
# RF ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("RF SIGNAL ANALYSIS")
print("=" * 70)

print("\nCalculating spectrum...")


# FFT of complete capture
spectrum = np.fft.fftshift(
    np.fft.fft(iq)
)

magnitude = np.abs(
    spectrum
)

power_spectrum = (
    magnitude ** 2
)


# Frequency axis
freqs = np.fft.fftshift(
    np.fft.fftfreq(
        len(iq),
        d=1 / SAMPLE_RATE
    )
)


# ============================================================
# PEAK FREQUENCY
# ============================================================

peak_index = np.argmax(
    power_spectrum
)

peak_frequency_offset = (
    freqs[peak_index]
)

peak_frequency = (
    CENTER_FREQUENCY
    + peak_frequency_offset
)


# ============================================================
# FREQUENCY OFFSET
# ============================================================

frequency_offset = (
    peak_frequency
    - CENTER_FREQUENCY
)


# ============================================================
# RELATIVE POWER
# ============================================================

power_db = 10 * np.log10(
    power_spectrum
    / (
        np.max(power_spectrum)
        + 1e-12
    )
)

relative_power = np.mean(
    10 * np.log10(
        np.abs(iq) ** 2
        + 1e-12
    )
)


# ============================================================
# PEAK PSD
# ============================================================

peak_psd = np.max(
    10 * np.log10(
        power_spectrum
        + 1e-12
    )
)


# ============================================================
# OCCUPIED BANDWIDTH
# ============================================================

# 1% of maximum spectral power
threshold = (
    0.01
    * np.max(power_spectrum)
)

occupied_frequencies = freqs[
    power_spectrum >= threshold
]

if len(occupied_frequencies) > 0:

    occupied_bandwidth = (
        np.max(
            occupied_frequencies
        )
        -
        np.min(
            occupied_frequencies
        )
    )

else:

    occupied_bandwidth = 0


# ============================================================
# RF RESULTS
# ============================================================

print("\n" + "-" * 70)
print("RF CHARACTERISTICS")
print("-" * 70)

print(
    f"\nCenter frequency:"
    f" {CENTER_FREQUENCY / 1e9:.6f} GHz"
)

print(
    f"Peak frequency:"
    f" {peak_frequency / 1e9:.6f} GHz"
)

print(
    f"Frequency offset:"
    f" {frequency_offset / 1e6:+.6f} MHz"
)

print(
    f"Relative power:"
    f" {relative_power:.2f} dB"
)

print(
    f"Peak PSD:"
    f" {peak_psd:.2f} dB"
)

print(
    f"Occupied bandwidth:"
    f" {occupied_bandwidth / 1e6:.3f} MHz"
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(signal):

    # --------------------------------------------------------
    # Time-domain features
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
            -
            np.min(occupied)
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
# WINDOW PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("AI FEATURE EXTRACTION")
print("=" * 70)

num_windows = (
    len(iq)
    // WINDOW_SIZE
)

print("\nWindow size:")
print(WINDOW_SIZE)

print("\nNumber of complete windows:")
print(
    f"{num_windows:,}"
)


feature_matrix = []


for i in range(
    num_windows
):

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

    if (i + 1) % 500 == 0:

        print(
            f"Processed "
            f"{i + 1:,}/"
            f"{num_windows:,} windows"
        )


X = pd.DataFrame(
    feature_matrix,
    columns=features
)

print("\nFeature matrix:")
print(
    X.shape
)


# ============================================================
# RANDOM FOREST CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION")
print("=" * 70)

rf_predictions = rf_model.predict(
    X
)

rf_probabilities = (
    rf_model.predict_proba(X)
)

rf_confidences = np.max(
    rf_probabilities,
    axis=1
)

rf_counts = pd.Series(
    rf_predictions
).value_counts()

rf_dominant = (
    rf_counts.index[0]
)

rf_agreement = (
    rf_counts.iloc[0]
    / num_windows
) * 100

rf_average_confidence = (
    np.mean(rf_confidences)
    * 100
)

rf_max_confidence = (
    np.max(rf_confidences)
    * 100
)

rf_min_confidence = (
    np.min(rf_confidences)
    * 100
)


print("\nPrediction distribution:")
print(
    rf_counts
)

print(
    "\nDominant classification:"
)

print(
    rf_dominant
)

print(
    f"\nWindow agreement:"
    f" {rf_agreement:.2f}%"
)

print(
    f"Average confidence:"
    f" {rf_average_confidence:.2f}%"
)

print(
    f"Maximum confidence:"
    f" {rf_max_confidence:.2f}%"
)

print(
    f"Minimum confidence:"
    f" {rf_min_confidence:.2f}%"
)


# ============================================================
# MLP CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("MLP NEURAL NETWORK CLASSIFICATION")
print("=" * 70)

mlp_predictions = mlp_model.predict(
    X
)

mlp_probabilities = (
    mlp_model.predict_proba(X)
)

mlp_confidences = np.max(
    mlp_probabilities,
    axis=1
)

mlp_counts = pd.Series(
    mlp_predictions
).value_counts()

mlp_dominant = (
    mlp_counts.index[0]
)

mlp_agreement = (
    mlp_counts.iloc[0]
    / num_windows
) * 100

mlp_average_confidence = (
    np.mean(mlp_confidences)
    * 100
)

mlp_max_confidence = (
    np.max(mlp_confidences)
    * 100
)

mlp_min_confidence = (
    np.min(mlp_confidences)
    * 100
)


print("\nPrediction distribution:")
print(
    mlp_counts
)

print(
    "\nDominant classification:"
)

print(
    mlp_dominant
)

print(
    f"\nWindow agreement:"
    f" {mlp_agreement:.2f}%"
)

print(
    f"Average confidence:"
    f" {mlp_average_confidence:.2f}%"
)

print(
    f"Maximum confidence:"
    f" {mlp_max_confidence:.2f}%"
)

print(
    f"Minimum confidence:"
    f" {mlp_min_confidence:.2f}%"
)


# ============================================================
# MODEL AGREEMENT
# ============================================================

same_prediction = (
    rf_predictions
    ==
    mlp_predictions
)

agreement_count = np.sum(
    same_prediction
)

model_agreement = (
    agreement_count
    / num_windows
) * 100


# ============================================================
# FINAL RFRecon RESULT
# ============================================================

print("\n" + "=" * 70)
print("RFRecon FINAL RESULT")
print("=" * 70)

print("\nRF CHARACTERISTICS")
print("-" * 40)

print(
    f"Center frequency      : "
    f"{CENTER_FREQUENCY / 1e9:.6f} GHz"
)

print(
    f"Peak frequency        : "
    f"{peak_frequency / 1e9:.6f} GHz"
)

print(
    f"Frequency offset      : "
    f"{frequency_offset / 1e6:+.6f} MHz"
)

print(
    f"Relative power        : "
    f"{relative_power:.2f} dB"
)

print(
    f"Peak PSD              : "
    f"{peak_psd:.2f} dB"
)

print(
    f"Occupied bandwidth    : "
    f"{occupied_bandwidth / 1e6:.3f} MHz"
)


print("\nAI CLASSIFICATION")
print("-" * 40)

print(
    f"Random Forest class   : "
    f"{rf_dominant}"
)

print(
    f"Random Forest agreement: "
    f"{rf_agreement:.2f}%"
)

print(
    f"Random Forest confidence: "
    f"{rf_average_confidence:.2f}%"
)


print(
    f"\nMLP class             : "
    f"{mlp_dominant}"
)

print(
    f"MLP agreement         : "
    f"{mlp_agreement:.2f}%"
)

print(
    f"MLP confidence        : "
    f"{mlp_average_confidence:.2f}%"
)


print(
    f"\nRF ↔ MLP agreement     : "
    f"{model_agreement:.2f}%"
)


# ============================================================
# FINAL DETECTION
# ============================================================

print("\n" + "-" * 40)

if rf_dominant == mlp_dominant:

    print(
        "FINAL DETECTION: "
        f"{rf_dominant}"
    )

    print(
        "Both AI models agree "
        "on the dominant class."
    )

else:

    print(
        "FINAL DETECTION: "
        f"{rf_dominant}"
    )

    print(
        "WARNING: AI models disagree."
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
    "rfrecon_complete_window_results.csv",
    index=False
)


# ============================================================
# SAVE RF SUMMARY
# ============================================================

summary = pd.DataFrame({

    "parameter": [

        "Capture file",
        "Total I/Q samples",
        "Sample rate",
        "Center frequency",
        "Peak frequency",
        "Frequency offset",
        "Relative power",
        "Peak PSD",
        "Occupied bandwidth",
        "Window size",
        "Number of windows",

        "RF dominant class",
        "RF window agreement",
        "RF average confidence",

        "MLP dominant class",
        "MLP window agreement",
        "MLP average confidence",

        "RF-MLP agreement"

    ],

    "value": [

        CAPTURE_FILE,
        len(iq),
        f"{SAMPLE_RATE / 1e6:.2f} MS/s",
        f"{CENTER_FREQUENCY / 1e9:.6f} GHz",
        f"{peak_frequency / 1e9:.6f} GHz",
        f"{frequency_offset / 1e6:+.6f} MHz",
        f"{relative_power:.2f} dB",
        f"{peak_psd:.2f} dB",
        f"{occupied_bandwidth / 1e6:.3f} MHz",
        WINDOW_SIZE,
        num_windows,

        rf_dominant,
        f"{rf_agreement:.2f}%",
        f"{rf_average_confidence:.2f}%",

        mlp_dominant,
        f"{mlp_agreement:.2f}%",
        f"{mlp_average_confidence:.2f}%",

        f"{model_agreement:.2f}%"

    ]

})


summary.to_csv(
    "rfrecon_complete_summary.csv",
    index=False
)


# ============================================================
# SAVE FEATURE MATRIX
# ============================================================

X.to_csv(
    "rfrecon_complete_features.csv",
    index=False
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(
    "\n1. rfrecon_complete_window_results.csv"
)

print(
    "2. rfrecon_complete_summary.csv"
)

print(
    "3. rfrecon_complete_features.csv"
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("RFRecon COMPLETE SYSTEM FINISHED")
print("=" * 70)