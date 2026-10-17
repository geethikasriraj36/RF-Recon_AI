import numpy as np
import pandas as pd
import joblib
from scipy.signal import welch

# ============================================================
# RFRecon - COMPLETE RF RECONNAISSANCE SYSTEM
# PlutoSDR + RF Analysis + Hybrid AI Classification
# ============================================================

CAPTURE_FILE = "real_wifi_capture_02.dat"

SAMPLE_RATE = 20_000_000
CENTER_FREQUENCY = 2_400_000_000

WINDOW_SIZE = 4096

MODEL_FILE = "rfrecon_hybrid_model.pkl"
FEATURE_FILE = "rfrecon_hybrid_features.pkl"


print("=" * 70)
print("RFRecon - COMPLETE RF RECONNAISSANCE SYSTEM")
print("=" * 70)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    signal = signal.astype(np.complex64)

    # Remove DC
    signal = signal - np.mean(signal)

    magnitude = np.abs(signal)

    # Time-domain features
    mean_power = np.mean(magnitude ** 2)

    rms = np.sqrt(
        np.mean(magnitude ** 2)
    )

    peak_amplitude = np.max(magnitude)

    variance = np.var(magnitude)

    # FFT
    spectrum = np.fft.fftshift(
        np.fft.fft(signal)
    )

    power_spectrum = np.abs(spectrum) ** 2

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(signal),
            1 / SAMPLE_RATE
        )
    )

    # Normalize spectrum
    total_power = np.sum(power_spectrum) + 1e-12

    normalized_power = (
        power_spectrum / total_power
    )

    # Peak frequency
    peak_index = np.argmax(power_spectrum)

    peak_frequency = abs(
        frequencies[peak_index]
    )

    peak_frequency_normalized = (
        peak_frequency /
        (SAMPLE_RATE / 2)
    )

    # Spectral centroid
    spectral_centroid = np.sum(
        np.abs(frequencies) *
        normalized_power
    )

    spectral_centroid_normalized = (
        spectral_centroid /
        (SAMPLE_RATE / 2)
    )

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                np.abs(frequencies)
                - spectral_centroid
            ) ** 2
            * normalized_power
        )
    )

    spectral_bandwidth_normalized = (
        spectral_bandwidth /
        (SAMPLE_RATE / 2)
    )

    # Occupied bandwidth - 99% power
    cumulative = np.cumsum(
        normalized_power
    )

    lower_index = np.searchsorted(
        cumulative,
        0.005
    )

    upper_index = np.searchsorted(
        cumulative,
        0.995
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
        occupied_bandwidth /
        SAMPLE_RATE
    )

    # Spectral flatness
    positive_power = (
        power_spectrum + 1e-12
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(positive_power)
        )
    )

    arithmetic_mean = np.mean(
        positive_power
    )

    spectral_flatness = (
        geometric_mean /
        (arithmetic_mean + 1e-12)
    )

    # Peak-to-average ratio
    peak_to_average_ratio = (
        peak_amplitude /
        (rms + 1e-12)
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


# ============================================================
# LOAD AI MODEL
# ============================================================

print("\nLoading AI model...")

model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

print("AI model loaded successfully.")
print("Features:", len(feature_names))


# ============================================================
# LOAD PLUTO SDR CAPTURE
# ============================================================

print("\nLoading PlutoSDR capture...")

samples = np.fromfile(
    CAPTURE_FILE,
    dtype=np.complex64
)

print("Capture:", CAPTURE_FILE)
print("Total samples:", len(samples))
print(
    "Sample rate:",
    SAMPLE_RATE / 1e6,
    "MHz"
)
print(
    "Center frequency:",
    CENTER_FREQUENCY / 1e9,
    "GHz"
)


# ============================================================
# RF ANALYSIS
# ============================================================

print("\nPerforming RF analysis...")


analysis_samples = samples[
    :min(len(samples), 2_000_000)
]

analysis_samples = (
    analysis_samples
    - np.mean(analysis_samples)
)

power_linear = np.mean(
    np.abs(analysis_samples) ** 2
)

relative_power_db = (
    10 * np.log10(
        power_linear + 1e-12
    )
)


# Welch PSD
frequencies, psd = welch(
    analysis_samples,
    fs=SAMPLE_RATE,
    nperseg=WINDOW_SIZE,
    return_onesided=False
)

psd = np.fft.fftshift(psd)
frequencies = np.fft.fftshift(
    frequencies
)

peak_index = np.argmax(psd)

peak_offset = frequencies[
    peak_index
]

peak_frequency = (
    CENTER_FREQUENCY
    + peak_offset
)


# Occupied bandwidth
total_power = np.sum(psd)

cumulative_power = np.cumsum(psd)

lower_index = np.searchsorted(
    cumulative_power,
    0.005 * total_power
)

upper_index = np.searchsorted(
    cumulative_power,
    0.995 * total_power
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


# ============================================================
# AI CLASSIFICATION
# ============================================================

print("\nRunning AI classification...")

windows = []

total_windows = (
    len(samples) // WINDOW_SIZE
)

for i in range(total_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window = samples[
        start:end
    ]

    if len(window) != WINDOW_SIZE:
        continue

    features = extract_features(
        window
    )

    windows.append(features)


print(
    "Windows processed:",
    len(windows)
)


# Convert to DataFrame
X = pd.DataFrame(
    windows
)


# Ensure exact feature order
X = X[
    feature_names
]


# Prevent excessive parallel workers
if hasattr(model, "n_jobs"):
    model.n_jobs = 1


predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence_values = (
    np.max(
        probabilities,
        axis=1
    )
)


# ============================================================
# FINAL AI DECISION
# ============================================================

prediction_counts = (
    pd.Series(predictions)
    .value_counts()
)

final_class = (
    prediction_counts
    .index[0]
)

final_count = (
    prediction_counts
    .iloc[0]
)

window_agreement = (
    final_count /
    len(predictions)
    * 100
)

average_confidence = (
    np.mean(confidence_values)
    * 100
)

minimum_confidence = (
    np.min(confidence_values)
    * 100
)

maximum_confidence = (
    np.max(confidence_values)
    * 100
)


# ============================================================
# FINAL RFRecon RESULT
# ============================================================

print("\n" + "=" * 70)
print("RFRecon - FINAL RESULTS")
print("=" * 70)

print("\nRF ACTIVITY")
print("-" * 70)

print(
    f"Frequency          : "
    f"{peak_frequency / 1e9:.6f} GHz"
)

print(
    f"Frequency Offset   : "
    f"{peak_offset / 1e6:.3f} MHz"
)

print(
    f"Relative Power     : "
    f"{relative_power_db:.2f} dB"
)

print(
    f"Occupied Bandwidth : "
    f"{occupied_bandwidth / 1e6:.3f} MHz"
)


print("\nAI CLASSIFICATION")
print("-" * 70)

print(
    f"Classification     : "
    f"{final_class}"
)

print(
    f"Window Agreement   : "
    f"{window_agreement:.2f}%"
)

print(
    f"Average Confidence : "
    f"{average_confidence:.2f}%"
)

print(
    f"Maximum Confidence : "
    f"{maximum_confidence:.2f}%"
)

print(
    f"Minimum Confidence : "
    f"{minimum_confidence:.2f}%"
)


print("\n" + "=" * 70)
print("FINAL RFRecon DECISION")
print("=" * 70)

print("\nRF Activity Detected")

print(
    f"Frequency     : "
    f"{peak_frequency / 1e9:.6f} GHz"
)

print(
    f"Power         : "
    f"{relative_power_db:.2f} dB (relative)"
)

print(
    f"Bandwidth     : "
    f"{occupied_bandwidth / 1e6:.3f} MHz"
)

print(
    f"Classification: "
    f"{final_class}"
)

print(
    f"Confidence    : "
    f"{average_confidence:.2f}%"
)

print(
    f"Agreement     : "
    f"{window_agreement:.2f}%"
)

print("\n" + "=" * 70)
print("RFRecon ANALYSIS COMPLETE")
print("=" * 70)