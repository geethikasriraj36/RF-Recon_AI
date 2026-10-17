import numpy as np
import pandas as pd

NOISE_FILE = r"real_dataset\real_noise_01.dat"
WIFI_FILE = r"real_dataset\real_wifi_01.dat"

NOISE_OUTPUT = r"real_dataset\real_noise_enhanced_features.csv"
WIFI_OUTPUT = r"real_dataset\real_wifi_enhanced_features.csv"

SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096


FEATURES = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
    "zero_crossing_rate",
    "amplitude_std",
    "amplitude_kurtosis",
    "power_std",
    "power_variation"
]


def extract_features(x):

    amplitude = np.abs(x)
    power = amplitude ** 2

    # =====================================================
    # EXISTING FEATURES
    # =====================================================

    mean_power = np.mean(power)

    rms = np.sqrt(mean_power)

    peak_amplitude = np.max(amplitude)

    variance = np.var(amplitude)

    window = np.hanning(len(x))

    spectrum = np.fft.fftshift(
        np.fft.fft(x * window)
    )

    power_spectrum = (
        np.abs(spectrum) ** 2
        + 1e-12
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(x),
            d=1 / SAMPLE_RATE
        )
    )

    peak_index = np.argmax(power_spectrum)

    peak_frequency = frequencies[peak_index]

    peak_frequency_normalized = (
        peak_frequency / (SAMPLE_RATE / 2)
    )

    spectral_centroid = (
        np.sum(
            np.abs(frequencies)
            * power_spectrum
        )
        /
        np.sum(power_spectrum)
    )

    spectral_centroid_normalized = (
        spectral_centroid
        / (SAMPLE_RATE / 2)
    )

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
        spectral_bandwidth
        / (SAMPLE_RATE / 2)
    )

    # 95% occupied bandwidth

    sorted_indices = np.argsort(
        power_spectrum
    )

    sorted_power = power_spectrum[
        sorted_indices
    ]

    cumulative = np.cumsum(
        sorted_power
    )

    total = cumulative[-1]

    low_idx = np.searchsorted(
        cumulative,
        0.025 * total
    )

    high_idx = np.searchsorted(
        cumulative,
        0.975 * total
    )

    low_freq = frequencies[
        sorted_indices[low_idx]
    ]

    high_freq = frequencies[
        sorted_indices[high_idx]
    ]

    occupied_bandwidth = (
        high_freq - low_freq
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth
        / SAMPLE_RATE
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(power_spectrum)
        )
    )

    arithmetic_mean = np.mean(
        power_spectrum
    )

    spectral_flatness = (
        geometric_mean
        / arithmetic_mean
    )

    peak_to_average_ratio = (
        np.max(power)
        /
        (np.mean(power) + 1e-12)
    )

    # =====================================================
    # NEW TEMPORAL FEATURES
    # =====================================================

    # 1. Zero crossing rate of real component

    real_signal = np.real(x)

    zero_crossings = np.sum(
        real_signal[:-1]
        * real_signal[1:]
        < 0
    )

    zero_crossing_rate = (
        zero_crossings
        / (len(real_signal) - 1)
    )

    # 2. Amplitude standard deviation

    amplitude_std = np.std(
        amplitude
    )

    # 3. Amplitude kurtosis
    #
    # Pearson kurtosis approximation

    centered_amplitude = (
        amplitude - np.mean(amplitude)
    )

    amplitude_variance = np.mean(
        centered_amplitude ** 2
    )

    if amplitude_variance > 1e-12:

        amplitude_kurtosis = (
            np.mean(
                centered_amplitude ** 4
            )
            /
            (amplitude_variance ** 2)
        )

    else:

        amplitude_kurtosis = 0.0

    # 4. Power standard deviation

    power_std = np.std(power)

    # 5. Power variation

    power_variation = (
        power_std
        /
        (mean_power + 1e-12)
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
        peak_to_average_ratio,
        zero_crossing_rate,
        amplitude_std,
        amplitude_kurtosis,
        power_std,
        power_variation
    ]


def process_file(input_file, output_file, label):

    data = np.memmap(
        input_file,
        dtype=np.complex64,
        mode="r"
    )

    num_windows = (
        len(data) // WINDOW_SIZE
    )

    print()
    print("=" * 70)
    print(f"Processing: {label}")
    print("=" * 70)

    print(
        f"Samples : {len(data):,}"
    )

    print(
        f"Windows : {num_windows:,}"
    )

    rows = []

    for i in range(num_windows):

        start = i * WINDOW_SIZE

        end = start + WINDOW_SIZE

        x = data[start:end]

        values = extract_features(x)

        row = dict(
            zip(FEATURES, values)
        )

        row["label"] = label

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("Saved:")
    print(output_file)

    print()
    print("FEATURE MEANS")
    print("-" * 70)

    for feature in FEATURES:

        print(
            f"{feature:35s}"
            f"{df[feature].mean():.6f}"
        )


# =========================================================
# PROCESS BOTH DATASETS
# =========================================================

process_file(
    NOISE_FILE,
    NOISE_OUTPUT,
    "noise"
)

process_file(
    WIFI_FILE,
    WIFI_OUTPUT,
    "wifi_like"
)

print()
print("=" * 70)
print("ENHANCED FEATURE EXTRACTION COMPLETE")
print("=" * 70)