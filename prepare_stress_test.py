import numpy as np
import pandas as pd
import os

from scipy.signal import periodogram


SAMPLE_RATE = 2_000_000

DATASET_FOLDER = "stress_test_data"

OUTPUT_FILE = "stress_test_dataset.csv"


class_labels = {
    "noise": 0,
    "wifi_like": 1,
    "bluetooth_like": 2,
    "zigbee_like": 3,
    "cellular_like": 4
}


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

    occupied_bandwidth = abs(
        frequencies[
            min(upper_index, len(frequencies) - 1)
        ]
        -
        frequencies[
            min(lower_index, len(frequencies) - 1)
        ]
    )

    occupied_bandwidth_normalized = (
        occupied_bandwidth / SAMPLE_RATE
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(psd + 1e-12)
        )
    )

    arithmetic_mean = np.mean(psd) + 1e-12

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


rows = []

print("=" * 65)
print("RFRecon - External Test Feature Extraction")
print("=" * 65)

print("\nReading independent test dataset...\n")


for class_name, label in class_labels.items():

    class_folder = os.path.join(
        DATASET_FOLDER,
        class_name
    )

    files = [
        f for f in os.listdir(class_folder)
        if f.endswith(".npy")
    ]

    print(
        f"Processing {class_name}: {len(files)} files"
    )

    for filename in files:

        filepath = os.path.join(
            class_folder,
            filename
        )

        signal = np.load(filepath)

        features = extract_features(signal)

        rows.append(
            features + [class_name, label]
        )


columns = feature_names + [
    "class_name",
    "label"
]

df = pd.DataFrame(
    rows,
    columns=columns
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 65)
print("EXTERNAL TEST FEATURES COMPLETE")
print("=" * 65)

print("\nTotal examples:", len(df))

print("\nClass distribution:")

print(
    df["class_name"].value_counts()
)

print(
    "\nSaved as:",
    OUTPUT_FILE
)