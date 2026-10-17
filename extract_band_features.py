import numpy as np
import pandas as pd

FS = 20e6
WINDOW_SIZE = 4096
NUM_BANDS = 10

noise_file = r"real_dataset\real_noise_01.dat"
wifi_file = r"real_dataset\real_wifi_01.dat"


def extract_band_features(file_path, label):

    print("=" * 70)
    print("Processing:", label)
    print("=" * 70)

    data = np.fromfile(
        file_path,
        dtype=np.complex64
    )

    n_windows = len(data) // WINDOW_SIZE

    print("Samples :", len(data))
    print("Windows :", n_windows)

    results = []

    for i in range(n_windows):

        x = data[
            i * WINDOW_SIZE:
            (i + 1) * WINDOW_SIZE
        ]

        spectrum = np.fft.fftshift(
            np.fft.fft(x)
        )

        power = np.abs(spectrum) ** 2

        total_power = np.sum(power) + 1e-12

        # Divide FFT into equal frequency bands
        band_edges = np.linspace(
            0,
            len(power),
            NUM_BANDS + 1,
            dtype=int
        )

        features = {}

        for b in range(NUM_BANDS):

            start = band_edges[b]
            end = band_edges[b + 1]

            band_power = np.sum(
                power[start:end]
            )

            normalized_power = (
                band_power / total_power
            )

            features[
                f"band_power_{b+1:02d}"
            ] = normalized_power

        features["label"] = label

        results.append(features)

        if (i + 1) % 5000 == 0:

            print(
                f"Processed {i+1}/{n_windows}"
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Noise
# ---------------------------------------------------------

noise_df = extract_band_features(
    noise_file,
    "noise"
)

noise_output = (
    r"real_dataset\real_noise_band_features.csv"
)

noise_df.to_csv(
    noise_output,
    index=False
)

print("\nSaved:")
print(noise_output)


# ---------------------------------------------------------
# WiFi
# ---------------------------------------------------------

wifi_df = extract_band_features(
    wifi_file,
    "wifi_like"
)

wifi_output = (
    r"real_dataset\real_wifi_band_features.csv"
)

wifi_df.to_csv(
    wifi_output,
    index=False
)

print("\nSaved:")
print(wifi_output)


print("\n" + "=" * 70)
print("BAND FEATURE EXTRACTION COMPLETE")
print("=" * 70)