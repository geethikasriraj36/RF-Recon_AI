import numpy as np
import pandas as pd

FS = 20e6
WINDOW_SIZE = 32768

noise_file = r"real_dataset\real_noise_01.dat"
wifi_file = r"real_dataset\real_wifi_01.dat"


def extract_features(file_path, label):

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

        p = power / total_power

        # Spectral entropy
        entropy = -np.sum(
            p * np.log2(p + 1e-12)
        )

        entropy /= np.log2(len(p))

        # Peak frequency
        peak_index = np.argmax(power)

        peak_frequency = (
            peak_index / len(power)
        ) - 0.5

        # Spectral centroid
        frequencies = np.linspace(
            -0.5,
            0.5,
            len(power)
        )

        centroid = np.sum(
            frequencies * p
        )

        # Spectral bandwidth
        bandwidth = np.sqrt(
            np.sum(
                ((frequencies - centroid) ** 2) * p
            )
        )

        # Spectral flatness
        geometric_mean = np.exp(
            np.mean(np.log(power + 1e-12))
        )

        arithmetic_mean = np.mean(power)

        flatness = (
            geometric_mean /
            (arithmetic_mean + 1e-12)
        )

        # Peak-to-average ratio
        par = (
            np.max(np.abs(x)) /
            (np.mean(np.abs(x)) + 1e-12)
        )

        results.append({

            "mean_power":
                np.mean(np.abs(x) ** 2),

            "rms":
                np.sqrt(np.mean(np.abs(x) ** 2)),

            "peak_amplitude":
                np.max(np.abs(x)),

            "variance":
                np.var(x),

            "peak_frequency_normalized":
                peak_frequency,

            "spectral_centroid_normalized":
                centroid,

            "spectral_bandwidth_normalized":
                bandwidth,

            "spectral_entropy":
                entropy,

            "spectral_flatness":
                flatness,

            "peak_to_average_ratio":
                par,

            "label":
                label
        })

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i+1}/{n_windows}"
            )

    return pd.DataFrame(results)


# Noise

noise_df = extract_features(
    noise_file,
    "noise"
)

noise_output = (
    r"real_dataset\real_noise_large_window_features.csv"
)

noise_df.to_csv(
    noise_output,
    index=False
)

print("\nSaved:")
print(noise_output)


# WiFi

wifi_df = extract_features(
    wifi_file,
    "wifi_like"
)

wifi_output = (
    r"real_dataset\real_wifi_large_window_features.csv"
)

wifi_df.to_csv(
    wifi_output,
    index=False
)

print("\nSaved:")
print(wifi_output)


print("\n" + "=" * 70)
print("LARGE WINDOW FEATURE EXTRACTION COMPLETE")
print("=" * 70)