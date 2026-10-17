import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FS = 20e6
WINDOW_SIZE = 4096

noise_file = r"real_dataset\real_noise_01.dat"
wifi_file = r"real_dataset\real_wifi_01.dat"

noise_output = r"real_dataset\real_noise_spectral_features.csv"
wifi_output = r"real_dataset\real_wifi_spectral_features.csv"


def extract_spectral_features(file_path, label):

    print("=" * 70)
    print("Processing:", label)
    print("=" * 70)

    data = np.fromfile(file_path, dtype=np.complex64)

    n_windows = len(data) // WINDOW_SIZE

    print("Samples :", len(data))
    print("Windows :", n_windows)

    results = []

    for i in range(n_windows):

        x = data[
            i * WINDOW_SIZE:
            (i + 1) * WINDOW_SIZE
        ]

        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(x))

        power = np.abs(spectrum) ** 2

        # Normalize power spectrum
        total_power = np.sum(power) + 1e-12

        p = power / total_power

        # --------------------------------------------------
        # 1. Spectral entropy
        # --------------------------------------------------

        spectral_entropy = -np.sum(
            p * np.log2(p + 1e-12)
        )

        spectral_entropy /= np.log2(len(p))

        # --------------------------------------------------
        # 2. Spectral peak prominence
        # --------------------------------------------------

        peaks, properties = find_peaks(
            power,
            prominence=np.max(power) * 0.02
        )

        if len(peaks) > 0:

            peak_prominence = np.max(
                properties["prominences"]
            ) / (np.mean(power) + 1e-12)

        else:

            peak_prominence = 0.0

        # --------------------------------------------------
        # 3. Number of significant spectral peaks
        # --------------------------------------------------

        number_of_peaks = len(peaks)

        # --------------------------------------------------
        # 4. Spectral rolloff
        # --------------------------------------------------

        cumulative = np.cumsum(power)

        rolloff_index = np.searchsorted(
            cumulative,
            0.85 * cumulative[-1]
        )

        spectral_rolloff = (
            rolloff_index / len(power)
        )

        # --------------------------------------------------
        # 5. Spectral flatness
        # --------------------------------------------------

        geometric_mean = np.exp(
            np.mean(np.log(power + 1e-12))
        )

        arithmetic_mean = np.mean(power)

        spectral_flatness = (
            geometric_mean /
            (arithmetic_mean + 1e-12)
        )

        results.append({

            "spectral_entropy": spectral_entropy,

            "spectral_peak_prominence":
                peak_prominence,

            "number_of_spectral_peaks":
                number_of_peaks,

            "spectral_rolloff":
                spectral_rolloff,

            "spectral_flatness":
                spectral_flatness,

            "label": label
        })

        if (i + 1) % 5000 == 0:

            print(
                f"Processed {i + 1}/{n_windows}"
            )

    df = pd.DataFrame(results)

    return df


# ----------------------------------------------------------
# Process Noise
# ----------------------------------------------------------

noise_df = extract_spectral_features(
    noise_file,
    "noise"
)

noise_df.to_csv(
    noise_output,
    index=False
)

print("\nSaved:")
print(noise_output)


# ----------------------------------------------------------
# Process WiFi
# ----------------------------------------------------------

wifi_df = extract_spectral_features(
    wifi_file,
    "wifi_like"
)

wifi_df.to_csv(
    wifi_output,
    index=False
)

print("\nSaved:")
print(wifi_output)


print("\n" + "=" * 70)
print("SPECTRAL FEATURE EXTRACTION COMPLETE")
print("=" * 70)