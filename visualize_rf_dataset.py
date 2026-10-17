import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

SAMPLE_RATE = 2_000_000

classes = [
    "noise",
    "wifi_like",
    "bluetooth_like",
    "zigbee_like",
    "cellular_like"
]

fig, axes = plt.subplots(
    5, 2,
    figsize=(14, 16)
)

for row, class_name in enumerate(classes):

    filename = (
        f"realistic_rf_data/{class_name}/"
        f"{class_name}_0000.npy"
    )

    signal = np.load(filename)

    # ==================================================
    # SPECTRUM
    # ==================================================

    spectrum = np.fft.fftshift(
        np.fft.fft(signal)
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(signal),
            1 / SAMPLE_RATE
        )
    )

    magnitude_db = 20 * np.log10(
        np.abs(spectrum) + 1e-12
    )

    axes[row, 0].plot(
        frequencies / 1e6,
        magnitude_db
    )

    axes[row, 0].set_title(
        f"{class_name} - Spectrum"
    )

    axes[row, 0].set_xlabel(
        "Frequency (MHz)"
    )

    axes[row, 0].set_ylabel(
        "Magnitude (dB)"
    )

    axes[row, 0].set_xlim(-1, 1)

    # ==================================================
    # SPECTROGRAM
    # ==================================================

    f, t, Sxx = spectrogram(
        signal,
        fs=SAMPLE_RATE,
        nperseg=256,
        noverlap=192,
        return_onesided=False
    )

    Sxx = np.fft.fftshift(
        Sxx,
        axes=0
    )

    f = np.fft.fftshift(f)

    Sxx_db = 10 * np.log10(
        np.abs(Sxx) + 1e-12
    )

    axes[row, 1].pcolormesh(
        t,
        f / 1e6,
        Sxx_db,
        shading="auto"
    )

    axes[row, 1].set_title(
        f"{class_name} - Spectrogram"
    )

    axes[row, 1].set_xlabel(
        "Time (s)"
    )

    axes[row, 1].set_ylabel(
        "Frequency (MHz)"
    )

    axes[row, 1].set_ylim(-1, 1)


plt.tight_layout()

plt.savefig(
    "improved_rf_visualization.png",
    dpi=300
)

plt.show()