import numpy as np
import matplotlib.pyplot as plt

FILE = r"real_dataset\real_cellular_01.dat"

SAMPLE_RATE = 20e6
CENTER_FREQ = 1.856e9

# Analyze 0.5 seconds
N = int(0.5 * SAMPLE_RATE)

x = np.memmap(FILE, dtype=np.complex64, mode="r")

samples = x[:N]

# ---------------------------------------------------------
# SPECTROGRAM
# ---------------------------------------------------------

FFT_SIZE = 16384
STEP = FFT_SIZE // 2

window = np.hanning(FFT_SIZE)

spectrogram = []

for start in range(0, len(samples) - FFT_SIZE, STEP):

    chunk = samples[start:start + FFT_SIZE]

    spectrum = np.fft.fftshift(
        np.fft.fft(chunk * window)
    )

    power = np.abs(spectrum) ** 2

    power_db = 10 * np.log10(power + 1e-12)

    spectrogram.append(power_db)

spectrogram = np.array(spectrogram)

frequency = np.fft.fftshift(
    np.fft.fftfreq(
        FFT_SIZE,
        d=1 / SAMPLE_RATE
    )
)

frequency_mhz = (
    CENTER_FREQ + frequency
) / 1e6

time = (
    np.arange(spectrogram.shape[0])
    * STEP
    / SAMPLE_RATE
)

# ---------------------------------------------------------
# PLOT WATERFALL
# ---------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.pcolormesh(
    frequency_mhz,
    time,
    spectrogram,
    shading="auto"
)

plt.xlim(1846, 1866)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Time (seconds)")

plt.title(
    "RFRecon - Real Cellular Candidate Time-Frequency Analysis"
)

plt.colorbar(
    label="Spectral Level (dB)"
)

plt.tight_layout()

plt.savefig(
    "real_cellular_persistence_waterfall.png",
    dpi=200
)

plt.show()

print("=" * 70)
print("RFRecon - CELLULAR CANDIDATE PERSISTENCE ANALYSIS")
print("=" * 70)

print(f"Analyzed duration : {N / SAMPLE_RATE:.3f} seconds")
print(f"FFT size          : {FFT_SIZE}")
print(f"Frequency span    : 1846 - 1866 MHz")

print()
print("Saved:")
print("real_cellular_persistence_waterfall.png")