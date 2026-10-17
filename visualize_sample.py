import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 2_000_000

# Load one clean signal
signal = np.load("generated_data/tone_0000.npy")

# Time axis
time = np.arange(len(signal)) / SAMPLE_RATE

# FFT
spectrum = np.fft.fftshift(np.fft.fft(signal))
frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(signal), 1 / SAMPLE_RATE)
)

# Plot IQ waveform
plt.figure()
plt.plot(time[:500] * 1e6, signal.real[:500])
plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")
plt.title("RFRecon - I Component")
plt.grid()
plt.show()

# Plot spectrum
plt.figure()
power = 20 * np.log10(np.abs(spectrum) + 1e-12)
plt.plot(frequencies / 1e3, power)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.title("RFRecon - Frequency Spectrum")
plt.grid()
plt.show()

# Plot spectrogram
plt.figure()
plt.specgram(
    signal,
    NFFT=256,
    Fs=SAMPLE_RATE,
    noverlap=128
)
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.title("RFRecon - Spectrogram")
plt.colorbar(label="Power")
plt.show()