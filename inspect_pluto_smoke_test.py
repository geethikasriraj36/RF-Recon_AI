
from pathlib import Path
import numpy as np
from scipy.signal import periodogram

CAPTURE = Path("real_dataset/pluto_rx_smoke_test.dat")
SAMPLE_RATE = 30_720_000
CENTER_FREQUENCY = 2_400_000_000

if not CAPTURE.exists():
    raise FileNotFoundError(CAPTURE)

if CAPTURE.stat().st_size % 8:
    raise ValueError("File size is not divisible by complex64 sample size.")

iq = np.fromfile(CAPTURE, dtype=np.complex64)

if iq.size == 0:
    raise ValueError("Capture is empty.")

power = np.abs(iq) ** 2
freq, psd = periodogram(
    iq,
    fs=SAMPLE_RATE,
    window="hann",
    detrend=False,
    return_onesided=False,
    scaling="density",
)
freq = np.fft.fftshift(freq)
psd = np.fft.fftshift(psd)

peak_index = int(np.argmax(psd))
peak_offset = float(freq[peak_index])
peak_power = float(psd[peak_index])
median_psd = float(np.median(psd))

print("Capture:", CAPTURE)
print("Complex samples:", iq.size)
print("Duration (ms):", iq.size / SAMPLE_RATE * 1000)
print("Assumed center frequency (Hz):", CENTER_FREQUENCY)
print("Assumed sample rate (Hz):", SAMPLE_RATE)
print("Mean power:", float(np.mean(power)))
print("Median power:", float(np.median(power)))
print("Peak magnitude:", float(np.max(np.abs(iq))))
print("Peak spectral offset (Hz):", peak_offset)
print("Approximate peak frequency (Hz):", CENTER_FREQUENCY + peak_offset)
print("Peak/median PSD ratio:", peak_power / max(median_psd, 1e-30))
print("Finite samples:", bool(np.isfinite(iq).all()))

print("\nThis is an unlabeled RF snapshot.")
print("Do not assign a protocol label from this output alone.")
