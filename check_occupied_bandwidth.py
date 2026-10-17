import numpy as np

FILE = r"real_dataset\real_wifi_01.dat"
SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096

data = np.memmap(FILE, dtype=np.complex64, mode="r")
x = np.asarray(data[:WINDOW_SIZE])

window = np.hanning(len(x))
spectrum = np.fft.fftshift(np.fft.fft(x * window))
psd = np.abs(spectrum) ** 2

frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(x), d=1 / SAMPLE_RATE)
)

total = np.sum(psd)
cumulative = np.cumsum(psd) / total

low_idx = np.searchsorted(cumulative, 0.025)
high_idx = np.searchsorted(cumulative, 0.975)

bandwidth = frequencies[high_idx] - frequencies[low_idx]

print("=" * 65)
print("RFRecon - OCCUPIED BANDWIDTH DIAGNOSTIC")
print("=" * 65)
print(f"Frequency bins: {len(frequencies)}")
print(f"Total spectral power: {total:.6g}")
print(f"Cumulative power at lower index: {cumulative[low_idx]:.6f}")
print(f"Cumulative power at upper index: {cumulative[high_idx]:.6f}")
print(f"Lower frequency: {frequencies[low_idx] / 1e6:.6f} MHz")
print(f"Upper frequency: {frequencies[high_idx] / 1e6:.6f} MHz")
print(f"Occupied bandwidth: {bandwidth / 1e6:.6f} MHz")
print(f"Normalized bandwidth: {bandwidth / SAMPLE_RATE:.6f}")
print("\nNo datasets or models were modified.")
