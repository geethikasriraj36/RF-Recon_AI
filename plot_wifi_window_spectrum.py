import numpy as np
import matplotlib.pyplot as plt

FILE = r"real_dataset\real_wifi_01.dat"
SAMPLE_RATE = 20e6
WINDOW_SIZE = 4096

data = np.memmap(FILE, dtype=np.complex64, mode="r")
x = np.asarray(data[:WINDOW_SIZE])

window = np.hanning(WINDOW_SIZE)
spectrum = np.fft.fftshift(np.fft.fft(x * window))
power = np.abs(spectrum) ** 2

frequencies = np.fft.fftshift(
    np.fft.fftfreq(WINDOW_SIZE, d=1 / SAMPLE_RATE)
)

power_db = 10 * np.log10(power / (np.max(power) + 1e-20) + 1e-20)

plt.figure(figsize=(12, 5))
plt.plot(frequencies / 1e6, power_db)
plt.xlabel("Frequency offset (MHz)")
plt.ylabel("Relative power (dB)")
plt.title("RFRecon - First Wi-Fi Capture Window")
plt.xlim(-SAMPLE_RATE / 2e6, SAMPLE_RATE / 2e6)
plt.ylim(-80, 5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = r"real_dataset\wifi_window_spectrum.png"
plt.savefig(output, dpi=160)
plt.close()

print("Saved:", output)
print("Frequency span: -10 MHz to +10 MHz")
print("No datasets or models were modified.")
