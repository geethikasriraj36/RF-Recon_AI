import numpy as np

x = np.load(r"real_pluto_data\real_pluto_0000.npy")

# Remove DC
x = x - np.mean(x)

fs = 2_000_000

# FFT
spectrum = np.fft.fftshift(np.fft.fft(x))
power = np.abs(spectrum) ** 2

freqs = np.fft.fftshift(
    np.fft.fftfreq(len(x), 1 / fs)
)

# Ignore the center ±10 kHz around DC
mask = np.abs(freqs) > 10_000

filtered_power = power.copy()
filtered_power[~mask] = 0

# Strongest non-DC peak
peak_index = np.argmax(filtered_power)

peak_frequency = freqs[peak_index]
peak_power = power[peak_index]

print("\n===== REAL PLUTO RF PEAK =====")
print(f"Peak frequency offset: {peak_frequency/1000:.2f} kHz")
print(f"Peak frequency offset: {peak_frequency/1e6:.6f} MHz")
print(f"Peak power: {10*np.log10(peak_power + 1e-12):.2f} dB")
print(f"Absolute RF frequency: {(2400e6 + peak_frequency)/1e6:.6f} MHz")