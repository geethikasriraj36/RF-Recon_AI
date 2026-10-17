import numpy as np

# Load one real PlutoSDR sample
file = r"real_pluto_data\real_pluto_0000.npy"
x = np.load(file)

# Sampling rate
fs = 2_000_000

# Basic signal quantities
magnitude = np.abs(x)

mean_power = np.mean(magnitude ** 2)
rms = np.sqrt(np.mean(magnitude ** 2))
peak_amplitude = np.max(magnitude)
variance = np.var(magnitude)

# FFT
spectrum = np.fft.fftshift(np.fft.fft(x))
power_spectrum = np.abs(spectrum) ** 2

# Frequency axis
freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))

# Normalize spectrum
total_power = np.sum(power_spectrum)

if total_power > 0:
    normalized_spectrum = power_spectrum / total_power
else:
    normalized_spectrum = power_spectrum

# Peak frequency
peak_frequency = freqs[np.argmax(power_spectrum)]
peak_frequency_normalized = abs(peak_frequency) / (fs / 2)

# Spectral centroid
spectral_centroid = np.sum(
    np.abs(freqs) * normalized_spectrum
)

spectral_centroid_normalized = spectral_centroid / (fs / 2)

# Spectral bandwidth
spectral_bandwidth = np.sqrt(
    np.sum(
        ((np.abs(freqs) - spectral_centroid) ** 2)
        * normalized_spectrum
    )
)

spectral_bandwidth_normalized = spectral_bandwidth / (fs / 2)

# Occupied bandwidth (90% of power)
sorted_indices = np.argsort(np.abs(freqs))
sorted_power = power_spectrum[sorted_indices]

cumulative_power = np.cumsum(sorted_power)

low_index = np.searchsorted(
    cumulative_power,
    0.05 * total_power
)

high_index = np.searchsorted(
    cumulative_power,
    0.95 * total_power
)

low_freq = np.abs(freqs)[sorted_indices[low_index]]
high_freq = np.abs(freqs)[sorted_indices[min(high_index, len(freqs)-1)]]

occupied_bandwidth = high_freq - low_freq
occupied_bandwidth_normalized = occupied_bandwidth / (fs / 2)

# Spectral flatness
epsilon = 1e-12
spectral_flatness = np.exp(
    np.mean(np.log(power_spectrum + epsilon))
) / (
    np.mean(power_spectrum + epsilon)
)

# Peak-to-average ratio
peak_to_average_ratio = (
    peak_amplitude / (rms + epsilon)
)

# Print results
print("\n===== RFRecon REAL PLUTO FEATURES =====")

print(f"mean_power                  : {mean_power:.6f}")
print(f"rms                         : {rms:.6f}")
print(f"peak_amplitude              : {peak_amplitude:.6f}")
print(f"variance                    : {variance:.6f}")
print(f"peak_frequency_normalized   : {peak_frequency_normalized:.6f}")
print(f"spectral_centroid_normalized: {spectral_centroid_normalized:.6f}")
print(f"spectral_bandwidth_normalized: {spectral_bandwidth_normalized:.6f}")
print(f"occupied_bandwidth_normalized: {occupied_bandwidth_normalized:.6f}")
print(f"spectral_flatness            : {spectral_flatness:.6f}")
print(f"peak_to_average_ratio        : {peak_to_average_ratio:.6f}")