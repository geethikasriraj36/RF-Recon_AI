import numpy as np
import matplotlib.pyplot as plt

FS = 20e6
N = 2**20

noise_file = r"real_dataset\real_noise_01.dat"
wifi_file = r"real_dataset\real_wifi_01.dat"


def analyze(file_path, name):

    data = np.fromfile(
        file_path,
        dtype=np.complex64,
        count=N
    )

    window = np.hanning(len(data))

    spectrum = np.fft.fftshift(
        np.fft.fft(data * window)
    )

    power = np.abs(spectrum) ** 2

    power_db = 10 * np.log10(
        power + 1e-12
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(data),
            1 / FS
        )
    )

    peak_index = np.argmax(power)

    peak_frequency = frequencies[peak_index]

    return frequencies, power_db, peak_frequency


print("=" * 75)
print("REAL NOISE vs REAL WIFI SPECTRUM COMPARISON")
print("=" * 75)


noise_freq, noise_power, noise_peak = analyze(
    noise_file,
    "Real Noise"
)

wifi_freq, wifi_power, wifi_peak = analyze(
    wifi_file,
    "Real WiFi"
)


print("\nREAL NOISE")
print("-" * 40)

print(
    f"Peak offset: {noise_peak / 1e6:.6f} MHz"
)

print(
    f"Peak level : {np.max(noise_power):.2f} dB"
)


print("\nREAL WIFI")
print("-" * 40)

print(
    f"Peak offset: {wifi_peak / 1e6:.6f} MHz"
)

print(
    f"Peak level : {np.max(wifi_power):.2f} dB"
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    noise_freq / 1e6,
    noise_power,
    label="Real Noise"
)

plt.plot(
    wifi_freq / 1e6,
    wifi_power,
    label="Real WiFi"
)

plt.xlabel("Frequency Offset (MHz)")
plt.ylabel("Spectral Level (dB)")

plt.title(
    "Real Noise vs Real WiFi Spectrum"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

output = (
    r"real_dataset\real_noise_vs_wifi_spectrum.png"
)

plt.savefig(
    output,
    dpi=200
)

plt.show()

print("\nSaved:")
print(output)

print("\n" + "=" * 75)
print("SPECTRUM COMPARISON COMPLETE")
print("=" * 75)