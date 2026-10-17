
import numpy as np
from scipy.signal import periodogram

FEATURE_NAMES = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
]


def extract_features(signal, sample_rate):
    """Extract consistent RF features from one complex IQ window."""

    x = np.asarray(signal, dtype=np.complex64).ravel()

    if x.size < 2:
        raise ValueError("Signal must contain at least two samples.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite IQ samples.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    magnitude = np.abs(x)
    power = magnitude ** 2

    mean_power = float(np.mean(power))
    rms = float(np.sqrt(mean_power))
    peak_amplitude = float(np.max(magnitude))
    variance = float(np.var(magnitude))

    # Consistent Hann-windowed, two-sided spectrum.
    frequencies, psd = periodogram(
        x,
        fs=sample_rate,
        window="hann",
        detrend=False,
        return_onesided=False,
        scaling="spectrum",
    )

    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(psd)
    psd = np.maximum(psd, 0.0)

    total = float(np.sum(psd))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Signal has invalid or zero spectral power.")

    weights = psd / total
    abs_freq = np.abs(frequencies)

    peak_frequency_normalized = float(
        frequencies[np.argmax(psd)] / (sample_rate / 2)
    )

    centroid = float(np.sum(abs_freq * weights))
    centroid_normalized = centroid / (sample_rate / 2)

    bandwidth = float(
        np.sqrt(np.sum((abs_freq - centroid) ** 2 * weights))
    )
    bandwidth_normalized = bandwidth / (sample_rate / 2)

    # 95% occupied bandwidth: cumulative power in frequency order.
    cumulative = np.cumsum(weights)
    low_idx = int(np.searchsorted(cumulative, 0.025))
    high_idx = int(np.searchsorted(cumulative, 0.975))

    low_idx = min(low_idx, len(frequencies) - 1)
    high_idx = min(high_idx, len(frequencies) - 1)

    occupied_bandwidth = max(
        0.0, float(frequencies[high_idx] - frequencies[low_idx])
    )
    occupied_bandwidth_normalized = occupied_bandwidth / sample_rate

    eps = np.finfo(np.float64).tiny
    spectral_flatness = float(
        np.exp(np.mean(np.log(psd + eps))) /
        (np.mean(psd) + eps)
    )

    peak_to_average_ratio = float(
        np.max(psd) / (np.mean(psd) + eps)
    )

    result = [
        mean_power,
        rms,
        peak_amplitude,
        variance,
        peak_frequency_normalized,
        centroid_normalized,
        bandwidth_normalized,
        occupied_bandwidth_normalized,
        spectral_flatness,
        peak_to_average_ratio,
    ]

    if not np.all(np.isfinite(result)):
        raise ValueError("Feature extraction produced non-finite values.")

    return dict(zip(FEATURE_NAMES, result))


if __name__ == "__main__":
    # Smoke test only; does not load or overwrite project data.
    rng = np.random.default_rng(42)
    test_iq = (
        rng.standard_normal(4096) +
        1j * rng.standard_normal(4096)
    ).astype(np.complex64)

    print("Smoke test at 2 MS/s:")
    print(extract_features(test_iq, 2_000_000))

    print("\nSmoke test at 20 MS/s:")
    print(extract_features(test_iq, 20_000_000))

    print("\nFeature extractor test passed.")
