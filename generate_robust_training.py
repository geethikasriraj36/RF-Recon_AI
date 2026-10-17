import numpy as np
import os


# ============================================================
# RFRecon - Improved Synthetic RF Dataset Generator
# ============================================================

SAMPLE_RATE = 2_000_000
NUM_SAMPLES = 4096
NUM_EXAMPLES = 500

OUTPUT_FOLDER = "robust_training_data"

rng = np.random.default_rng(24680)

# ============================================================
# Utility Functions
# ============================================================

def normalize_power(signal, target_power=1.0):
    """Normalize signal to a common average power."""

    power = np.mean(np.abs(signal) ** 2)

    if power < 1e-12:
        return signal

    return signal * np.sqrt(target_power / power)


def add_awgn(signal, snr_db):
    """Add complex Gaussian noise at a specified SNR."""

    signal_power = np.mean(np.abs(signal) ** 2)

    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = (
        rng.normal(0, np.sqrt(noise_power / 2), len(signal))
        + 1j * rng.normal(0, np.sqrt(noise_power / 2), len(signal))
    )

    return signal + noise


def apply_frequency_offset(signal, frequency):

    t = np.arange(len(signal)) / SAMPLE_RATE

    return signal * np.exp(
        1j * 2 * np.pi * frequency * t
    )


def apply_random_burst(signal, duty_cycle):

    n = len(signal)

    burst_length = int(n * duty_cycle)

    start = rng.integers(
        0,
        max(1, n - burst_length)
    )

    output = np.zeros_like(signal)

    output[start:start + burst_length] = signal[
        start:start + burst_length
    ]

    return output


# ============================================================
# CLASS 0: NOISE
# ============================================================

def generate_noise():

    noise_level = rng.uniform(0.7, 1.0)

    signal = (
        rng.normal(0, noise_level, NUM_SAMPLES)
        + 1j * rng.normal(0, noise_level, NUM_SAMPLES)
    )

    return normalize_power(signal)


# ============================================================
# CLASS 1: WIFI-LIKE
#
# OFDM-inspired multicarrier burst
# ============================================================

def generate_wifi_like():

    n = NUM_SAMPLES

    signal = np.zeros(
        n,
        dtype=complex
    )

    # Occupied bandwidth approximately 1 MHz
    num_subcarriers = rng.integers(32, 65)

    frequencies = np.linspace(
        -450_000,
        450_000,
        num_subcarriers
    )

    # Each subcarrier receives a random QPSK symbol
    for frequency in frequencies:

        symbol = rng.choice(
            [1, -1, 1j, -1j]
        )

        t = np.arange(n) / SAMPLE_RATE

        signal += (
            symbol *
            np.exp(
                1j * 2 * np.pi * frequency * t
            )
        )

    signal /= np.sqrt(num_subcarriers) * 3.0

    # Make it burst-like
    signal = apply_random_burst(
        signal,
        rng.uniform(0.45, 0.8)
    )

    # Random frequency offset
    signal = apply_frequency_offset(
        signal,
        rng.uniform(-250_000, 250_000)
    )

    signal = normalize_power(signal)

    # Moderate noise
    signal = add_awgn(
        signal,
        rng.uniform(0, 18)
    )

    return normalize_power(signal)


# ============================================================
# CLASS 2: BLUETOOTH-LIKE
#
# GFSK-inspired narrowband frequency modulation
# with frequency hopping
# ============================================================

def generate_bluetooth_like():

    n = NUM_SAMPLES

    # Binary symbols
    bits = rng.choice(
        [-1, 1],
        size=n
    )

    # Smooth the frequency changes
    kernel_size = rng.integers(8, 20)

    kernel = np.ones(kernel_size) / kernel_size

    smooth_bits = np.convolve(
        bits,
        kernel,
        mode="same"
    )

    # Narrow frequency deviation
    deviation = rng.uniform(
        60_000,
        120_000
    )

    instantaneous_frequency = (
        smooth_bits * deviation
    )

    phase = np.cumsum(
        2 * np.pi *
        instantaneous_frequency /
        SAMPLE_RATE
    )

    signal = np.exp(
        1j * phase
    )

    # Simulate frequency hopping
    hop_offset = rng.uniform(
        -500_000,
        500_000
    )

    signal = apply_frequency_offset(
        signal,
        hop_offset
    )

    # Make signal bursty
    signal = apply_random_burst(
        signal,
        rng.uniform(0.2, 0.5)
    )

    signal = normalize_power(signal)

    signal = add_awgn(
        signal,
        rng.uniform(0, 18)
    )

    return normalize_power(signal)


# ============================================================
# CLASS 3: ZIGBEE-LIKE
#
# OQPSK / chip-spreading inspired signal
# ============================================================

def generate_zigbee_like():

    n = NUM_SAMPLES

    # Generate chip sequence
    chips = rng.choice(
        [-1, 1],
        size=n // 8 + 2
    )

    # Spread each chip over multiple samples
    spread = np.repeat(
        chips,
        8
    )

    spread = np.resize(
        spread,
        n
    )

    # I branch
    i_signal = spread

    # Offset Q branch
    q_signal = np.roll(
        spread,
        4
    )

    signal = (
        i_signal +
        1j * q_signal
    ) / np.sqrt(2)

    # Narrower occupied band than WiFi-like
    signal = apply_frequency_offset(
        signal,
        rng.uniform(
            -400_000,
            400_000
        )
    )

    # Burst behavior
    signal = apply_random_burst(
        signal,
        rng.uniform(0.3, 0.65)
    )

    signal = normalize_power(signal)

    signal = add_awgn(
        signal,
        rng.uniform(0, 18)
    )

    return normalize_power(signal)


# ============================================================
# CLASS 4: CELLULAR-LIKE
#
# Wider-band OFDM-inspired signal
# ============================================================

def generate_cellular_like():

    n = NUM_SAMPLES

    signal = np.zeros(
        n,
        dtype=complex
    )

    # Wider bandwidth than WiFi-like
    num_subcarriers = rng.integers(
        80,
        140
    )

    frequencies = np.linspace(
        -800_000,
        800_000,
        num_subcarriers
    )

    t = np.arange(n) / SAMPLE_RATE

    for frequency in frequencies:

        symbol = rng.choice(
            [1, -1, 1j, -1j]
        )

        signal += (
            symbol *
            np.exp(
                1j * 2 * np.pi *
                frequency * t
            )
        )

    signal /= np.sqrt(
        num_subcarriers
    )

    # Longer burst than Bluetooth/Zigbee
    signal = apply_random_burst(
        signal,
        rng.uniform(0.6, 0.95)
    )

    signal = apply_frequency_offset(
        signal,
        rng.uniform(
            -150_000,
            150_000
        )
    )

    signal = normalize_power(signal)

    signal = add_awgn(
        signal,
        rng.uniform(0, 18)
    )

    return normalize_power(signal)


# ============================================================
# DATASET STRUCTURE
# ============================================================

classes = {
    "noise": generate_noise,
    "wifi_like": generate_wifi_like,
    "bluetooth_like": generate_bluetooth_like,
    "zigbee_like": generate_zigbee_like,
    "cellular_like": generate_cellular_like
}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# GENERATE DATASET
# ============================================================

print("=" * 65)
print("RFRecon - Improved Synthetic RF Dataset Generator")
print("=" * 65)

print(
    "\nSample rate:",
    SAMPLE_RATE / 1e6,
    "MHz"
)

print(
    "Samples per example:",
    NUM_SAMPLES
)

print(
    "Examples per class:",
    NUM_EXAMPLES
)

print("\nGenerating dataset...\n")


for class_name, generator in classes.items():

    class_folder = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    os.makedirs(
        class_folder,
        exist_ok=True
    )

    print(
        f"Generating {class_name}..."
    )

    for i in range(NUM_EXAMPLES):

        signal = generator()

        signal = signal.astype(
            np.complex64
        )

        filename = os.path.join(
            class_folder,
            f"{class_name}_{i:04d}.npy"
        )

        np.save(
            filename,
            signal
        )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("DATASET GENERATION COMPLETE")
print("=" * 65)

print("\nClasses:")

for class_name in classes:
    print(
        f"  {class_name}: {NUM_EXAMPLES}"
    )

print(
    "\nTotal examples:",
    NUM_EXAMPLES * len(classes)
)

print(
    "\nDataset location:",
    OUTPUT_FOLDER
)