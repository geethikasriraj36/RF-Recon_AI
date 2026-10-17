import numpy as np
from pathlib import Path

# ==========================================
# RFRecon AI - Realistic Synthetic RF Dataset
# ==========================================

SAMPLE_RATE = 2_000_000
NUM_SAMPLES = 4096
NUM_EXAMPLES = 500

OUTPUT_DIR = Path("generated_data")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


def generate_tone(frequency, amplitude, phase):
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE

    return amplitude * np.exp(
        1j * (2 * np.pi * frequency * t + phase)
    )


def generate_noise(noise_amplitude):
    return (
        rng.normal(0, noise_amplitude, NUM_SAMPLES)
        + 1j * rng.normal(0, noise_amplitude, NUM_SAMPLES)
    )


def generate_noisy_signal(signal, noise_amplitude):
    return signal + generate_noise(noise_amplitude)


# ------------------------------------------
# Generate dataset
# ------------------------------------------

print("Generating RFRecon dataset...")

for i in range(NUM_EXAMPLES):

    # Random frequency: 300 kHz - 900 kHz
    frequency = rng.uniform(300_000, 900_000)

    # Random amplitude
    amplitude = rng.uniform(0.5, 1.0)

    # Random phase
    phase = rng.uniform(0, 2 * np.pi)

    # Random noise level
    noise_level = rng.uniform(0.05, 0.5)

    # -----------------------------
    # Class 1: Clean RF signal
    # -----------------------------

    clean_signal = generate_tone(
        frequency,
        amplitude,
        phase
    )

    np.save(
        OUTPUT_DIR / f"tone_{i:04d}.npy",
        clean_signal.astype(np.complex64)
    )

    # -----------------------------
    # Class 2: RF + noise
    # -----------------------------

    noisy_signal = generate_noisy_signal(
        clean_signal,
        noise_level
    )

    np.save(
        OUTPUT_DIR / f"tone_noise_{i:04d}.npy",
        noisy_signal.astype(np.complex64)
    )

    # -----------------------------
    # Class 3: Noise only
    # -----------------------------

    noise_signal = generate_noise(
        noise_level
    )

    np.save(
        OUTPUT_DIR / f"noise_{i:04d}.npy",
        noise_signal.astype(np.complex64)
    )


print()
print("Dataset generation complete!")

total_files = len(list(OUTPUT_DIR.glob("*.npy")))

print(f"Total files: {total_files}")
print(f"Samples per class: {NUM_EXAMPLES}")
print(f"Sample length: {NUM_SAMPLES}")
print(f"Sample rate: {SAMPLE_RATE / 1e6} MHz")