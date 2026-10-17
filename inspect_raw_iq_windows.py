
from pathlib import Path
import numpy as np

CAPTURE = Path("real_dataset/real_bluetooth_01.dat")
WINDOW_SIZE = 4096

WINDOWS_TO_CHECK = [
    4197, 4956, 6476, 8900, 10963, 11831, 12917
]

if not CAPTURE.exists():
    raise FileNotFoundError(CAPTURE.resolve())

if CAPTURE.stat().st_size % 8 != 0:
    raise ValueError("File size is not divisible by 8 bytes.")

total_samples = CAPTURE.stat().st_size // 8
total_windows = total_samples // WINDOW_SIZE

print("Capture:", CAPTURE)
print("Total samples:", total_samples)
print("Complete windows:", total_windows)

with CAPTURE.open("rb") as file:
    for window_index in WINDOWS_TO_CHECK:
        if not 0 <= window_index < total_windows:
            print("Skipping invalid window:", window_index)
            continue

        file.seek(window_index * WINDOW_SIZE * 8)
        iq = np.fromfile(
            file, dtype=np.complex64, count=WINDOW_SIZE
        )

        i = iq.real
        q = iq.imag
        magnitude = np.abs(iq)

        near_i_limit = np.mean(np.abs(i) >= 0.999)
        near_q_limit = np.mean(np.abs(q) >= 0.999)
        near_unit_magnitude = np.mean(magnitude >= 1.40)

        print(f"\nWindow {window_index}")
        print(f"Mean power: {np.mean(magnitude**2):.6g}")
        print(f"Peak magnitude: {np.max(magnitude):.6g}")
        print(f"I range: [{i.min():.6g}, {i.max():.6g}]")
        print(f"Q range: [{q.min():.6g}, {q.max():.6g}]")
        print(f"I samples near +/-1: {near_i_limit:.3%}")
        print(f"Q samples near +/-1: {near_q_limit:.3%}")
        print(f"Magnitude >= 1.40: {near_unit_magnitude:.3%}")

        print(
            "Unique I values:",
            np.unique(i).size,
            "| Unique Q values:",
            np.unique(q).size,
        )

print("\nRead-only raw-IQ inspection complete.")
