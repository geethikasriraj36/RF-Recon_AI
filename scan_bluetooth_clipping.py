
from pathlib import Path
import numpy as np

CAPTURE = Path("real_dataset/real_bluetooth_01.dat")
WINDOW_SIZE = 4096
BATCH_WINDOWS = 256

total_samples = CAPTURE.stat().st_size // 8
total_windows = total_samples // WINDOW_SIZE

near_limit_windows = 0
severe_windows = 0
max_near_limit_fraction = 0.0
checked = 0

with CAPTURE.open("rb") as file:
    for start in range(0, total_windows, BATCH_WINDOWS):
        count = min(BATCH_WINDOWS, total_windows - start)
        iq = np.fromfile(
            file, dtype=np.complex64,
            count=count * WINDOW_SIZE
        )

        if iq.size != count * WINDOW_SIZE:
            break

        i = iq.real.reshape(count, WINDOW_SIZE)
        q = iq.imag.reshape(count, WINDOW_SIZE)

        i_fraction = np.mean(np.abs(i) >= 0.999, axis=1)
        q_fraction = np.mean(np.abs(q) >= 0.999, axis=1)
        fraction = np.maximum(i_fraction, q_fraction)

        near_limit_windows += np.sum(fraction > 0.01)
        severe_windows += np.sum(fraction > 0.10)
        max_near_limit_fraction = max(
            max_near_limit_fraction, float(fraction.max())
        )
        checked += count

print("Capture:", CAPTURE)
print("Windows checked:", checked)
print(
    "Windows with >1% of I or Q samples near limits:",
    near_limit_windows,
    f"({100 * near_limit_windows / checked:.2f}%)"
)
print(
    "Windows with >10% of I or Q samples near limits:",
    severe_windows,
    f"({100 * severe_windows / checked:.2f}%)"
)
print(
    "Maximum near-limit fraction in either component:",
    f"{100 * max_near_limit_fraction:.2f}%"
)
