
import numpy as np

file_path = r"real_dataset\real_bluetooth_01.dat"

print("Loading Bluetooth capture...")
data = np.fromfile(file_path, dtype=np.complex64)

sections = np.array_split(data, 10)

print("\nBluetooth Capture: 10-Section Analysis")
print("-" * 65)

for i, section in enumerate(sections, start=1):
    magnitude = np.abs(section)

    mean_power = np.mean(magnitude ** 2)
    p99 = np.percentile(magnitude, 99)
    i_limits = np.mean(np.abs(section.real) >= 0.999) * 100
    q_limits = np.mean(np.abs(section.imag) >= 0.999) * 100

    print(
        f"Section {i}: "
        f"mean_power={mean_power:.6f}, "
        f"p99_magnitude={p99:.4f}, "
        f"I_near_limits={i_limits:.3f}%, "
        f"Q_near_limits={q_limits:.3f}%"
    )
