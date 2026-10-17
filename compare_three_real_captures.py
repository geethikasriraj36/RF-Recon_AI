
import subprocess
import sys

captures = [
    r"real_dataset\real_noise_01.dat",
    r"real_dataset\real_wifi_01.dat",
    r"real_dataset\real_bluetooth_01.dat",
]

print("=" * 75)
print("RFRecon - THREE REAL CAPTURE EVALUATION")
print("=" * 75)

for capture in captures:
    print(f"\nCapture: {capture}")
    print("-" * 75)

    result = subprocess.run(
        [sys.executable, "rfrecon_random_classifier.py"],
        input=capture + "\n",
        text=True,
        capture_output=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print("The classifier exited with an error:")
        print(result.stderr)
        break
