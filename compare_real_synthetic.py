import pandas as pd

REAL_FILE = "real_wifi_dataset.csv"
SYNTHETIC_FILE = "robust_training_dataset.csv"

features = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio"
]

real = pd.read_csv(REAL_FILE)
synthetic = pd.read_csv(SYNTHETIC_FILE)

# Label mapping:
# 0 = Noise
# 1 = WiFi-like
# 2 = Bluetooth-like
# 3 = Zigbee-like
# 4 = Cellular-like

synthetic_wifi = synthetic[
    synthetic["label"] == 1
]

print("==============================================")
print("RFRecon REAL vs SYNTHETIC Wi-Fi COMPARISON")
print("==============================================")

print()
print("REAL Wi-Fi samples     :", len(real))
print("SYNTHETIC Wi-Fi samples:", len(synthetic_wifi))

print()
print(f"{'Feature':35s} {'Real Mean':>12s} {'Synthetic Mean':>16s}")
print("-" * 68)

for feature in features:

    real_mean = real[feature].mean()
    synthetic_mean = synthetic_wifi[feature].mean()

    print(
        f"{feature:35s} "
        f"{real_mean:12.6f} "
        f"{synthetic_mean:16.6f}"
    )

print()
print("==============================================")
print("REAL Wi-Fi FEATURE RANGES")
print("==============================================")

for feature in features:

    minimum = real[feature].min()
    maximum = real[feature].max()

    print(
        f"{feature:35s} "
        f"min={minimum:.6f}  max={maximum:.6f}"
    )