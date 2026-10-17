import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN_FILE = Path("hybrid_training_dataset.csv")
REAL_FILES = {
    "Noise": Path("real_dataset/real_noise_features.csv"),
    "Wi-Fi intended": Path("real_dataset/real_wifi_features.csv"),
    "Bluetooth intended": Path("real_dataset/real_bluetooth_features.csv"),
    "Cellular intended": Path("real_dataset/real_cellular_features.csv"),
}

FEATURES = [
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

train = pd.read_csv(TRAIN_FILE)
captures = {
    name: pd.read_csv(path)
    for name, path in REAL_FILES.items()
    if path.exists()
}

fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

for ax, feature in zip(axes, FEATURES):
    groups = []
    names = []

    values = pd.to_numeric(
        train[feature], errors="coerce"
    ).dropna().to_numpy()

    groups.append(values)
    names.append("Training")

    for name, df in captures.items():
        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature], errors="coerce"
        ).dropna().to_numpy()

        # Limit plotting workload without modifying the CSV.
        if len(values) > 3000:
            values = values[::max(1, len(values) // 3000)]

        groups.append(values)
        names.append(name)

    ax.boxplot(groups, tick_labels=names, showfliers=False)
    ax.set_title(feature)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "RFRecon: Training vs Real-Capture Feature Distributions",
    fontsize=16
)
fig.tight_layout()

output = Path("real_dataset/feature_scale_audit.png")
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot: {output}")
print("Diagnostic only: no dataset or model was modified.")
