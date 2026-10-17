import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

FILES = {
    "Synthetic WiFi": ("hybrid_training_dataset.csv", "wifi_like"),
    "Real WiFi": ("real_dataset/real_wifi_features.csv", None),
    "Real noise": ("real_dataset/real_noise_features.csv", None),
}

def load_features(path, label=None):
    df = pd.read_csv(path)

    if label is not None:
        if "label" not in df.columns:
            raise ValueError(f"No label column in {path}")
        df = df[df["label"] == label]

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{path} missing features: {missing}")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return df[FEATURES]

datasets = {}

for name, (path, label) in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = load_features(path, label)
    if df.empty:
        raise ValueError(f"No usable rows for {name}")

    datasets[name] = df
    print(f"{name}: {len(df):,} windows")

# Plot empirical distributions with common x-axes per feature.
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

rng = np.random.default_rng(42)

for ax, feature in zip(axes, FEATURES):
    all_values = []

    for df in datasets.values():
        values = df[feature].to_numpy()
        all_values.extend(values[np.isfinite(values)])

    low, high = np.quantile(all_values, [0.01, 0.99])
    if high <= low:
        low, high = min(all_values), max(all_values) + 1e-9

    bins = np.linspace(low, high, 50)

    for name, df in datasets.items():
        values = df[feature].to_numpy()
        values = values[np.isfinite(values)]

        if len(values) > 5000:
            values = rng.choice(values, size=5000, replace=False)

        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name
        )

    ax.set_title(feature)
    ax.set_xlim(low, high)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

fig.suptitle("RFRecon: Synthetic WiFi vs Real WiFi and Noise", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out = "real_dataset/synthetic_real_feature_distributions.png"
fig.savefig(out, dpi=160)
plt.close(fig)

print(f"\nSaved plot: {out}")
print("No model or source CSV was modified.")
