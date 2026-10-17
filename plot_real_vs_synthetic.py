import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RFRecon - Real vs Synthetic Wi-Fi Feature Comparison
# Actual Feature Means
# ============================================================

features = [
    "Mean Power",
    "RMS",
    "Peak Amplitude",
    "Variance",
    "Peak Frequency",
    "Spectral Centroid",
    "Spectral Bandwidth",
    "Occupied Bandwidth",
    "Spectral Flatness",
    "Peak-to-Average Ratio"
]

real_mean = [
    0.009563,
    0.070853,
    0.217175,
    0.001943,
    0.330964,
    0.410652,
    0.242918,
    0.735787,
    0.411383,
    3.388524
]

synthetic_mean = [
    1.000000,
    1.000000,
    3.153487,
    0.355323,
    -0.001650,
    0.292513,
    0.204814,
    0.519987,
    0.134331,
    53.623538
]

x = np.arange(len(features))
width = 0.36

fig, ax = plt.subplots(figsize=(16, 9))

bars1 = ax.bar(
    x - width / 2,
    real_mean,
    width,
    label="Real PlutoSDR Wi-Fi"
)

bars2 = ax.bar(
    x + width / 2,
    synthetic_mean,
    width,
    label="Synthetic Wi-Fi"
)

ax.set_title(
    "RFRecon - Real vs Synthetic Wi-Fi Feature Means",
    fontsize=20,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Extracted RF Features",
    fontsize=13
)

ax.set_ylabel(
    "Mean Feature Value",
    fontsize=13
)

ax.set_xticks(x)
ax.set_xticklabels(
    features,
    rotation=40,
    ha="right",
    fontsize=10
)

# Log scale makes features with very different magnitudes
# easier to compare.
ax.set_yscale("symlog", linthresh=0.01)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.legend(
    fontsize=11,
    loc="upper right"
)

# Add values above bars
for bar, value in zip(bars1, real_mean):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )

for bar, value in zip(bars2, synthetic_mean):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )

plt.tight_layout()

plt.savefig(
    "real_vs_synthetic_features_actual.png",
    dpi=300,
    bbox_inches="tight"
)

print("=" * 65)
print("RFRecon - REAL VS SYNTHETIC FEATURE COMPARISON")
print("=" * 65)
print()
print("Actual feature mean comparison completed.")
print("Saved as: real_vs_synthetic_features_actual.png")
print()

plt.show()