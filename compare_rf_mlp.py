import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# RFRecon - Random Forest vs MLP Comparison
# ============================================================

metrics = [
    "Hybrid Validation",
    "Unseen Real Wi-Fi",
    "Real PlutoSDR Capture",
    "Low-SNR Stress Test"
]

random_forest = [
    99.38,
    100.00,
    99.95,
    97.84
]

mlp = [
    98.96,
    100.00,
    99.82,
    79.08
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

bars1 = plt.bar(
    x - width / 2,
    random_forest,
    width,
    label="Random Forest"
)

bars2 = plt.bar(
    x + width / 2,
    mlp,
    width,
    label="MLP Neural Network"
)

plt.ylabel("Performance (%)")
plt.title("RFRecon: Random Forest vs MLP Neural Network")

plt.xticks(
    x,
    metrics,
    rotation=15
)

plt.ylim(0, 110)

plt.legend()

# Add values above bars
for bars in [bars1, bars2]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "RFRecon_RF_vs_MLP_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("RFRecon RF vs MLP Comparison")
print("=" * 60)

print("\nRandom Forest:")
for metric, value in zip(metrics, random_forest):
    print(f"{metric}: {value:.2f}%")

print("\nMLP Neural Network:")
for metric, value in zip(metrics, mlp):
    print(f"{metric}: {value:.2f}%")

print("\nComparison graph saved as:")
print("RFRecon_RF_vs_MLP_comparison.png")