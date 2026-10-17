import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# RFRecon FINAL RESULTS COMPARISON
# ============================================================

experiments = [
    "Hybrid\nValidation",
    "Unseen Real\nWi-Fi",
    "Capture 02\n12,616 windows",
    "Capture 03\n4,328 windows",
    "Low-SNR\nStress Test"
]

random_forest = [
    99.38,
    100.00,
    99.95,
    100.00,
    97.84
]

mlp = [
    98.96,
    100.00,
    99.82,
    100.00,
    79.08
]


x = np.arange(len(experiments))

width = 0.35


# ============================================================
# CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(13, 7)
)


rf_bars = plt.bar(
    x - width / 2,
    random_forest,
    width,
    label="Random Forest"
)


mlp_bars = plt.bar(
    x + width / 2,
    mlp,
    width,
    label="MLP Neural Network"
)


# ============================================================
# LABELS
# ============================================================

plt.ylabel(
    "Performance (%)",
    fontsize=12
)

plt.xlabel(
    "Evaluation Experiment",
    fontsize=12
)

plt.title(
    "RFRecon Final AI Model Evaluation",
    fontsize=16
)

plt.xticks(
    x,
    experiments,
    fontsize=10
)

plt.ylim(
    0,
    110
)


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [
    rf_bars,
    mlp_bars
]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 1,

            f"{value:.2f}%",

            ha="center",
            va="bottom",

            fontsize=9
        )


# ============================================================
# GRID + LEGEND
# ============================================================

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.legend(
    loc="lower left"
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

OUTPUT = (
    "RFRecon_FINAL_AI_MODEL_EVALUATION.png"
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("RFRecon FINAL AI MODEL EVALUATION")
print("=" * 70)

print("\nRandom Forest:")

for experiment, value in zip(
    experiments,
    random_forest
):

    print(
        f"{experiment.replace(chr(10), ' ')}: "
        f"{value:.2f}%"
    )


print("\nMLP Neural Network:")

for experiment, value in zip(
    experiments,
    mlp
):

    print(
        f"{experiment.replace(chr(10), ' ')}: "
        f"{value:.2f}%"
    )


print("\nSaved as:")
print(OUTPUT)