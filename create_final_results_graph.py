import matplotlib.pyplot as plt

# RFRecon final AI results
labels = [
    "Hybrid\nValidation",
    "Unseen Real\nWi-Fi Test",
    "Real Capture\nWindow Agreement"
]

values = [
    99.38,
    100.00,
    100.00
]

plt.figure(figsize=(10, 6))

bars = plt.bar(labels, values)

plt.ylim(0, 105)
plt.ylabel("Performance (%)")
plt.title("RFRecon - Final AI Classification Performance",
          fontsize=16, fontweight="bold")

plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    "RFRecon_final_AI_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("==============================================")
print("RFRecon FINAL AI RESULTS GRAPH CREATED")
print("==============================================")
print("Saved as: RFRecon_final_AI_results.png")