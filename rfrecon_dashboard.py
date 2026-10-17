import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# RFRecon - FINAL RESULTS DASHBOARD
# ============================================================

frequency = 2.405005
power = -23.16
bandwidth = 18.418
classification = "WiFi-like"
confidence = 95.71
agreement = 99.95

# Load existing spectrum image
spectrum = Image.open("real_wifi_capture_02_spectrum.png")

plt.figure(figsize=(14, 8))

# Main spectrum
plt.imshow(spectrum)
plt.axis("off")

plt.title(
    "RFRecon - Real PlutoSDR RF Reconnaissance\n"
    f"Detected: {classification} | "
    f"Frequency: {frequency:.6f} GHz | "
    f"Confidence: {confidence:.2f}%",
    fontsize=16,
    fontweight="bold"
)

plt.figtext(
    0.5,
    0.04,
    f"Power: {power:.2f} dB relative   |   "
    f"Occupied Bandwidth: {bandwidth:.3f} MHz   |   "
    f"Window Agreement: {agreement:.2f}%",
    ha="center",
    fontsize=12
)

plt.tight_layout()
plt.savefig(
    "RFRecon_FINAL_DASHBOARD.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("=" * 65)
print("RFRecon FINAL DASHBOARD CREATED")
print("=" * 65)
print("Saved as: RFRecon_FINAL_DASHBOARD.png")