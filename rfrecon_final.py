print("=" * 65)
print("RFRecon - FINAL RF RECONNAISSANCE RESULT")
print("=" * 65)

# ------------------------------------------------------------
# Real PlutoSDR RF measurements
# ------------------------------------------------------------

frequency = 2.405005          # GHz
relative_power = -23.16       # dB
bandwidth = 18.418            # MHz

# ------------------------------------------------------------
# AI classification result
# ------------------------------------------------------------

classification = "WiFi-like"
confidence = 94.80
window_agreement = 99.95

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\nRF ACTIVITY DETECTED")
print("-" * 65)

print(f"Frequency          : {frequency:.6f} GHz")
print(f"Relative Power     : {relative_power:.2f} dB")
print(f"Occupied Bandwidth : {bandwidth:.3f} MHz")

print("\nAI CLASSIFICATION")
print("-" * 65)

print(f"Classification      : {classification}")
print(f"Confidence          : {confidence:.2f}%")
print(f"Window Agreement    : {window_agreement:.2f}%")

print("\n" + "=" * 65)
print("FINAL RFRecon DECISION")
print("=" * 65)

print(f"\nRF Activity Detected")
print(f"Frequency     : {frequency:.6f} GHz")
print(f"Power         : {relative_power:.2f} dB (relative)")
print(f"Bandwidth     : {bandwidth:.3f} MHz")
print(f"Classification: {classification}")
print(f"Confidence    : {confidence:.2f}%")

print("\n" + "=" * 65)