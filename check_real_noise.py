import numpy as np


# ============================================================
# RFRecon - REAL NOISE CAPTURE CHECK
# ============================================================

FILE = "real_dataset/real_noise_01.dat"

SAMPLE_RATE = 20_000_000


print("=" * 60)
print("RFRecon - REAL NOISE CAPTURE CHECK")
print("=" * 60)


# ============================================================
# LOAD FILE
# ============================================================

iq = np.fromfile(
    FILE,
    dtype=np.complex64
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nFile:")
print(FILE)

print("\nNumber of complex samples:")
print(f"{len(iq):,}")

print("\nSample rate:")
print(f"{SAMPLE_RATE / 1e6:.2f} MS/s")

print("\nRecording duration:")
print(
    f"{len(iq) / SAMPLE_RATE:.3f} seconds"
)


# ============================================================
# BASIC SIGNAL STATISTICS
# ============================================================

amplitude = np.abs(iq)

power = amplitude ** 2


print("\n" + "-" * 60)
print("SIGNAL STATISTICS")
print("-" * 60)

print(
    f"\nMean amplitude : "
    f"{np.mean(amplitude):.6f}"
)

print(
    f"RMS amplitude  : "
    f"{np.sqrt(np.mean(power)):.6f}"
)

print(
    f"Peak amplitude : "
    f"{np.max(amplitude):.6f}"
)

print(
    f"Mean power     : "
    f"{np.mean(power):.8f}"
)

print(
    f"Power variance : "
    f"{np.var(power):.8f}"
)


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

nan_count = np.sum(
    np.isnan(iq)
)

inf_count = np.sum(
    np.isinf(iq)
)


print("\n" + "-" * 60)
print("DATA VALIDITY")
print("-" * 60)

print(
    f"\nNaN samples : {nan_count}"
)

print(
    f"Inf samples : {inf_count}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)

if len(iq) > 0 and nan_count == 0 and inf_count == 0:

    print(
        "REAL NOISE CAPTURE CHECK: PASSED"
    )

    print(
        "\nThe capture is readable and "
        "contains valid complex64 I/Q data."
    )

else:

    print(
        "REAL NOISE CAPTURE CHECK: FAILED"
    )

print("=" * 60)n