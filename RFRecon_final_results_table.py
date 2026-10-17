import pandas as pd


# ============================================================
# RFRecon FINAL RESULTS TABLE
# ============================================================

results = [

    {
        "Experiment": "Hybrid Validation",
        "Dataset": "Synthetic + Real PlutoSDR",
        "Samples/Windows": 480,
        "Random Forest (%)": 99.38,
        "MLP (%)": 98.96
    },

    {
        "Experiment": "Unseen Real Wi-Fi",
        "Dataset": "100 unseen real PlutoSDR samples",
        "Samples/Windows": 100,
        "Random Forest (%)": 100.00,
        "MLP (%)": 100.00
    },

    {
        "Experiment": "Real PlutoSDR Capture 02",
        "Dataset": "Real PlutoSDR Wi-Fi capture",
        "Samples/Windows": 12616,
        "Random Forest (%)": 99.95,
        "MLP (%)": 99.82
    },

    {
        "Experiment": "Real PlutoSDR Capture 03",
        "Dataset": "Real PlutoSDR Wi-Fi capture",
        "Samples/Windows": 4328,
        "Random Forest (%)": 100.00,
        "MLP (%)": 100.00
    },

    {
        "Experiment": "Low-SNR Stress Test",
        "Dataset": "Synthetic low-SNR RF",
        "Samples/Windows": 2500,
        "Random Forest (%)": 97.84,
        "MLP (%)": 79.08
    }

]


df = pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("RFRecon FINAL RESULTS TABLE")
print("=" * 80)

print()

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    "RFRecon_final_results_table.csv",
    index=False
)


print(
    "\nSaved as:"
)

print(
    "RFRecon_final_results_table.csv"
)