import pandas as pd

data = {
    "Experiment": [
        "Robust Synthetic Validation",
        "Hybrid Model Validation",
        "Unseen Real PlutoSDR Wi-Fi",
        "Real PlutoSDR Capture"
    ],
    "Dataset": [
        "Synthetic RF",
        "Synthetic + Real RF",
        "Unseen Real Wi-Fi",
        "Live/Recorded PlutoSDR"
    ],
    "Samples": [
        500,
        480,
        100,
        4328
    ],
    "Result": [
        "97.84%",
        "99.38%",
        "100.00%",
        "100.00% window agreement"
    ],
    "Average Confidence": [
        "—",
        "—",
        "—",
        "94.50%"
    ]
}

df = pd.DataFrame(data)

print("\n==============================================================")
print("RFRecon - FINAL EXPERIMENT COMPARISON")
print("==============================================================\n")

print(df.to_string(index=False))

df.to_csv("RFRecon_final_experiment_results.csv", index=False)

print("\n==============================================================")
print("Saved as: RFRecon_final_experiment_results.csv")
print("==============================================================")