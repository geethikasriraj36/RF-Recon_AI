
import pandas as pd

results = [
    {
        "Capture": "Real Noise 01",
        "Expected_Class": "noise",
        "RF_Prediction": "wifi_like",
        "RF_Confidence_Percent": 96.28,
        "MLP_Prediction": "wifi_like",
        "MLP_Confidence_Percent": 99.45,
        "Model_Agreement_Percent": 99.73,
        "Assessment": "Incorrect class prediction"
    },
    {
        "Capture": "Real WiFi 01",
        "Expected_Class": "wifi_like",
        "RF_Prediction": "wifi_like",
        "RF_Confidence_Percent": 96.36,
        "MLP_Prediction": "wifi_like",
        "MLP_Confidence_Percent": 99.45,
        "Model_Agreement_Percent": 98.86,
        "Assessment": "Expected class predicted"
    },
    {
        "Capture": "Real Bluetooth 01",
        "Expected_Class": "bluetooth_like",
        "RF_Prediction": "wifi_like",
        "RF_Confidence_Percent": 85.62,
        "MLP_Prediction": "wifi_like",
        "MLP_Confidence_Percent": 98.79,
        "Model_Agreement_Percent": 94.89,
        "Assessment": "Incorrect class prediction"
    }
]

df = pd.DataFrame(results)

output = r"real_dataset\real_capture_evaluation_summary.csv"
df.to_csv(output, index=False)

print("=" * 70)
print("REAL CAPTURE EVALUATION SUMMARY")
print("=" * 70)
print(df.to_string(index=False))
print("\nSaved:", output)
