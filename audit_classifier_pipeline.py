
import os
import pickle
import pandas as pd

print("=" * 65)
print("RFRecon - READ-ONLY CLASSIFIER PIPELINE AUDIT")
print("=" * 65)

files = {
    "Hybrid dataset": "hybrid_training_dataset.csv",
    "Noise features": "real_dataset/real_noise_features.csv",
    "Wi-Fi features": "real_dataset/real_wifi_features_bwtest.csv",
    "Bluetooth features": "real_dataset/real_bluetooth_features_bwtest.csv",
    "RF feature artifact": "rfrecon_hybrid_features.pkl",
    "MLP feature artifact": "rfrecon_mlp_features.pkl",
}

for label, path in files.items():
    print(f"\n--- {label} ---")

    if not os.path.exists(path):
        print("NOT FOUND:", path)
        continue

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())

        if "label" in df.columns:
            print("Label counts:")
            print(df["label"].value_counts().to_dict())

    elif path.endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)

            print("Object type:", type(obj).__name__)

            if isinstance(obj, dict):
                print("Keys:", list(obj.keys()))
            elif isinstance(obj, (list, tuple)):
                print("Contents:", obj)
            else:
                print("Contents:", repr(obj)[:1000])

        except Exception as exc:
            print("Could not inspect artifact:", exc)

print("\nAudit complete. No files were modified.")
