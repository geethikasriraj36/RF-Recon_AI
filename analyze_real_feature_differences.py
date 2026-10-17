import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio",
]

FILES = {
    "Noise": r"real_dataset\real_noise_features.csv",
    "WiFi": r"real_dataset\real_wifi_features.csv",
    "Bluetooth-intended": r"real_dataset\real_bluetooth_features.csv",
}

OUTPUT_DIR = "real_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("RFRecon - REAL CAPTURE FEATURE DIFFERENCE ANALYSIS")
print("=" * 70)

datasets = {}

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"SKIPPED: {name} — file not found: {path}")
        continue

    df = pd.read_csv(path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"SKIPPED: {name} — missing features: {missing}")
        continue

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    if df.empty:
        print(f"SKIPPED: {name} — no valid rows")
        continue

    datasets[name] = df[FEATURES]

    print(f"\n{name}")
    print(f"  Valid windows: {len(df):,}")

if len(datasets) < 2:
    raise SystemExit("\nNeed at least two valid feature CSVs to compare.")

# Compare feature means and standard deviations
summary_rows = []

for name, df in datasets.items():
    for feature in FEATURES:
        summary_rows.append({
            "capture": name,
            "feature": feature,
            "mean": df[feature].mean(),
            "std": df[feature].std(),
            "median": df[feature].median(),
            "q25": df[feature].quantile(0.25),
            "q75": df[feature].quantile(0.75),
        })

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "real_feature_summary.csv")
summary.to_csv(summary_path, index=False)

# Standardized mean differences using pooled standard deviation
difference_rows = []
names = list(datasets.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name_a, name_b = names[i], names[j]
        a, b = datasets[name_a], datasets[name_b]

        for feature in FEATURES:
            mean_a, mean_b = a[feature].mean(), b[feature].mean()
            std_a, std_b = a[feature].std(), b[feature].std()

            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

            if pooled_std > 1e-12:
                effect = (mean_a - mean_b) / pooled_std
            else:
                effect = 0.0

            difference_rows.append({
                "capture_A": name_a,
                "capture_B": name_b,
                "feature": feature,
                "mean_A": mean_a,
                "mean_B": mean_b,
                "absolute_mean_difference": abs(mean_a - mean_b),
                "standardized_mean_difference": effect,
                "absolute_effect_size": abs(effect),
            })

differences = pd.DataFrame(difference_rows)
differences = differences.sort_values(
    ["capture_A", "capture_B", "absolute_effect_size"],
    ascending=[True, True, False]
)

difference_path = os.path.join(OUTPUT_DIR, "real_feature_differences.csv")
differences.to_csv(difference_path, index=False)

print("\nTOP 5 FEATURES FOR EACH CAPTURE PAIR")
print("-" * 70)

for (name_a, name_b), group in differences.groupby(["capture_A", "capture_B"], sort=False):
    print(f"\n{name_a} vs {name_b}")
    for _, row in group.head(5).iterrows():
        print(
            f"  {row['feature']:<35} "
            f"effect={row['standardized_mean_difference']:+.4f}"
        )

# Plot distributions. Subsampling keeps plotting responsive.
rng = np.random.default_rng(42)
plot_rows = []

for feature in FEATURES:
    for name, df in datasets.items():
        values = df[feature].to_numpy()
        if len(values) > 3000:
            values = rng.choice(values, size=3000, replace=False)

        for value in values:
            plot_rows.append({
                "feature": feature,
                "capture": name,
                "value": value,
            })

plot_df = pd.DataFrame(plot_rows)

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for ax, feature in zip(axes, FEATURES):
    groups = [
        datasets[name][feature].dropna().to_numpy()
        for name in datasets
    ]

    ax.boxplot(groups, tick_labels=list(datasets.keys()), showfliers=False)
    ax.set_title(feature)
    ax.set_ylabel("Feature value")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "RFRecon: Feature Distributions Across Real Captures",
    fontsize=15
)
fig.tight_layout(rect=[0, 0, 1, 0.98])

plot_path = os.path.join(OUTPUT_DIR, "real_feature_comparison.png")
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Feature summary:    {summary_path}")
print(f"Feature differences:{difference_path}")
print(f"Comparison plot:    {plot_path}")
print("\nNo models were trained or overwritten.")