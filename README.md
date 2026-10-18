# RFRecon_AI — AI-Based RF Signal Classification

## Overview

RFRecon_AI is a Software-Defined Radio (SDR) project that analyzes complex in-phase and quadrature (I/Q) samples and extracts signal features for radio-frequency classification.

The project combines machine-learning models, synthetic RF datasets, PlutoSDR capture analysis, feature extraction, stress testing, and evaluation utilities.

The system explores classification of signals belonging to the following categories:

- Bluetooth-like
- Cellular-like
- Noise
- Wi-Fi-like
- Zigbee-like

**Important:** Classification results are model predictions, not proof that a particular radio protocol is present. Reliable real-world identification requires independently verified signal labels and a consistent feature-extraction pipeline.

## Objectives

- Analyze captured I/Q samples using fixed-size windows.
- Extract time-domain and frequency-domain signal features.
- Train and evaluate machine-learning classifiers.
- Analyze existing SDR captures and newly acquired PlutoSDR samples.
- Compare synthetic-data performance with real-capture observations.
- Evaluate classification performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Identify limitations in dataset quality, capture metadata, and generalization.

## System Architecture

```text
RF Signal / I-Q Capture
          |
          v
   I/Q Sample Loading
          |
          v
  Window Segmentation
          |
          v
   Feature Extraction
          |
          v
  Machine-Learning Model
          |
          v
  Predicted Signal Class
          |
          v
  Results and Evaluation
```

The four-class interactive dashboard and five-class synthetic baseline are separate workflows. They use separate model artifacts and should not be treated as one integrated five-class live classifier.

## Technologies Used

- **Python** — implementation and data processing
- **NumPy** — numerical operations and I/Q processing
- **SciPy** — signal-processing operations
- **pandas** — dataset and evaluation-table handling
- **scikit-learn** — machine-learning models and evaluation metrics
- **joblib** — model and feature-artifact storage
- **matplotlib** — result visualization
- **PlutoSDR** — radio-frequency signal capture
- **libiio / pylibiio** — SDR device communication
- **Tkinter** — interactive desktop dashboard interface

## Machine-Learning Models

The repository contains Random Forest and Multi-Layer Perceptron (MLP) model artifacts, together with feature lists and evaluation utilities.

### Four-class dashboard model

The existing interactive dashboard uses a four-class Random Forest model:

| Class | Label |
|---|---|
| 1 | `bluetooth_like` |
| 2 | `cellular_like` |
| 3 | `noise` |
| 4 | `wifi_like` |

The dashboard loads its model and associated feature list from:

- `rfrecon_four_class_best_rf.pkl`
- `rfrecon_four_class_best_features.pkl`

### Five-class synthetic baseline

A separate five-class Random Forest baseline was trained using a synthetic dataset with 2,500 rows, comprising 500 rows per class.

The classes are:

- `bluetooth_like`
- `cellular_like`
- `noise`
- `wifi_like`
- `zigbee_like`

Its saved artifacts are:

- `rfrecon_five_class_synthetic_rf.pkl`
- `rfrecon_five_class_synthetic_features.pkl`

**The five-class synthetic model is an experimental baseline. It does not replace the four-class model used by the live dashboard.**

## Dataset and Feature Extraction

The project uses synthetic datasets and available I/Q captures.

The feature-extraction pipeline includes measurements such as:

- Mean power
- Root mean square (RMS)
- Peak amplitude
- Signal variance
- Normalized peak frequency
- Normalized spectral centroid
- Normalized spectral bandwidth
- Normalized occupied bandwidth
- Spectral flatness
- Peak-to-average ratio

Feature definitions, sample rates, window sizes, and signal normalization affect model performance. The same feature-extraction implementation and compatible preprocessing must be used during training and inference.

### Data sources

**Synthetic data:** Used for controlled training, baseline comparisons, and stress testing.

**PlutoSDR captures:** Used to test the capture-analysis workflow and inspect predictions on acquired I/Q samples.

**External or candidate captures:** Available datasets may provide useful experimental examples, but their protocol labels and acquisition metadata must be verified before they can be used as confirmed ground truth.

## Evaluation Results

### Five-class synthetic baseline

The five-class synthetic Random Forest baseline was evaluated on a stratified, held-out test split containing 500 synthetic rows.

| Metric | Result |
|---|---:|
| Test accuracy | 99.20% |
| Test samples | 500 |
| Classes | 5 |

The confusion matrix is shown below. Rows represent actual synthetic labels, and columns represent predicted labels.

| Actual / Predicted | Bluetooth | Cellular | Noise | Wi-Fi | Zigbee |
|---|---:|---:|---:|---:|---:|
| Bluetooth | 100 | 0 | 0 | 0 | 0 |
| Cellular | 0 | 100 | 0 | 0 | 0 |
| Noise | 0 | 0 | 100 | 0 | 0 |
| Wi-Fi | 0 | 0 | 0 | 100 | 0 |
| Zigbee | 3 | 0 | 0 | 1 | 96 |

The model correctly classified 496 out of 500 synthetic test rows.

**Interpretation:** These results demonstrate performance on the particular synthetic dataset and split. They do not establish 99.20% accuracy on real-world RF signals. The synthetic data has limitations, including normalized signal power, and independent-capture validation remains necessary.

### PlutoSDR dashboard checks

The four-class dashboard was tested in the local Windows environment using existing and freshly captured I/Q data.

The tested workflow successfully processed captures containing 262,144 complex I/Q samples, corresponding to 64 windows at the configured window size.

The dashboard displayed predicted classes and model scores for the tested captures. These results confirm that the capture-analysis workflow runs in the tested environment; they do not independently verify the true protocol labels or establish real-world classification accuracy.

### Input validation checks

The dashboard was also tested for invalid input conditions:

- Missing capture selection produces a warning.
- A non-positive sample rate, such as `0 MS/s`, produces a validation error.
- Capture analysis checks for empty or invalid I/Q data and insufficient samples for a complete window.

## Dataset and Validation Limitations

- Synthetic training signals are normalized to approximately unit average power, limiting the usefulness of raw power and RMS features for distinguishing synthetic classes.
- Synthetic and real-data feature extraction must be checked for consistency.
- Some available real captures contain multiple windows from a single source capture. Windows from the same capture are not independent recordings.
- Zigbee-like and cellular-like candidate labels have not been independently verified.
- The available IEEE 802.15.4 / 6LoWPAN capture is not confirmed ground truth for Zigbee.
- Sample-rate and center-frequency verification flags are not confirmed for all available captures.
- A predicted class does not prove that a particular protocol is present.
- Model confidence and agreement between classifiers are not substitutes for independent ground-truth evaluation.
- Synthetic-data performance must be reported separately from real-capture performance.

## Important Project Files

| File | Purpose |
|---|---|
| `rfrecon_live_dashboard.py` | Four-class interactive dashboard |
| `rfrecon_consistent_features.py` | Consistent feature extraction used by the dashboard |
| `rfrecon_four_class_best_rf.pkl` | Four-class Random Forest model |
| `rfrecon_four_class_best_features.pkl` | Feature list for the four-class model |
| `rfrecon_five_class_synthetic_rf.pkl` | Five-class synthetic Random Forest baseline |
| `rfrecon_five_class_synthetic_features.pkl` | Feature list for the five-class baseline |
| `rfrecon_five_class_synthetic_report.csv` | Five-class synthetic evaluation report |
| `rfrecon_five_class_synthetic_confusion_matrix.csv` | Five-class synthetic confusion matrix |
| `robust_training_dataset_consistent_5class.csv` | Five-class synthetic training dataset |
| `train_five_class_consistent.py` | Script used to train the five-class baseline |
| `build_consistent_synthetic_dataset.py` | Synthetic dataset-generation utility |
| `rfrecon_final_system.py` | Existing batch-prediction script |
| `rfrecon_hybrid_model.pkl` | Existing hybrid Random Forest model artifact |
| `rfrecon_hybrid_features.pkl` | Feature list for the hybrid model |
| `rfrecon_mlp_model.pkl` | Existing MLP model artifact |
| `rfrecon_mlp_features.pkl` | Feature list for the MLP model |
| `hybrid_training_dataset.csv` | Existing hybrid dataset |
| `robust_training_dataset.csv` | Existing synthetic training dataset |
| `stress_test_dataset.csv` | Existing synthetic stress-test dataset |
| `four_class_baseline_oct17_results.csv` | Existing four-class evaluation results |
| `rfrecon_final_summary.csv` | Previously saved final-capture summary |
| `RFRecon_final_results_table.csv` | Existing experiment comparison table |

The repository contains artifacts from different experimental stages. Check each script's model and feature paths before running it; do not assume every script uses the same model.

## Software Requirements

The tested local environment uses:

- Python 3.13.13
- A virtual environment named `.venv`
- NumPy
- SciPy
- pandas
- scikit-learn
- joblib
- matplotlib
- Pillow
- pylibiio and the locally configured `libiio.dll` for SDR-related functionality

A clean installation on another computer has not been independently verified. Package versions and native-library locations may differ.

## Running the Four-Class Dashboard on Windows

### 1. Open the project directory

Open Command Prompt in the `RFRecon_AI` project root, where `rfrecon_live_dashboard.py` is located.

### 2. Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### 3. Launch the dashboard

On the tested Windows setup, the PlutoSDR library is located at:

`C:\Program Files\Analog Devices\Scopy\libiio.dll`

Run the following command from the project root. It applies a local DLL lookup workaround to the Python process that launches the dashboard.

```cmd
python -c "import os,ctypes.util,runpy,sys; p=r'C:\Program Files\Analog Devices\Scopy'; os.add_dll_directory(p); ctypes.util.find_library=lambda name: os.path.join(p,'libiio.dll') if name in ('libiio.dll','iio') else None; sys.argv=['rfrecon_live_dashboard.py']; runpy.run_path('rfrecon_live_dashboard.py',run_name='__main__')"
```

This command was tested on the local Windows environment using Python 3.13.13 and the existing Scopy installation. The DLL directory, model files, and project files must exist at the configured paths. Other Windows setups may require a different DLL path.

### 4. Use the dashboard

- **Select IQ capture:** Choose an existing compatible complex64 I/Q capture file.
- **Analyze capture:** Extracts features and displays predictions using the four-class model.
- **Capture from Pluto:** Captures new I/Q data from a connected and configured PlutoSDR, then analyzes it.

Use the sample rate that matches the actual acquisition settings for the capture. Do not assume that a configured frequency or sample rate has been independently verified simply because the dashboard runs.

**Safety note:** Keep the four-class live dashboard model separate from the five-class synthetic baseline. Back up existing prediction files before running any separate script that may overwrite results.

## Current Status

### Implemented and tested

- Random Forest and MLP model artifacts.
- Synthetic datasets and stress-testing utilities.
- Feature-extraction and evaluation scripts.
- Four-class dashboard launch in the tested Windows environment.
- Existing-capture analysis and fresh PlutoSDR capture analysis.
- Five-class synthetic baseline training and held-out evaluation.
- Five-class report and confusion-matrix exports.
- Input-validation checks for missing capture selection and invalid sample rate.

### Remaining validation work

1. Verify feature definitions and preprocessing across all training and inference scripts.
2. Confirm acquisition metadata against capture logs and actual hardware settings.
3. Independently verify cellular-like and Zigbee-related ground-truth labels.
4. Collect or identify independently labelled real captures for each target class.
5. Evaluate using capture-aware train/test splits to reduce leakage between windows from the same capture.
6. Report real-capture metrics separately from synthetic-data results.
7. Confirm the final README instructions and required files on the submission environment.

## Conclusion

RFRecon_AI is an experimental SDR classification pipeline that combines I/Q signal processing, feature extraction, machine-learning models, PlutoSDR capture analysis, and evaluation utilities.

The current implementation demonstrates a working four-class capture-analysis dashboard and a separate five-class synthetic baseline. Further validation with consistent preprocessing, verified capture metadata, independently labelled real signals, and capture-aware evaluation is required before making claims about reliable real-world protocol identification.
