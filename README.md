# RFRecon - AI-Based RF Signal Classification

## Overview

RFRecon is a software-defined radio (SDR) project that analyzes complex in-phase and quadrature (I/Q) samples and extracts signal features for radio-frequency classification.

The project includes Random Forest and Multi-Layer Perceptron (MLP) classifiers, synthetic RF datasets, PlutoSDR capture analysis, stress testing, and evaluation utilities.

**Important:** Classification outputs are model predictions. Their reliability depends on feature-pipeline consistency, independently verified signal labels, and evaluation methodology.

## Objectives

- Analyze captured I/Q samples using fixed-size windows.
- Extract time-domain and frequency-domain features.
- Compare Random Forest and MLP predictions.
- Evaluate model behavior on synthetic and real SDR captures.
- Investigate classification errors, feature differences, and robustness under stress conditions.

## System Architecture

1. **Signal acquisition:** RF samples are obtained from existing PlutoSDR capture files or, when configured, from the SDR hardware.
2. **Windowing:** Samples are divided into windows, commonly 4,096 complex samples.
3. **Feature extraction:** Time-domain and spectral features are calculated.
4. **Classification:** Saved Random Forest and MLP models produce predictions and confidence scores.
5. **Evaluation:** CSV files and plots summarize predictions, confusion matrices, and model comparisons.

## Features

The feature set includes:

- Mean power
- Root mean square (RMS)
- Peak amplitude
- Variance
- Normalized peak frequency
- Normalized spectral centroid
- Normalized spectral bandwidth
- Normalized occupied bandwidth
- Spectral flatness
- Peak-to-average ratio

Feature definitions must remain consistent between model training and inference. The current final-system script uses some definitions that differ from existing feature extractors; its results require validation before being treated as reliable performance measurements.

## Machine Learning Models

### Random Forest

The Random Forest model is used as the primary classifier in the final-system script.

### Multi-Layer Perceptron

The MLP model provides a comparative classification result.

Both models load saved model and feature artifacts. Agreement between them indicates prediction consistency, not necessarily correctness.

## Existing Evaluation Results

The following table summarizes previously saved project results.

| Experiment | Random Forest | MLP |
|---|---:|---:|
| Hybrid validation | 99.38% | 98.96% |
| Unseen real Wi-Fi experiment | 100% | 100% |
| PlutoSDR capture 02 | 99.95% | 99.82% |
| PlutoSDR capture 03 | 100% | 100% |
| Synthetic low-SNR stress test | 97.84% | 79.08% |

**These are reported results from existing project artifacts, not independently revalidated measurements.** The evaluation methods and ground-truth labels differ or remain unverified across experiments.

A separate four-class temporal evaluation reported 69.33% accuracy and a macro F1-score of 69.06%. It used windows from existing captures rather than an independent capture for each class.

The high prediction agreement reported for PlutoSDR captures must not be interpreted as verified classification accuracy without independent ground truth.

## Dataset and Validation Limitations

- Synthetic training signals are normalized to approximately unit average power, limiting the usefulness of raw power and RMS features for distinguishing synthetic classes.
- Synthetic and real-data feature extraction methods are not fully consistent.
- Some real captures have only one source capture per class, limiting independent-capture evaluation.
- Zigbee-like and cellular-like real-capture labels have not been independently verified.
- A signal classified as a particular protocol is not proof that the protocol is present.
- Confidence scores and agreement between classifiers are not substitutes for ground-truth evaluation.
- Capture sample rates and center frequencies should be checked against acquisition logs before being presented as confirmed hardware settings.
- Synthetic stress-test performance must be distinguished from performance on independently captured and labelled real signals.

## Important Project Files

| File | Purpose |
|---|---|
| `rfrecon_live_dashboard.py` | Four-class interactive dashboard |
| `rfrecon_consistent_features.py` | Feature extractor used by the dashboard |
| `rfrecon_four_class_best_rf.pkl` | Four-class Random Forest model |
| `rfrecon_four_class_best_features.pkl` | Feature list associated with the four-class model |
| `rfrecon_final_system.py` | Final batch-prediction script |
| `rfrecon_hybrid_model.pkl` | Saved Random Forest model artifact |
| `rfrecon_hybrid_features.pkl` | Random Forest feature list |
| `rfrecon_mlp_model.pkl` | Saved MLP model artifact |
| `rfrecon_mlp_features.pkl` | MLP feature list |
| `hybrid_training_dataset.csv` | Existing hybrid dataset |
| `robust_training_dataset.csv` | Existing synthetic training dataset |
| `stress_test_dataset.csv` | Existing synthetic stress-test dataset |
| `four_class_baseline_oct17_results.csv` | Saved four-class evaluation metrics |
| `rfrecon_final_summary.csv` | Previously saved final-capture summary |
| `RFRecon_final_results_table.csv` | Existing experiment comparison table |

The files listed above serve different stages of the project. Verify that a particular script's model and feature paths match the intended artifacts before running it.

## Software Requirements

The tested local environment uses:

- Python 3.13.13
- A Python virtual environment named `.venv`
- NumPy
- SciPy
- pandas
- scikit-learn
- joblib
- matplotlib
- Pillow
- pylibiio and the locally configured libiio DLL for SDR-related functionality

Exact dependency compatibility may vary. A clean installation on another computer has not been verified.

## Running the Four-Class Dashboard on Windows

### 1. Open the project directory

Open Command Prompt in the RFRecon_AI project root, where `rfrecon_live_dashboard.py` is located.

### 2. Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### 3. Launch the dashboard

On the tested Windows setup, the PlutoSDR library is located at:

`D:\RFRecon\Windows\Windows-VS-2022-x64\libiio.dll`

The following command applies the local DLL lookup workaround to the Python process that launches the dashboard:

```cmd
python -c "import os,ctypes.util,runpy,sys; p=r'D:\RFRecon\Windows\Windows-VS-2022-x64'; os.add_dll_directory(p); old=ctypes.util.find_library; ctypes.util.find_library=lambda name: p+r'\libiio.dll' if name=='libiio.dll' else old(name); sys.argv=['rfrecon_live_dashboard.py']; runpy.run_path('rfrecon_live_dashboard.py',run_name='__main__')"
```

This command is specific to the tested Windows environment. The DLL directory and model files must exist at the configured paths. The dashboard may access the PlutoSDR when capture controls are used.

**Safety note:** Opening the dashboard does not validate classification accuracy. Back up existing prediction files before running any separate script that exports or overwrites results.

## Current Status

**Implemented:**

- Saved Random Forest and MLP classifiers
- Synthetic datasets and stress-test utilities
- SDR capture files and capture-analysis utilities
- Feature-extraction scripts
- Evaluation scripts and result visualizations
- Four-class dashboard launch verified in the local environment

**Remaining validation work:**

1. Align and verify feature definitions between training and inference.
2. Validate the corrected feature pipeline without overwriting existing artifacts.
3. Evaluate on independently captured and independently labelled signals.
4. Verify real Zigbee-like and cellular-like ground truth.
5. Confirm acquisition metadata, including sample rate and center frequency.
6. Recalculate final metrics only after the evaluation protocol and labels are established.
7. Verify installation and execution on a clean environment if portability is required.

## Conclusion

RFRecon provides an experimental SDR classification pipeline combining feature extraction, Random Forest, MLP, and capture analysis. Existing results demonstrate promising performance on some saved benchmarks, but further validation is required before making claims about reliable real-world protocol identification.

The project distinguishes model predictions and synthetic stress-test results from independently verified real-world classification performance. This distinction is essential for interpreting the results responsibly.
