\# RFRecon\_AI — Task 2: Feature Pipeline Audit



\## 1. Objective



Audit the feature extraction and dataset preparation pipelines used by RFRecon\_AI. The objective is to identify differences in feature definitions, dataset provenance, normalization, and model input preparation without modifying existing datasets or model artifacts.



\## 2. Shared Feature Schema



The shared extractor, `rfrecon\_consistent\_features.py`, defines ten features:



1\. `mean\_power`

2\. `rms`

3\. `peak\_amplitude`

4\. `variance`

5\. `peak\_frequency\_normalized`

6\. `spectral\_centroid\_normalized`

7\. `spectral\_bandwidth\_normalized`

8\. `occupied\_bandwidth\_normalized`

9\. `spectral\_flatness`

10\. `peak\_to\_average\_ratio`



The presence of matching feature names does not establish that different pipelines calculate the features identically.



\## 3. Feature Formula Consistency



\### 3.1 Synthetic training pipeline



`prepare\_robust\_dataset.py` computes features for the robust training dataset. Its periodogram configuration and numerical safeguards differ from those in `rfrecon\_consistent\_features.py`.



A deterministic comparison using a seeded complex IQ signal found that five of the ten features matched within the comparison tolerance:



\- `mean\_power`

\- `rms`

\- `peak\_amplitude`

\- `variance`

\- `peak\_frequency\_normalized`



The following five differed:



\- `spectral\_centroid\_normalized`

\- `spectral\_bandwidth\_normalized`

\- `occupied\_bandwidth\_normalized`

\- `spectral\_flatness`

\- `peak\_to\_average\_ratio`



The comparison was performed in memory using the extracted functions. The top-level dataset-generation pipeline was not run for this comparison.



\### 3.2 Real Wi-Fi feature preparation



`prepare\_real\_wifi\_dataset.py` uses FFT-based calculations. Its occupied-bandwidth calculation uses a 5%–95% cumulative-energy interval, and its peak-to-average feature uses an amplitude-to-RMS ratio.



The shared extractor instead uses a periodogram-based spectral calculation, a different occupied-bandwidth calculation, and a spectral peak-to-average ratio.



A read-only numerical comparison on `real\_wifi\_dataset/real\_wifi\_0000.npy` found five matching features and five differing features. This is a single-window comparison, not a full-dataset validation.



\### 3.3 Final inference pipeline



`rfrecon\_final\_system.py` contains a separate feature extraction function. Inspection found important differences:



\- `variance` is calculated from power rather than amplitude.

\- `occupied\_bandwidth\_normalized` is based on bins exceeding 1% of peak spectral power rather than the shared cumulative-energy method.

\- `peak\_to\_average\_ratio` is calculated as peak amplitude divided by RMS rather than as a spectral ratio.

\- Spectral calculations use an FFT-based implementation rather than the shared periodogram configuration.



Therefore, matching feature names and ordering alone do not guarantee compatibility between training and inference.



\## 4. Dataset Composition and Provenance



The existing `hybrid\_training\_dataset.csv` was inspected read-only.



\- Total rows: 2,400

\- `noise`: 500 rows

\- `bluetooth\_like`: 500 rows

\- `zigbee\_like`: 500 rows

\- `cellular\_like`: 500 rows

\- `wifi\_like`: 400 rows



The ten feature columns had no missing values in this inspection. Where `class\_name` was populated, it agreed with `label`.



The dataset combines synthetic class data with real Wi-Fi feature rows. The label distribution and absence of missing feature values do not, by themselves, establish that the classes are independently validated or that the model generalizes to new captures.



\## 5. Power Normalization and Distribution Shift



`generate\_robust\_training.py` includes a normalization function that rescales synthetic signals toward unit average power.



In the inspected hybrid dataset, the synthetic classes had mean `mean\_power` and `rms` values close to 1.0, while the real Wi-Fi rows had substantially lower average values.



This is evidence of a potential distribution mismatch between synthetic training examples and the real Wi-Fi examples. It does not, by itself, prove that a classifier relies on power differences or that its predictions are inaccurate.



A controlled evaluation should determine whether performance changes when feature scaling and capture conditions are handled consistently.



\## 6. Split Strategy and Leakage Considerations



The inspected training scripts use stratified random row-level train/test splits with a fixed random seed.



If multiple windows originate from the same capture, a row-level split can place windows from that capture in both training and testing sets. This can produce optimistic estimates of generalization to unseen captures.



A capture-aware split is preferable when the dataset contains sufficient independent captures for each class. Current provenance limitations may prevent such a split for some classes.



\## 7. Known Data and Metadata Limitations



The available collection records contain unresolved acquisition metadata discrepancies, including conflicting configured center-frequency values for the cellular capture. Acquisition settings have not been independently confirmed.



The existing IEEE 802.15.4/6LoWPAN-related material is not sufficient to establish verified Zigbee application traffic. The relevant feature windows originate from a single capture, and the capture's actual acquisition settings are not fully verified.



These limitations should be stated explicitly in evaluation results and presentation materials. Synthetic or unverified data must not be presented as independently verified real-world ground truth.



\## 8. Model Input Compatibility



The inspected training scripts consume precomputed feature columns rather than implementing the shared feature extractor themselves.



The final inference pipeline contains its own feature calculations. The inspection therefore identifies a compatibility risk between precomputed training features and inference-time features.



A complete validation should compare the exact feature vectors supplied to each model on identical IQ windows. This audit did not load model pickle files or execute the final inference pipeline.



\## 9. Safe Audit Method



The audit used source inspection, existing CSV inspection, and in-memory numerical comparisons.



The following were deliberately not executed:



\- Dataset-generation scripts that write existing CSV files.

\- Model-training scripts that create or overwrite model artifacts.

\- The final inference pipeline that reads captures and writes output reports.

\- The existing feature-formula audit script, because it performs file I/O at top level.



No dataset or model file was intentionally modified by the audit steps described here. The repository already contained a modified `robust\_training\_dataset.csv` and multiple untracked files; those changes were preserved and should be reviewed separately.



\## 10. Recommended Next Steps



1\. Standardize feature extraction so training and inference use the same implementation and configuration.

2\. Recompute datasets only after creating and verifying a separate backup or a new output path.

3\. Add numerical consistency tests using identical IQ windows and explicit tolerances.

4\. Track source capture, sample rate, center frequency, window size, and acquisition provenance for every feature row.

5\. Prefer capture-aware evaluation and report when the available number of independent captures is insufficient.

6\. Evaluate synthetic-to-real distribution shift with controlled experiments.

7\. Verify acquisition metadata before making claims about cellular or Zigbee classification.

8\. Preserve current datasets, models, snapshots, and local changes until their roles are understood.



\## 11. Conclusion



The audit identifies material differences among feature extraction implementations, a potential synthetic-to-real power distribution mismatch, and limitations in capture-aware evaluation and acquisition metadata. Matching feature names are not sufficient to guarantee consistent model inputs.



These findings justify standardizing the feature pipeline and strengthening provenance and evaluation before making broad claims about real-world classification performance.



\---



\*\*Scope note:\*\* This is a source and dataset audit, not proof of end-to-end model accuracy. No new model performance metrics were generated as part of this report.

