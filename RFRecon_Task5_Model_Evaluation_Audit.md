\# RFRecon\_AI — Task 5: Model Evaluation Audit



\## 1. Objective



This audit reviews the current model training and evaluation setup in RFRecon\_AI. It focuses on dataset splits, feature compatibility, synthetic-to-real distribution differences, capture independence, and the evidence required to support model performance claims.



This is a source and dataset audit. It does not generate new model scores or claim independently verified real-world accuracy.



\## 2. Models and Training Pipelines Inspected



The inspected project includes:



\- `train\_robust\_model.py`

\- `train\_hybrid\_model.py`

\- `create\_hybrid\_dataset.py`

\- `rfrecon\_final\_system.py`



The training scripts consume precomputed feature columns. The hybrid training script uses a Random Forest classifier with 300 trees and saves model and feature-list pickle files.



The final inference pipeline contains its own feature extraction function and loads previously saved model artifacts. These operations were not executed during this audit.



\## 3. Feature Compatibility Risk



The feature consistency audit found that the shared extractor, synthetic training extractor, real Wi-Fi feature generator, and final inference pipeline do not calculate all ten features identically.



Differences include:



\- Periodogram versus FFT-based spectral calculations.

\- Different occupied-bandwidth definitions.

\- Different spectral-flatness calculations and numerical safeguards.

\- Different peak-to-average-ratio definitions.

\- Different definitions of the variance feature in the final inference pipeline.



Matching feature names and ordering are not sufficient to guarantee equivalent model inputs.



\*\*Evaluation implication:\*\* performance results can be misleading if the model is trained on feature values calculated by one implementation and receives values calculated by another.



Before interpreting final inference results, the exact training and inference feature vectors should be compared on identical IQ windows.



\## 4. Training/Test Split Strategy



The inspected training scripts use stratified random row-level splits, generally with an 80/20 ratio and a fixed random seed.



Stratification helps preserve class proportions, but it does not guarantee independence between training and testing examples.



When multiple windows originate from one capture, a row-level split may place windows from that capture in both sets. Similar signal conditions or capture-specific characteristics can then appear in both training and testing data.



\*\*Recommendation:\*\* use a capture-aware split where independent captures are available. Keep all windows from a source capture within one split whenever the evaluation goal is generalization to unseen captures.



\## 5. Limitations of Current Capture Availability



The inspected `real\_dataset/real\_capture\_features\_consistent.csv` contains 500 windows each for `bluetooth\_like`, `noise`, and `wifi\_like`, with each class associated with one source capture.



Five hundred windows from one capture are not equivalent to 500 independent captures. With only one identified source capture per class, a capture-aware holdout for those classes cannot be constructed from this dataset alone.



Additional independently acquired captures are needed to estimate generalization across capture sessions.



\## 6. Synthetic-to-Real Distribution Shift



The synthetic signal generator normalizes signals toward unit average power. In the inspected hybrid dataset, the synthetic classes have mean `mean\_power` and `rms` values close to 1.0, while the real Wi-Fi rows have substantially lower values.



This is evidence of a potential distribution mismatch.



A classifier could potentially learn differences associated with data origin or signal scaling rather than only the intended modulation or protocol characteristics. However, the observed mismatch alone does not prove that this occurs or that the classifier performs poorly.



\*\*Recommended experiments:\*\*



1\. Evaluate each class using held-out captures where possible.

2\. Compare feature distributions by class and data origin.

3\. Test consistent preprocessing and normalization on training and evaluation data.

4\. Report the impact of each change instead of assuming it improves performance.

5\. Keep all preprocessing decisions independent of the held-out test results.



\## 7. Label and Provenance Limitations



The hybrid dataset contains 2,400 rows: 500 each for `noise`, `bluetooth\_like`, `zigbee\_like`, and `cellular\_like`, and 400 for `wifi\_like`.



These counts do not imply equivalent evidence quality across classes. The hybrid dataset combines synthetic examples with real Wi-Fi features, while some candidate real-capture labels and acquisition settings remain unverified.



The inspected cellular collection records contain an unresolved center-frequency discrepancy: 1.872 GHz in the flowgraph configuration versus 1.856 GHz in metadata. The actual acquisition setting has not been independently confirmed.



The inspected IEEE 802.15.4/6LoWPAN-related data originates from one capture, and the available evidence does not establish verified Zigbee application traffic. These data limitations should be reflected in the evaluation report.



\## 8. Evaluation Metrics to Report



For a five-class classification task, report at least:



\- Overall accuracy.

\- Per-class precision, recall, and F1-score.

\- Macro-average F1-score.

\- Confusion matrix.

\- Number of test examples per class.

\- Number of independent source captures per class.

\- Exact split method and random seed.

\- Feature extractor version and preprocessing configuration.

\- Whether the evaluation uses synthetic data, real captures, or a mixture.



A confusion matrix and per-class metrics help reveal cases where high overall accuracy hides poor performance on a particular class.



No new metric values were generated by this audit; they must be obtained from an actual, verified evaluation run.



\## 9. Recommended Evaluation Protocol



1\. Freeze the feature definitions and preprocessing configuration used by both training and inference.

2\. Record the provenance and source capture ID for every example.

3\. Split by capture when enough independent captures exist.

4\. Fit any learned preprocessing only on the training split.

5\. Train using the selected training set and record the configuration.

6\. Evaluate once on an untouched test set.

7\. Calculate accuracy, per-class precision/recall/F1, macro-F1, and a confusion matrix.

8\. Document unsupported classes, unverified labels, and limitations.

9\. Repeat with additional independent captures before making broad generalization claims.

10\. Keep the current datasets and model artifacts unchanged until the new protocol and outputs have been reviewed.



\## 10. Safe Audit Scope



This review was based on source inspection and existing dataset findings. Training and inference scripts that can write output files were not executed, and model pickle files were not loaded.



Existing dataset modifications, untracked files, snapshots, and reports should remain untouched unless separately reviewed and explicitly approved.



\## 11. Conclusion



The main evaluation risks identified are inconsistent feature formulas between training and inference, possible capture leakage from row-level splits, synthetic-to-real distribution differences, and limited or unverified acquisition provenance for some classes.



These risks do not establish that the model is inaccurate; they limit the strength of conclusions that can currently be drawn from unverified performance measurements.



A reliable evaluation requires consistent feature extraction, traceable labels, independent capture-aware testing where possible, and transparent reporting of per-class metrics and data limitations.



\---



\*\*Scope note:\*\* This report documents evaluation risks and recommendations. It does not claim new accuracy results or independently verified real-world classification performance.

