# RFRecon_AI — Five-Class Evaluation
Date: 2026-10-18

## Evaluation type
Existing synthetic stress-test feature table evaluated with the existing saved Random Forest model. No training was performed and no existing model or dataset was modified.

## Dataset and compatibility
- Dataset: stress_test_dataset.csv
- Samples: 2,500; 500 per class
- Saved model: rfrecon_rf_model.pkl
- Feature list: rfrecon_features.pkl
- Feature-column order matched the saved feature list.
- Model classes: 0, 1, 2, 3, 4

## Overall result
Accuracy: 97.84% (2,446/2,500 correct)

## Confusion matrix
Rows are actual classes; columns are predicted classes. Order: Noise, Wi-Fi-like, Bluetooth-like, Zigbee-like, Cellular-like.

| Actual \\ Predicted | Noise | Wi-Fi | Bluetooth | Zigbee | Cellular |
|---|---:|---:|---:|---:|---:|
| Noise | 500 | 0 | 0 | 0 | 0 |
| Wi-Fi-like | 0 | 497 | 0 | 3 | 0 |
| Bluetooth-like | 0 | 11 | 464 | 25 | 0 |
| Zigbee-like | 0 | 10 | 1 | 485 | 4 |
| Cellular-like | 0 | 0 | 0 | 0 | 500 |

## Per-class metrics
| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Noise | 1.0000 | 1.0000 | 1.0000 | 500 |
| Wi-Fi-like | 0.9595 | 0.9940 | 0.9764 | 500 |
| Bluetooth-like | 0.9978 | 0.9280 | 0.9617 | 500 |
| Zigbee-like | 0.9454 | 0.9700 | 0.9576 | 500 |
| Cellular-like | 0.9921 | 1.0000 | 0.9960 | 500 |
| Macro average | 0.9790 | 0.9784 | 0.9783 | 2,500 |

## Limitations
This result is for the existing synthetic stress-test dataset. It does not establish real-world accuracy or independent-capture generalization. Real candidate labels and acquisition metadata remain unverified. A defensible real five-class evaluation requires trustworthy ground truth, compatible feature extraction, and appropriately independent captures.

## Remaining work
1. Verify real capture labels and acquisition metadata.
2. Establish feature compatibility before combining datasets.
3. Evaluate real captures only where ground truth supports scoring.
4. Report synthetic and real results separately.
