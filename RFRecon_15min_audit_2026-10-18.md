# RFRecon_AI — Data Readiness Audit
Date: 2026-10-18

## Preservation
Existing captures, feature tables, models, and prior results are to remain unchanged. This report is a separate artifact.

## Frequency discrepancy
The inspected trial.grc configuration specifies 1.872 GHz. Some cellular-capture metadata also specifies 1.872 GHz, while the collection log mentions 1.856 GHz. Historical tuned frequency is NOT independently verified. Do not silently choose one value as confirmed.

## Label and capture evidence
- Noise, Wi-Fi-like, Bluetooth-like, and cellular-like entries exist in the four-class baseline feature table; each class has one source recording.
- The cellular capture remains a candidate; its class label is not independently verified.
- The IEEE 802.15.4/6LoWPAN dataset has associated PCAP evidence, but this alone does not prove the signal is Zigbee. Keep its protocol label as unverified.
- Existing BLE IQ data is evidence of BLE-related capture, not a verified Zigbee capture.
- Existing feature windows from one capture are not independent captures. Randomly splitting windows from the same recording can leak capture-specific patterns.

## Existing evaluation evidence
- Synthetic five-class stress test: 2,446/2,500 correct; accuracy 97.84%. This is synthetic evaluation, not real-world accuracy.
- Existing four-class baseline report: accuracy 69.33%; its window-level split does not establish independent-capture generalization.
- Pluto smoke test: 64/64 windows predicted bluetooth_like; mean maximum model probability 94.68%. True signal class is unverified, so this is a prediction-consistency check, not an accuracy result.
- The saved four-class model and five-class model must not be interchanged without verifying label mapping and feature compatibility.

## Evaluation decision
Use the existing synthetic result as synthetic-only evidence. Do not claim verified five-class real-data accuracy until class labels, capture provenance, sample rate, and feature compatibility are supported. Existing saved models and original captures must remain unchanged.

## Outstanding limitations
1. Actual historical capture frequency remains unresolved between 1.856 and 1.872 GHz.
2. No independently verified Zigbee capture has been established from the evidence reviewed.
3. The cellular capture label remains a candidate.
4. There is only one source recording per class in the four-class baseline table, so independent-capture testing is not supported by that table.
5. The existing 97.84% result is synthetic; do not present it as real-data accuracy.
