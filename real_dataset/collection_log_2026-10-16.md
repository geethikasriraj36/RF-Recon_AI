\# RFRecon — Data Collection Log

\*\*Date:\*\* 16 October 2026



\## 1. Existing recordings audit



| File | Size (approx.) | Label | Sample rate | LO / center frequency | RF bandwidth | Status |

|---|---:|---|---|---|---|---|

| `real\_noise\_01.dat` | 0.274 GiB | Noise | Unknown | Unknown | Unknown | Existing capture; settings need verification |

| `real\_wifi\_01.dat` | 0.763 GiB | Wi-Fi-like | Unknown | Unknown | Unknown | Existing capture; settings need verification |

| `real\_bluetooth\_01.dat` | 0.551 GiB | Bluetooth-like | Unknown | Unknown | Unknown | Existing capture; settings need verification |

| `real\_cellular\_01.dat` | 1.75 GiB | Cellular-like | Configured 20 MS/s; actual acquisition unverified | Flowgraph configured 1.872 GHz; actual acquisition unverified | Unknown | Frequency discrepancy unresolved |

| `real\_cellular\_scan\_01.dat` | 3.247 GiB | Cellular scan; exact capture conditions unknown | Unknown | Unknown | Unknown | Original recording preserved |



File sizes are approximate; use the actual file sizes for exact reporting.



\## 2. Cellular recording discrepancy



The inspected flowgraph configures:

\- Sample rate: 20 MS/s

\- LO frequency: 1.872 GHz

\- Gain mode: `slow\_attack`

\- Configured gain: 64 dB



The dataset metadata lists:

\- Sample rate: 20 MS/s

\- Center frequency: 1.856 GHz



The actual acquisition settings have not been independently confirmed. The metadata discrepancy remains unresolved. Do not treat configured settings as proof of actual recording conditions.



\## 3. Zigbee dataset investigation



Source investigated: SDR4IoT BLE and Zigbee dataset, Zenodo record 4639390.



\- Downloaded archive: `zigbee\_ble\_dataset.zip`

\- Checked 123 Zigbee `.sigmf` containers.

\- All inspected `.sigmf-data` members were empty.

\- No usable IQ data was found in the checked archive.

\- `dataset.zip` was confirmed identical to `zigbee\_ble\_dataset.zip` using SHA-256 and removed as a duplicate copy.

\- The remaining archive is retained for reference.



\## 4. IEEE 802.15.4 / 6LoWPAN recording



Source: SDRLab educational dataset page.

Recording: `real\_dataset/6lowpan\_ping.dat`



\- File size: 317,864,640 bytes

\- Interpreted sample format: complex64, pending independent format confirmation

\- Sample count under complex64 interpretation: 39,733,080

\- Flowgraph default sample rate: 4 MS/s

\- Flowgraph default channel frequency: 2.415 GHz

\- Actual recording sample rate and center frequency: unverified

\- RF bandwidth and capture gain: unknown

\- Protocol label: `ieee802154\_6lowpan`

\- Zigbee application-layer identity: unverified



Initial inspection of 100,000 samples found finite values and a byte count divisible by 8.



\## 5. Extracted features



Output: `real\_dataset/ieee802154\_6lowpan\_features.csv`



\- Rows: 500

\- Feature columns: 10

\- Metadata columns: 13

\- Window size: 4,096 samples

\- Missing feature values: 0

\- Non-finite feature values: 0

\- Duplicate feature rows: 0

\- Unique capture IDs: 1

\- Sample-rate verification flag: false

\- Center-frequency verification flag: false

\- RF bandwidth: unknown



Features were generated using `rfrecon\_consistent\_features.py`. The 4 MS/s flowgraph setting was used provisionally. Regenerate the features if the recording's actual sample rate is later verified to be different.



\## 6. Decisions and outstanding work



1\. Preserve all original `.dat` recordings and trained models.

2\. Keep the 6LoWPAN feature CSV separate from the existing four-class training dataset.

3\. Do not label this recording as verified Zigbee application traffic without supporting evidence.

4\. Verify the acquisition metadata for all original captures.

5\. Obtain a defensible protocol label and verify the IQ format and sample rate.

6\. Only then decide whether to create a five-class dataset and evaluate it with capture-aware splits.

7\. Recheck feature order, class counts, missing values, and provenance before training.



\*\*Status:\*\* Initial feature extraction and CSV integrity checks passed. Acquisition metadata verification and final five-class dataset construction remain incomplete.

## PCAP Reference Validation — 16 October 2026

Reference file: `real_dataset/6lowpan_ping.pcap`
Size: 1,998 bytes
Format: Classic PCAP, version 2.4
Link type: 195 — IEEE 802.15.4 with FCS

Validation:
- Packet records: 14
- Packet sizes: 125 bytes each
- Malformed records: 0
- Trailing bytes: 0
- Timestamp span: 6.01492 seconds
- Validation report: `real_dataset/ieee802154_pcap_validation.txt`

Conclusion:
The PCAP structure is valid and declares IEEE 802.15.4 frames. This supports documenting the reference as IEEE 802.15.4-related, but does not independently prove that the raw IQ file corresponds to these packets or establish its sample rate, center frequency, or RF bandwidth. Zigbee application-layer identity remains unverified.

Dataset decision:
Keep `ieee802154_6lowpan_features.csv` separate from the four-class training dataset until the recording's provenance and acquisition metadata are sufficiently verified. Do not relabel it as `zigbee_like` merely to reach five classes.


 

## 7. Additional Zigbee Archive Audit — 16 October 2026

Directory inspected: `real_dataset/zigbee/scenario6/scene38`

Audit results:
- PCAP files found: 123
- Total PCAP file size: 175,332 bytes
- SigMF containers found: 123
- SigMF metadata members found: 123
- Non-empty `.sigmf-data` members found: 0
- CSV files found: 123
- CSV files containing data rows: 0
- File inspection errors: 0

The inspected SigMF metadata describes BLE advertising packets, while the associated IQ data members are empty. The CSV files contain headers only and provide no recording-setting values.

The PCAP files are packet captures, not raw IQ recordings. Their presence does not establish that usable Zigbee IQ data is available.

**Decision:** This local archive does not provide usable IQ samples for the fifth-class feature dataset. Do not label its recordings as verified Zigbee IQ data.

## 8. IEEE 802.15.4 / 6LoWPAN Feature CSV Recheck — 16 October 2026

File: `real_dataset/ieee802154_6lowpan_features.csv`

- Shape: 500 rows × 23 columns
- Feature columns: 10
- Metadata columns: 13
- Class label: `ieee802154_6lowpan` for all 500 rows
- Sample-rate verification: `False` for all rows
- Center-frequency verification: `False` for all rows
- RF-bandwidth verification: `False` for all rows
- Missing values: `rf_bandwidth_hz` is missing for all 500 rows
- Configured sample rate: 4 MS/s, provisional
- Configured center frequency: 2.415 GHz, provisional
- Capture gain: unknown
- Protocol label status: `IEEE 802.15.4 / 6LoWPAN source; Zigbee unverified`

The feature CSV retains the expected feature names and order. The missing RF-bandwidth values are intentional because the actual acquisition bandwidth is unknown.

**Decision:** Retain the feature CSV separately from the four-class training dataset. Do not retrain the classifier or promote this recording to a verified Zigbee class until the relevant provenance and acquisition uncertainties are addressed.

## 9. Current Project Status

The four-class model and original IQ recordings remain unchanged. The additional archive audit found no usable Zigbee IQ samples in the inspected SigMF containers. The 6LoWPAN feature extraction and structural checks are complete, but acquisition settings remain unverified.

Outstanding work:
1. Verify the actual acquisition settings and format of the 6LoWPAN IQ recording.
2. Continue verifying the original recordings' metadata, especially the cellular frequency discrepancy.
3. Decide whether IEEE 802.15.4 / 6LoWPAN is an acceptable fifth class for the project requirements.
4. Build and validate a combined dataset only after the fifth class is adequately documented.
5. Evaluate with capture-aware splits to avoid leakage between windows from the same recording.

