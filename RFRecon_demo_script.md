\# RFRecon\_AI — Demo Script



\*\*Duration:\*\* 2–3 minutes



\## 1. Introduction

“RFRecon\_AI is an RF signal classification project that analyzes IQ samples from saved captures or an ADALM-PLUTO SDR. It extracts ten signal features and uses a Random Forest classifier to predict one of four classes: Bluetooth-like, cellular-like, noise, and Wi-Fi-like.”



\## 2. Demonstrate the dashboard

“I will select a saved IQ capture and analyze it. The dashboard displays the predicted class, model probabilities, sample and window counts, and spectrum visualization.”



Show the successful analysis of `real\_noise\_01.dat`. Explain that the displayed Noise prediction is a model output, not independently verified ground truth.



\## 3. Demonstrate input validation

“I will now enter a sample rate of zero. The application rejects this invalid input and displays a validation message instead of proceeding with the analysis.”



Show the message: “Sample rate must be a positive number.”



\## 4. Explain model evaluation

“The project also includes a separate five-class synthetic evaluation. Its reported accuracy is 99.20% on a row-level synthetic test split. This is synthetic-only performance and must not be interpreted as verified real-world RF accuracy.”



Show `rfrecon\_five\_class\_synthetic\_confusion\_matrix.png`.



\## 5. Explain real-data limitations

“The acquisition settings for the cellular recording are not independently confirmed. The saved flowgraph specifies 20 MS/s and 1.872 GHz, while dataset metadata lists 1.856 GHz. The inspected Zigbee reference archive contains no usable IQ samples in its SigMF data members. IEEE 802.15.4/6LoWPAN data is kept separate and is not relabeled as verified Zigbee.”



\## 6. Conclusion

“RFRecon\_AI demonstrates the saved-capture analysis workflow, feature extraction, four-class prediction, visualization, and invalid-input validation. Further acquisition verification and real-data evaluation are required before claiming validated five-class real-world performance.”
