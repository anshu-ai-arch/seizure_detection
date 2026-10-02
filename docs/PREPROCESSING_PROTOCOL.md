# Preprocessing Protocol: CNN-Informer Reproduction

This document defines the exact preprocessing operations for reproducing the CHB-MIT pipeline from the CNN-Informer paper. Every operation's status determines whether it is ready for real EEG processing.

## Status Definitions
* `VERIFIED — EXPLICIT`: Explicitly stated in the primary source.
* `VERIFIED — DERIVED`: Inferred directly from an explicit detail.
* `NOT SPECIFIED IN PAPER`: Missing from the primary source.
* `AMBIGUOUS / CONFLICTING`: Source material is unclear or contradicts itself.
* `REQUIRES RESEARCHER APPROVAL`: An engineering decision blocking the pipeline.
* `NOT YET VERIFIED`: Needs further investigation.

## Preprocessing Stages

### A. Dataset and Recording Inventory
* **Operation**: Identify all recordings for the 22 target patients (excluding chb06, chb16).
* **Paper Spec**: 22 cases evaluated.
* **Source**: Page 3, Section 3.1, Table 1.
* **Status**: `VERIFIED — EXPLICIT`.
* **Validation**: Code must explicitly filter out excluded patients and log inventory.

### B. EDF Reading and Recording Metadata
* **Operation**: Load raw signals and sampling rates from EDF files.
* **Paper Spec**: 256 Hz sampling frequency, 16-bit resolution.
* **Source**: Page 3, Section 3.1.
* **Status**: `VERIFIED — EXPLICIT`.
* **Validation**: Verify loaded tensors match expected duration and frequency.

### C. Annotation Parsing and Seizure-Event Representation
* **Operation**: Parse summary files to determine seizure onset and offset times.
* **Paper Spec**: "onset and offset of each seizure were annotated by professional clinical physicians".
* **Source**: Page 3, Section 3.1.
* **Status**: `VERIFIED — EXPLICIT`.
* **Validation**: Test parser on synthetic summary text to confirm correct intervals.

### D. Channel-Name Normalization and Channel Selection
* **Operation**: Map and extract exactly 18 channels.
* **Paper Spec**: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2, FZ-CZ, CZ-PZ.
* **Source**: Page 3, Section 3.1.
* **Status**: `REQUIRES RESEARCHER APPROVAL` (The paper lists 18 channels but does not provide an explicit mapping for mismatched EDF names, nor a rule for missing channels).
* **Validation**: Raise error if channels are missing or mismatched, pending approval policy.

### E. Sampling Frequency and Resampling
* **Operation**: Ensure signals are 256 Hz.
* **Paper Spec**: CHB-MIT is natively 256 Hz. No resampling mentioned for this dataset.
* **Source**: Page 3, Section 3.1.
* **Status**: `VERIFIED — EXPLICIT`.

### F. Filtering and Baseline/Noise Handling
* **Operation**: Remove artifacts beyond DWT.
* **Paper Spec**: None mentioned.
* **Source**: N/A.
* **Status**: `NOT SPECIFIED IN PAPER`.
* **Validation**: No additional filtering implemented.

### G. Wavelet Decomposition and Band Reconstruction
* **Operation**: DWT filtering using Db4 mother wavelet.
* **Paper Spec**: Decompose into 5 scales (cD1-cD5, cA5). Reconstruct using cD5, cD4, cD3, and cA5 (0-32 Hz).
* **Source**: Page 4, Section 3.2.1.
* **Status**: `VERIFIED — EXPLICIT`.
* **Validation**: Input `(18, N)` -> Output `(18, N)` containing only targeted frequency bands.

### H. Window Generation and Overlap
* **Operation**: Slice continuous signals into 4-second discrete windows.
* **Paper Spec**: 4s window. 50% overlap for seizure EEG, 0% overlap for non-seizure EEG.
* **Source**: Page 3, Section 3.2.1.
* **Status**: `VERIFIED — EXPLICIT` (Shape: `18 x 1024`).
* **Validation**: Output shape assertions; overlap stride calculations.

### I. Seizure/Non-Seizure Window Labels
* **Operation**: Assign binary label to each window.
* **Paper Spec**: Missing logic for handling windows crossing onset/offset. Missing guard interval specifications.
* **Source**: N/A.
* **Status**: `REQUIRES RESEARCHER APPROVAL`.
* **Validation**: Pipeline explicitly flags boundary windows as `UNRESOLVED_BOUNDARY` and stops.

### J. Patient/Recording Split and Leakage Prevention
* **Operation**: Partition data into train/val/test.
* **Paper Spec**: Patient-specific evaluation. Train/test ratios and chronological vs random split unstated.
* **Source**: Page 12, Section 6.
* **Status**: `BLOCKED ON PAPER VERIFICATION` / `REQUIRES RESEARCHER APPROVAL`.
* **Validation**: Check for overlapping windows across partitions and patient mixing.

### K. Processed-Data Validation and Audit Reports
* **Operation**: Log all dropped windows, channels, and recordings.
* **Status**: `IMPLEMENTED` (Pending researcher approval of drop logic).
* **Validation**: `metadata_auditor.py` explicitly tracks and exports CSVs of recording statuses without silent deletion.
