# Research Protocol: CNN-Informer Reproduction

## 1. Research Objective and Scope
This document outlines the source-verified protocol for reproducing the results from the paper: **“CNN-Informer: A hybrid deep learning model for seizure detection on long-term EEG.”** It establishes the exact methodological details extracted from the paper, identifies critical gaps requiring researcher approval, and defines the criteria for faithful reproduction on the CHB-MIT dataset.

## 2. Primary Source and Source Hierarchy
- **Primary Source**: The original PDF `referenceeeg.pdf` (CNN-Informer: A hybrid deep learning model for seizure detection on long-term EEG, Li et al., 2024, Neural Networks).
- **Secondary Source**: None used for methodology. A separate survey PDF (`seizure detectiion2 .pdf`) exists but is explicitly excluded as a source of truth for this reproduction.

## 3. Dataset and Patient Selection
- **Dataset**: CHB-MIT Scalp EEG Dataset. `VERIFIED — EXPLICIT` (Page 3, Sec 3.1)
- **Included Patients/Recordings**: 22 cases evaluated. `VERIFIED — EXPLICIT` (Table 1, Page 3)
- **Excluded Patients**: chb06 and chb16 are explicitly excluded "due to their seizure events being mostly shorter than 15 s." `VERIFIED — EXPLICIT` (Page 3, Sec 3.1)
- **Total Seizures**: 198 epileptic seizure events, total duration 11,536 s. `VERIFIED — EXPLICIT` (Page 3, Sec 3.1)
- **Handling of Unlisted/Corrupted EDFs**: `NOT SPECIFIED IN PAPER`.

## 4. Channel Mapping and Preprocessing
- **Selected Channels**: 18 common channels: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2, FZ-CZ, and CZ-PZ. `VERIFIED — EXPLICIT` (Page 3, Sec 3.1)
- **Sampling Rate**: 256 Hz. `VERIFIED — EXPLICIT` (Page 3, Sec 3.1)
- **DWT Preprocessing**: Daubechies-4 (Db4) mother wavelet. Decomposed into 5 scales. Reconstructed using cD5, cD4, cD3, and cA5 (corresponding to 0–32 Hz). `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.1)
- **Additional Filtering/Artifact Removal**: None stated beyond DWT. `NOT SPECIFIED IN PAPER`.

## 5. Windowing and Label Generation
- **Window Duration**: 4 seconds (1024 sampling points). `VERIFIED — EXPLICIT` (Page 3, Sec 3.2.1)
- **Overlap Strategy**: 50% overlap for seizure EEG recordings; 0% overlap (non-overlapping) for non-seizure EEG recordings. `VERIFIED — EXPLICIT` (Page 3, Sec 3.2.1)
- **Labeling of Boundary Windows**: (Windows crossing onset/offset) `NOT SPECIFIED IN PAPER`.
- **Guard Intervals**: `NOT SPECIFIED IN PAPER`.

## 6. Train/Validation/Test Protocol
- **Evaluation Paradigm**: Patient-specific (intra-patient). `VERIFIED — EXPLICIT` (Page 12, Sec 6)
- **Partitioning Procedure / Split Ratio**: Exact train/val/test split methodology or ratios are absent. Table 4 reports the duration of the test set per patient, but how this split was chosen (e.g., chronological, random, leave-one-out) is missing. `NOT SPECIFIED IN PAPER`.
- **Hyperparameter Selection & Validation Set**: `NOT SPECIFIED IN PAPER`.
- **Random Seeds / Repeated Runs**: `NOT SPECIFIED IN PAPER`.

## 7. CNN-Informer Architecture
- **CNN Layer 1**: Conv1D(18 filters, kernel 4, stride 2), Batch Normalization, ELU, Maxpooling1D(kernel 2, stride 2). `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2 & Fig 4)
- **CNN Layer 2**: Conv1D(18 filters, kernel 4, stride 2), Batch Normalization, ELU, Maxpooling1D(kernel 2, stride 2). `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2 & Fig 4)
- **CNN Layer 3**: Conv1D(64 filters, kernel 3, padding 1), Batch Normalization, ELU, Dropout. (No pooling). `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2 & Fig 4)
- **CNN Output Shape**: (64, 63) where 64 is channels and 63 is sequence length. `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2)
- **Informer Encoder**: 3 ProbSparse self-attention blocks. 2 self-attention distilling operations (between block 1-2 and 2-3). `VERIFIED — EXPLICIT` (Page 6, Sec 3.2.2 & Fig 5)
- **ProbSparse Sampling Factor**: $c = 3$. `VERIFIED — EXPLICIT` (Page 10, Sec 5.2.2)
- **Output Layer**: Fully connected layer followed by Softmax mapping. `VERIFIED — EXPLICIT` (Page 6, Algorithm 1)
- **Ambiguities**: Dropout probability value, exact Informer embedding dimensions (though Fig 5 shows 64), and number of attention heads are not explicitly stated in the text. `AMBIGUOUS / CONFLICTING`.

## 8. Training Configuration
- **Optimizer**: Adam. `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2)
- **Learning Rate**: $10^{-4}$. `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2)
- **Batch Size**: 32. `VERIFIED — EXPLICIT` (Page 4, Sec 3.2.2)
- **Epoch Count**: 100 epochs mentioned in parameter analysis. `VERIFIED — DERIVED` (Page 10, Sec 5.2.2)
- **Loss Function**: `NOT SPECIFIED IN PAPER`. (Cross-entropy is standard but unstated).
- **Scheduler, Weight Init, Checkpoint Selection**: `NOT SPECIFIED IN PAPER`.

## 9. Post-Processing
- **Moving Average Filter (MAF)**: Applied to output scores. Length $2N+1$. `VERIFIED — EXPLICIT` (Page 7, Eq 5)
- **Thresholding**: Binary decision threshold applied after MAF. `VERIFIED — EXPLICIT` (Page 7, Sec 3.2.3)
- **Collar Technique**: Extends detected seizure by $K$ points at beginning and end. `VERIFIED — EXPLICIT` (Page 7, Sec 3.2.3)
- **Parameters**: Exact values for $N$, Threshold, and $K$ are "adjusted for each patient" but specific values are `NOT SPECIFIED IN PAPER`.

## 10. Evaluation Metrics
- **Segment-Based Metrics**: Sensitivity, Specificity, Accuracy. `VERIFIED — EXPLICIT` (Page 7, Eqs 6-8)
- **Event-Based Metrics**: Sensitivity, False Detection Rate (FDR /h), Latency (s). `VERIFIED — EXPLICIT` (Page 7, Sec 4)

## 11. Paper-Reported Reference Results
*Note: These are reference benchmarks for the CHB-MIT dataset. We do not claim reproduction until our implementation matches these.*
- **Segment-Based Average**: Sens: 99.54%, Spec: 98.55%, Acc: 98.54%. (Table 3, Page 8)
- **Event-Based Average**: Sens: 99.07%, FDR: 0.16 /h, Latency: 22.21 s. (Table 4, Page 8)

## 12. Reconciliation with Our Dataset Audit
*Note: Previous CHB-MIT audits were performed on Kaggle (e.g., `/kaggle/working/chbmit_master_audit_v3/`). They are not locally available but must be consulted during Kaggle execution.*
- **Excluded Patients**: The paper explicitly excludes chb06 and chb16. **Action**: We will exclude these.
- **Unlisted EDFs (e.g., chb24_*.edf)**: The paper does not mention them. **Action**: `REQUIRES RESEARCHER APPROVAL` on whether to include or discard them.
- **Channel Mismatches & High-Amplitude Artifacts**: The paper uses 18 specific channels but does not detail how missing channels or severe artifacts within those 18 are handled. **Action**: `REQUIRES RESEARCHER APPROVAL` for an exclusion/imputation policy.

## 13. Unresolved Questions and Researcher Decisions
The following items block faithful reproduction and require explicit researcher approval to proceed with real-EEG training:
1. **Data Split Protocol**: How to divide each patient's data into train/val/test, and whether to preserve temporal order.
2. **Loss Function**: Choice of loss function (e.g., CrossEntropyLoss).
3. **Window Boundary Labels**: How to label a 4s window that partially overlaps a seizure.
4. **Data Exclusion Policy**: How to handle corrupted EDFs, missing channels among the required 18, and extreme artifacts.
5. **Post-Processing Parameters**: Heuristics for selecting MAF length, threshold, and Collar $K$ per patient.

## 14. Implementation Readiness Checklist

| Area | Status | Evidence / unresolved issue | Next action |
| ---- | ------ | --------------------------- | ----------- |
| Dataset and patient selection | `READY` | Excludes chb06, chb16. | Implement exclusion in dataloader. |
| Channel mapping | `REQUIRES RESEARCHER APPROVAL` | 18 channels verified. Handling of missing/corrupted channels unstated. | Define missing-channel policy. |
| DWT preprocessing | `READY` | Db4, 5 scales, reconstruct cD5, cD4, cD3, cA5. | Implement DWT pipeline. |
| Windowing and labels | `REQUIRES RESEARCHER APPROVAL` | 4s window, 50% overlap (seizure), 0% (non-seizure). Boundary logic missing. | Define boundary window logic. |
| Data splitting | `BLOCKED ON PAPER VERIFICATION` | Patient-specific. Train/test split rules are completely missing. | Formulate and approve a split strategy. |
| CNN architecture | `READY` | Exact layers, kernels, strides, channels verified. | Implement CNN module. |
| Informer architecture | `BLOCKED ON PAPER VERIFICATION` | ProbSparse c=3. Hidden dims, heads, and dropout rates are unstated. | Formulate/approve dimensions. |
| Training settings | `BLOCKED ON PAPER VERIFICATION` | Adam, LR $10^{-4}$, Batch 32, 100 epochs. Loss function missing. | Approve loss function. |
| Post-processing | `REQUIRES RESEARCHER APPROVAL` | MAF, threshold, collar mentioned. Parameters are patient-specific but undefined. | Define tuning heuristic. |
| Evaluation metrics | `READY` | Formulas explicitly provided in paper. | Implement metric functions. |
| Audit reconciliation | `REQUIRES RESEARCHER APPROVAL` | Kaggle audit outputs exist. Handling unlisted/artifact data is undefined. | Review Kaggle audit & approve policy. |
| Implementation readiness | `READY FOR ENGINEERING SCAFFOLDING` | Scientific reproduction is blocked, but software scaffolding can begin. | Build repo structure & synthetic tests. |

## 15. Source Evidence Index
- `referenceeeg.pdf`: The sole source of truth for all verified methodological details. All references (e.g., "Page X, Sec Y") refer to this document.
