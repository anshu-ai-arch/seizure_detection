# Preprocessing Protocol: CNN-Informer Reproduction

This document defines the exact preprocessing operations for reproducing the CHB-MIT pipeline from the CNN-Informer paper.

## Paper Fidelity Checks

| Paper requirement | Paper evidence/section | Existing implementation | Change made | Verification result |
|-------------------|------------------------|-------------------------|-------------|---------------------|
| Dataset: CHB-MIT | Sec 3.1 | Kaggle runner targets CHB-MIT. | Default dir updated. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| Exclude Chb6/16 | Sec 3.1 | Config yaml excludes them. | Excluded in yaml. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| 18 bipolar channels | Sec 3.1 | Channel mapping extracts 18 exactly. | Explicit exact equality duplicate rule for T8-P8. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| 256 Hz sampling | Sec 3.1 | Validated during load. | Metadata sfreq check added. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| 4-second segments | Sec 3.2.1 | `window_generator.py` maps 4s. | Validated array shapes. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| 50% overlap seizure | Sec 3.2.1 | `annotation_mapper.py` assigns overlap. | Dynamic stride implemented. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| 0% overlap non-seizure | Sec 3.2.1 | `annotation_mapper.py` | Dynamic stride implemented. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| Boundary handling | NOT_SPECIFIED_IN_PAPER | Pipeline sets UNRESOLVED_BOUNDARY. | Status maintained. | IMPLEMENTATION_VERIFIED_BUT_PAPER_DETAIL_UNSPECIFIED |
| DWT (db4, lvl 5) | Sec 3.2.1 | `process_multichannel_dwt` applies db4. | None. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| DWT retain cA5,cD5-3| Sec 3.2.1 | Zeroes cD1/cD2, waverec. | Ref implementation test added. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| Output representation| Sec 3.2.1 (Shape 18x1024)| `.npy` saved, float64. | Manifest includes full metadata. | PAPER_SPECIFIED_AND_IMPLEMENTED |
| Train/val/test split | NOT_SPECIFIED_IN_PAPER | Pipeline blocked before split. | - | NOT_YET_VERIFIED |

## Unresolved Protocol Decisions

The following details are not fully specified in the original paper and require researcher approval:
1. **Boundary Windows**: Do we discard windows crossing seizure boundaries (UNRESOLVED_BOUNDARY), or consider them seizure/non-seizure?
2. **Train/Val/Test Split Strategy**: Is the evaluation cross-patient (patient-wise split) or within-patient? If within-patient, is it recording-wise or randomly shuffled? (Random shuffling with overlapping windows causes severe train-test leakage).
