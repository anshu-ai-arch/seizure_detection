# Project Architecture Blueprint: CNN-Informer Reproduction

## A. Project Objective and Scope

### Research Goal
The primary objective of this project is to create a publication-oriented, reproducible implementation of the research paper: **“CNN-Informer: A hybrid deep learning model for seizure detection on long-term EEG.”** 

### Intended Reproduction Workflow and Compute Arrangement
- **Code Development and Version Control**: Local Mac environment using Antigravity IDE, version-controlled via GitHub.
- **Compute and Data Access**: Kaggle Notebooks/Scripts will serve as the execution environment. The CHB-MIT dataset is accessed directly via Kaggle's input datasets to avoid local downloads or checking large datasets into version control.

### Boundaries of the Current Phase
This phase focuses exclusively on the foundational software architecture and rigorous reproduction of the paper's claimed methodology.

#### Clarification of Details
1. **Details Verified from the Original Paper**: To be determined (TBD) upon full literature review. The core architecture conceptually involves a CNN for local feature extraction and an Informer (attention mechanism) for long-sequence dependency modeling.
2. **Details Known from Existing Notes**: The repository is currently empty, meaning all foundational code and structural notes are to be established in this document.
3. **Details Requiring Verification**: 
   - Preprocessing steps (filtering, artifact removal, channel selection).
   - Windowing strategy (overlap, duration, labeling criteria).
   - Exact CNN architecture (layers, kernel sizes, channels) and Informer dimensions (heads, layers, sequence length).
   - Data split proportions (train/val/test ratios) and exact partitioning rules for the patient-specific evaluation.
   - Loss functions, optimization schedules, and handling of severe class imbalance.
   - Specific definitions for reported metrics (Sensitivity, Specificity, Accuracy, FPR/h).
4. **Proposed Engineering Decisions Needing Approval**:
   - Configuration-driven experiment management (YAML/TOML).
   - Decoupled Kaggle-GitHub workflow.
   - Strict separation of preprocessing from model training to prevent data leakage.

---

## B. Proposed Repository Structure

A clean, modular Python project layout is proposed to support both the initial reproduction and future exploratory research.

```text
project-root/
├── README.md                 # Project overview and quickstart
├── pyproject.toml            # Dependencies and project metadata
├── .gitignore                # Git exclusions (data, outputs, logs)
├── configs/                  # YAML/TOML configuration files
│   ├── data/                 # Data paths, splits, dataset definitions
│   ├── model/                # Architecture hyperparameters
│   ├── training/             # Optimizer, loss, epochs, batch size
│   └── experiments/          # Composed configurations for specific runs
├── docs/                     # Documentation and notes
│   ├── PROJECT_ARCHITECTURE.md (This file)
│   ├── RESEARCH_PROTOCOL.md  # Detailed verification of paper claims
│   ├── EXPERIMENT_LOG.md     # Tracking of run results and insights
│   └── decisions/            # Architecture Decision Records (ADRs)
├── src/
│   └── seizure_detection/    # Main package
│       ├── data/             # Dataset classes, split definitions
│       ├── preprocessing/    # Filtering, channel mapping, EDF reading
│       ├── windowing/        # Signal slicing and label assignment
│       ├── models/           # PyTorch modules (CNN, Informer, Hybrid)
│       ├── losses/           # Objective functions and weighting
│       ├── training/         # Training loops, validation, checkpoints
│       ├── evaluation/       # Metric computation, threshold tuning
│       ├── experiments/      # Orchestration of end-to-end runs
│       └── utils/            # Logging, reproducibility seeding, I/O
├── scripts/                  # Entry points for training/eval (Kaggle compatible)
├── tests/                    # Test suite
│   ├── unit/                 # Independent function/class tests
│   ├── integration/          # Pipeline tests (data -> model -> loss)
│   └── research_validity/    # Tests explicitly checking for leakage/overlap
└── outputs/                  # Ignored by git; local storage for logs/checkpoints
    └── .gitkeep
```

*Note: This structure isolates data handling from model definition, ensuring the preprocessing pipeline can be tested and frozen before training begins.*

---

## C. Data Contracts and Traceability

To ensure scientific rigor, explicit metadata tracking and clear interfaces between modules are required:

- **EDF Recording Reader**: Outputs raw signals and metadata (Patient ID, Recording ID, start time, sampling rate, physical channels).
- **Seizure Annotation Parser**: Outputs a list of verified seizure events (start/end times in seconds) per recording.
- **Channel Mapping and Preprocessing**: Accepts raw signals and target channel lists. Outputs filtered, standardized tensors of shape `(selected_channels, time)`. Missing channels, inconsistent ordering, or unreadable EDFs must raise explicit exceptions (e.g., `MissingChannelError`), not be silently imputed. NaNs or infinite values must log a warning and either be explicitly dropped or trigger a recording exclusion, per approved protocol.
- **Window Generation and Labeling**: Outputs discrete tensors. Every window must carry a metadata dictionary: `{patient_id, recording_id, window_start_sec, window_end_sec, label, split_assignment}`.
- **Dataset Split Manifest**: A JSON/YAML file explicitly listing which recordings/patients belong to `train`, `val`, and `test`.
- **PyTorch Dataset/DataLoader**: Yields `(input_tensor, label, metadata)` tuples.
- **Model Input and Output**: Expects `(B, C, S)` tensors; outputs logits of shape `(B, num_classes)`. *(Note: All tensor shapes, class counts, and output-head choices are provisional until explicitly verified against the original paper.)*
- **Evaluation and Reporting**: Consumes logits, true labels, and metadata to produce aggregate metrics and segment-level/event-level reports.

## D. Configuration and Experiment Lifecycle

Configurations will be managed via simple, hierarchical **YAML files**. 

- **Verified Baseline**: `configs/experiments/baseline_paper.yaml` will strictly define the verified reproduction settings.
- **Future Variants**: Independent YAML files (e.g., `baseline_attention_variant.yaml`) will define future experiments.
- **Validation**: Configurations will be parsed and validated using a schema (e.g., Pydantic or OmegaConf structured configs) to catch invalid parameters immediately.
- **Experiment Traceability**: 
  - Every run generates a unique Run ID (e.g., `YYYYMMDD_HHMMSS_runname`).
  - The run directory will store a resolved, immutable copy of the configuration, the Git commit hash, working-tree status (clean/dirty), random seeds, and software versions.
- **Recovery**: Interrupted training can resume by explicitly pointing to a checkpoint file in the run directory (e.g., `--resume outputs/runs/run_id/checkpoint_last.pt`). The configuration from the resumed run directory is loaded to guarantee consistency.

## E. Dataset Splits and Scientific Leakage Prevention

Split membership must be explicit, auditable, and defined prior to windowing.

- **Identity Preservation**: Patient, recording, window, and annotation identities are preserved through the entire pipeline via metadata dictionaries.
- **Split Strategy & Ordering**: The evaluation is verified to be patient-specific, but the exact split proportions (train/val/test) are **blocked**. The ordering of splitting and window generation must exactly follow the verified paper protocol.
- **Leakage Prevention**: Explicit leakage checks must be implemented for overlapping windows, patient/recording identity, and any required temporal separation. Overlapping windows must never bridge `train` and `val/test` sets.
- **Stateful Preprocessing**: Normalization statistics (mean, std) must be computed *only* on the `train` split and applied to `val/test`.
- **Blinded Test Set**: The test set remains completely untouched until final evaluation. Model selection and threshold tuning occur exclusively on the validation set.
- **Exclusion Policy**: Explicit approval and audit logging are required before dropping or excluding any recordings, channels, or windows due to invalid values or suspected artifacts.

## F. Testing and Acceptance Criteria

A staged testing plan will validate engineering robustness independent of scientific outcomes:

1. **Configuration parsing and validation**: Passes if invalid YAMLs raise schema errors and valid ones instantiate correctly.
2. **Synthetic EEG tensor and annotation tests**: Passes if dummy EDF-like data correctly flows through the dataset classes.
3. **Window-boundary and labeling tests**: Passes if edge-case windows (partially overlapping seizures) are assigned labels matching the exact approved rules.
4. **Split-integrity and leakage tests**: Passes if `train ∩ val ∩ test = ∅` for patient/recording IDs, and no window from one split leaks into another.
5. **Model tensor-shape and gradient-flow tests**: Passes if dummy tensors propagate through the network and gradients update all intended weights without NaNs.
6. **Checkpoint save/load tests**: Passes if a loaded model produces identical outputs to its pre-saved state.
7. **Metric calculation tests**: Passes if Sens/Spec functions return expected manual calculations on small, hand-crafted label arrays.
8. **Synthetic-Data End-to-End Smoke Test**: Passes if a 2-epoch run using purely synthetic dummy data completes without crashing.
9. **Real-EEG Pipeline Validation**: Testing on the real CHB-MIT dataset must not start until the relevant annotation, split, and preprocessing rules are explicitly approved. Must pass all above stages before full training commences.

## G. Kaggle Execution and Recovery

- **Workflow Constraints**: 
  - Development occurs locally (Mac + Antigravity).
  - Version control relies on GitHub.
  - Execution occurs on Kaggle utilizing free GPU/CPU resources.
  - Raw EEG data is never downloaded locally nor committed to GitHub.
- **Kaggle Data Handling**: Kaggle's `/kaggle/input/` is treated as strictly read-only. Dataset paths are exposed as YAML configurable parameters, not hardcoded.
- **Artifact Export**: Because `/kaggle/working/` is ephemeral, required artifacts (checkpoints, JSON metrics, PNG plots) must be explicitly exported or saved to a persistent Kaggle dataset/GCS bucket.
- **Environment**: Dependency versions and runtime constraints (e.g., PyTorch version) will be strictly pinned in `pyproject.toml` or `requirements.txt`.

## H. Storage, Git, and Security

- **`.gitignore`**: Strictly ignores raw data (`*.edf`), caches, logs, checkpoints (`*.pt`, `*.pth`), temporary files, and local secrets (e.g., `.env`).
- **Committed Artifacts**: Only source code, YAML configs, documentation (`docs/`), small metric summaries, and data split manifests (JSON/CSV) may be committed.
- **External Storage**: Large artifacts (checkpoints, massive logs) require Kaggle output storage or external buckets.
- **Security**: Tokens (like GitHub or Kaggle API keys) must be injected via environment variables/secrets. They must never appear in configurations, notebooks, logs, or Git history.
- **Export Process**: Artifacts moving from Kaggle to GitHub must be reviewed for size limits (<50MB) and data privacy (no patient data) before committing.

## I. Observability and Failure Handling

- **Consistent Logging**: Structured logging (e.g., Python's `logging` module) will track dataset inventory, preprocessing steps, and split distributions.
- **Error Handling**: Missing data or dimension mismatches will throw explicit custom exceptions (`DataLeakageError`, `ChannelMismatchError`) rather than generic index errors.
- **Experiment Output**: Every completed or failed run will leave a trail: `run_config.yaml`, `logs.txt`, and `errors.json` (if applicable), easily identifying the Git revision and dataset manifest used.

## J. Future Research Extensibility

- **Extensibility**: The architecture relies on standard, decoupled PyTorch `nn.Module` instances and configurable data pipelines.
- **Modifying Components**: To change the CNN or Informer, one simply points the YAML config to a new module class (e.g., `model._target_: seizure_detection.models.AttentionVariant`).
- **Baseline Integrity**: Future experiments must use *new* configuration files and *new* module classes (if logic changes). The original verified baseline code must not be silently mutated. We avoid complex plugin frameworks in favor of simple dependency injection via configuration.
