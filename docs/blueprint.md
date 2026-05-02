# Implementation Blueprint: Persuasion Principle Detection and Explainability System

Target audience: AI programming assistant / engineering team
Version: 1.1
Status: Target architecture with partial implementation in this repo
Scope constraints: Multi-label bilingual classification, explainability, conditional generation, hardware-agnostic deployment, deterministic fallback, strict validation gates

Related conceptual document:
- `docs/system_foundations.md` explains the theoretical framing, premises, hypotheses, solution strategy, and high-level architecture in a single narrative.

## 0. Current repository reality check

This document describes the target architecture. The current repository only implements part of it.

Implemented today:
- Data harmonization from IWSPA and Spaphish
- Synthetic/assisted augmentation using a teacher model
- Classifier training for `microsoft/mdeberta-v3-base`
- SLM fine-tuning pipeline with LoRA/QLoRA logic
- ONNX and OpenVINO export for the classifier
- Shared runtime paths through `configs/architecture.yaml`
- Adaptive hardware tuning through the runtime `turbo` layer
- Paper-ready export of figures, tables, predictions, reports, and persisted splits

Not fully implemented today:
- The repo now uses `70/10/20` for train/validation/test in the active classifier pipeline, not the `70/15/15` target split described below
- Advanced stratification by joint label combination plus language
- Threshold optimization per class
- Probability calibration and ECE control
- Model selection/ensemble logic across multiple backbones
- Final held-out test evaluation pipeline
- Production inference API and monitoring stack

Important code-status note:
- The classifier training pipeline previously prepared train/eval splits but passed the wrong dataset object into `Trainer`. That has already been corrected in code.
- The SLM pipeline previously trained/evaluated on the same dataset and could incorrectly treat a CPU fallback checkpoint as a completed fine-tune. That has also been corrected in code.

## 1. Project objective and scope

Build a production-grade pipeline that:

1. Ingests raw email datasets from `IWSPA` (English) and `Spaphish` (Spanish).
2. Harmonizes, cleans, validates, and exports a static multi-label dataset.
3. Trains and selects an optimal multilingual encoder classifier from:
   - `microsoft/mdeberta-v3-base`
   - `FacebookAI/xlm-roberta-base`
4. Calibrates probabilities, optimizes per-class thresholds, and validates against strict F1 and ECE targets.
5. Integrates a constrained generative module for causal explanation and conditional text synthesis.
6. Exports artifacts for three runtime targets:
   - A100: BF16/CUDA
   - Intel Arc GPU: OpenVINO IR
   - CPU: ONNX FP32 or INT8
7. Deploys with deterministic hardware fallback, runtime monitoring, and drift detection.

Persuasion taxonomy (Ferreira et al., 2015):
- `AUTH`
- `SP`
- `LSD`
- `CRC`
- `DIS`

These are treated as 5 independent binary labels.

## 2. Directory structure

Current canonical runtime structure in this repository:
- `caches/downloads/`
- `caches/partials/`
- `outputs/artifacts/`
- `outputs/checkpoints/`
- `outputs/splits/`
- `outputs/results/models/`
- `outputs/results/figures/`
- `outputs/results/tables/`
- `outputs/results/predictions/`
- `outputs/results/reports/`
- `logs/runs/`

Legacy top-level `models/` and `checkpoints/` are no longer part of the
active runtime contract and should not be treated as canonical outputs.

```text
persuasion-system/
├── data/
│   ├── IWSPA_AP_persuasion_annotated.csv
│   └── Spaphish dataset - DiB.csv
├── scripts/
│   ├── 01_data_harmonization.py
│   ├── 02_train_classifiers.py
│   ├── 03_calibrate_select.py
│   ├── 04_train_generative.py
│   └── 05_export_runtime.py
├── configs/
│   ├── training_defaults.yaml
│   ├── thresholds.json
│   └── runtime_detection.yaml
├── models/
│   ├── classifiers/
│   ├── generative/
│   └── exports/
├── logs/
├── outputs/
│   ├── dataset_persuasion_v1.jsonl
│   └── audit_report_phase1.txt
└── requirements.txt
```

## 2.1 Cross-cutting runtime contract: checkpoints, logs, and caches

Every executable pipeline phase must use the shared runtime layer in
`scripts/pipeline_utils.py`. This is mandatory for all current and future
scripts, including data harmonization, classifier training, calibration,
generative training, export, benchmarks, and inference services.

Runtime rules:

1. **Logging**
   - Each phase writes human-readable progress logs to `logs/<phase>.log`.
   - Long-running phases must log start/end of major steps and emit heartbeat
     progress messages while work is ongoing.
   - Logs must go to both console and file so terminal runs never appear dead.

2. **Checkpoints**
   - Each phase writes JSON manifests under `checkpoints/<phase>/`.
   - Manifests must record `running`, `completed`, or `failed` status, input
     signatures, output paths, timestamps, and error messages when applicable.
   - Long-running model training must also use native framework checkpoints,
     such as Hugging Face `checkpoint-*` directories, with automatic resume.
   - Completed stages may be skipped only when their checkpoint and expected
     output artifacts are present and input signatures still match.

3. **Atomic outputs**
   - JSON/JSONL/report outputs must be written atomically where practical:
     write to a temporary file first, then replace the final path.
   - This prevents corrupted final artifacts when a process is interrupted.

4. **Caches**
   - Internet downloads must use project-local cache directories under
     `caches/`.
   - The active repository contract now consolidates runtime downloads under
     `caches/downloads/`.
   - Partial generation artifacts must reuse `caches/partials/`.
   - Downloads occur lazily: fetch only when first needed, then reuse cache on
     subsequent runs.
   - Offline mode must be supported by setting the runtime config flag
     `offline: true`, which enables Hugging Face offline environment variables.

5. **Failure/restart behavior**
   - If a process fails after hours of work, rerunning the same command must
     resume from the last valid checkpoint whenever the underlying framework
     supports it.
   - Commands may expose explicit override flags such as `--force`,
     `--force-splits`, or `--no-resume`, but the default behavior must protect
     prior work.

## 3. Phase 1: Data harmonization pipeline

### 3.1 Input specifications

| Source | File | Language | Key columns |
|---|---|---|---|
| IWSPA | `data/IWSPA_AP_persuasion_annotated.csv` | EN | `subject`, `body`, `AUTH`, `SP`, `LSD`, `CRC`, `DIS` |
| Spaphish | `data/Spaphish dataset - DiB.csv` | ES | Raw text block, semicolon-delimited labels, justification field |

### 3.2 Processing rules

1. **IWSPA parsing**
   - Read columns directly.
   - Preserve label order as `[AUTH, SP, LSD, CRC, DIS]`.

2. **Spaphish parsing**
   - Extract labels using regex:
     - `PRINCIPIO:\s*\[?(SI|NO)\]?;?\s*(\d)`
   - If regex fails, attempt positional parsing by semicolon split.
   - If both fail, mark row as `ambiguous` and exclude it.
   - Extract justification field:
     - If justification contains `"omitida"`, `"no capturada"`, or is empty, store `null`.
     - Otherwise, parse into a dictionary such as:
       - `{"AUTH": "...", "LSD": "..."}`
   - Validate that every principle mentioned in the justification has label value `1`.
   - If justification and labels disagree, either:
     - exclude the row, or
     - flag it for manual review, depending on pipeline mode

3. **Text normalization**
   - Build canonical text as:
     - `"[SUBJECT] {subject} [BODY] {body}"`
   - If `subject` is empty:
     - `"[BODY] {body}"`
   - Remove:
     - zero-width spaces
     - soft hyphens
     - invisible control characters
   - Preserve:
     - URLs
     - domains
     - email addresses
     - technical tokens
   - Collapse repeated whitespace and newlines into a single space.

4. **Language detection**
   - Apply a heuristic stopword-ratio detector.
   - If `ratio_ES > 0.7`, assign `es`.
   - If `ratio_EN > 0.7`, assign `en`.
   - Otherwise, assign `mixed`.

5. **Output schema**
   - Export to `outputs/artifacts/harmonized_dataset.jsonl`
   - One JSON object per line:

```json
{
  "id": "sha256_hash_of_original_row",
  "text": "[SUBJECT] ... [BODY] ...",
  "labels": [1, 0, 1, 0, 0],
  "language": "es",
  "justification": {
    "AUTH": "...",
    "LSD": "..."
  },
  "source": "Spaphish"
}
```

### 3.3 Validation gates

These are hard stops.

- **Row loss tolerance**
  - Fewer than 2 percent of rows may be dropped.
  - If exceeded, abort and audit the Spaphish parser.

- **Label integrity**
  - Every retained row must contain exactly 5 binary values.
  - Invalid rows must be dropped.

- **Language ratio preservation**
  - The ES/EN distribution in the output must not deviate by more than 5 percent from the input.
  - If exceeded, review the language heuristic.

- **Audit report**
  - Generate `audit_report_phase1.txt` with:
    - total row counts
    - retained rows
    - dropped rows
    - loss reasons
    - label distribution
    - language distribution
    - justification availability

## 4. Phase 2: Classifier training and validation

### 4.1 Dataset split

- **Stratification key**
  - 5-bit label combination vector plus detected language

- **Split ratios**
  - 70 percent train
  - 10 percent validation
  - 20 percent test

- **Rare combinations**
  - Any combination with fewer than 10 occurrences is assigned entirely to train.
  - Validation and test metrics are reported per principle, not per rare combination.

- **Random seed**
  - Fixed seed: `42`

### 4.2 Architecture and hyperparameters

| Component | Specification |
|---|---|
| Backbones | `microsoft/mdeberta-v3-base`, `FacebookAI/xlm-roberta-base` |
| Classification head | 5 independent sigmoid outputs |
| Learning setting | Multi-label BCE with logits |
| Max length | 256 tokens |
| Learning rate | `3e-5` |
| Batch size | A100: `32`, CPU: `4` |
| Accumulation | Use gradient accumulation if memory-constrained |
| Precision | A100: `bf16`, CPU: `fp32` |
| Max epochs | `6` |
| Early stopping | patience=`2`, monitor=`eval_loss`, restore best |

### 4.3 Loss function

Use weighted binary cross-entropy with logits:

```text
Loss = Σ(i=1..5) [ w_i * BCEWithLogits(logits_i, label_i) ]
w_i = 1 / sqrt(frequency_i)
```

Rules:

- Compute `w_i` from label frequencies in the training split only.
- Do not normalize the weights.
- Apply weights independently per label head.

### 4.4 Validation metrics

Mandatory metrics:

- F1-macro
- F1 per class
- AUC-PR per class
- ECE per class
- Multi-label consistency ratio

Primary model selection criterion:

- `F1-macro` on validation

Hard validation gate:

- If `F1-macro(validation) < 0.65` after epoch 3, pause training and audit:
  - loss weighting
  - class imbalance
  - parser quality
  - label noise

## 5. Phase 3: Selection, calibration, and thresholding

### 5.1 Model selection logic

1. If only one model reaches `F1-macro(validation) >= 0.70`, select that model.
2. If both models reach `F1-macro(validation) >= 0.70`:
   - Compute an error-overlap matrix on the validation set.
   - If disjoint errors exceed 30 percent, use a soft-voting ensemble with averaged calibrated probabilities.
   - If overlap exceeds 70 percent, select the model with higher `F1-macro`.
3. If neither reaches `0.70`, halt and trigger:
   - augmentation review
   - loss review
   - data quality review

### 5.2 Threshold optimization

- Generate a precision-recall curve for each principle on validation.
- Select threshold `t_i` that maximizes per-class F1.
- Do not use a global threshold of `0.5`.
- Store thresholds in `configs/thresholds.json`.

### 5.3 Probability calibration

- Measure ECE per principle.
- If `ECE > 0.15` for a given head, apply `IsotonicRegression` on validation probabilities for that head.
- Freeze calibration parameters after fitting.
- Validate that calibrated test ECE remains below `0.15`.

### 5.4 Final evaluation

Run the selected model or ensemble on the held-out test set.

Report:

- F1-macro
- F1 per class
- AUC-PR per class
- ECE per class
- ES/EN performance gap

Hard acceptance criteria:

- `F1-macro >= 0.70`
- `ECE < 0.15`
- `consistency >= 0.85`

## 6. Phase 4: Generative module and cross-verification

### 6.1 Base model and training

- **Backbone**
  - Multilingual SLM in the 1.5B to 3B range
  - Example: `Qwen/Qwen2.5-3B-Instruct`

- **Training method**
  - LoRA with:
    - `r = 16`
    - `alpha = 32`
    - `target_modules = ["q_proj", "v_proj"]`

- **Training data**
  - Real justifications from Spaphish
  - Filtered synthetic examples that pass consistency checks

- **Loss**
  - Cross-entropy
  - Plus consistency penalty:
    - if generated principle `p` is present but classifier probability for `p` is below threshold `t_p`, add `λ * margin_loss`

### 6.2 Input-output contract

**Prompt template**

```text
[TEXT] {email_text}
[PROBABILITIES] AUTH:{p_auth}, SP:{p_sp}, LSD:{p_lsd}, CRC:{p_crc}, DIS:{p_dis}
[DEFINITIONS] {Ferreira_2015_operational_definitions}
[INSTRUCTION] Analyze the text and return strictly valid JSON.
```

**Expected output**

```json
{
  "detected_principles": ["AUTH", "LSD"],
  "confidence_scores": {
    "AUTH": 0.82,
    "LSD": 0.71,
    "SP": 0.11,
    "CRC": 0.08,
    "DIS": 0.15
  },
  "reasoning": {
    "AUTH": "The text invokes a legitimate-sounding institutional source.",
    "LSD": "The text uses familiarity cues and affiliation framing."
  },
  "generated_text": "..."
}
```

Rules:

- `generated_text` is included only when conditional synthesis is requested.
- Output must satisfy a strict JSON schema.
- Use constrained decoding or schema-guided generation.

### 6.3 Cross-verification engine

1. Parse JSON output.
2. Compare `detected_principles` against classifier thresholds.
3. If a principle is present in `detected_principles` but its confidence is below threshold, reject the output.
4. Regenerate with a corrective prompt such as:

```text
The classifier indicates low evidence for {principle}. Either justify the principle with stronger evidence or omit it.
```

Approval rule:

- If consistency across validation examples is at least `0.85`, approve the generative module.
- Otherwise, freeze the generative component and review penalty weight `λ`.

## 7. Phase 5: Multi-backend export and runtime

### 7.1 Export targets

| Target | Format | Precision | Toolchain |
|---|---|---|---|
| A100 | PyTorch `.pt` or TensorRT `.engine` | `bf16` / `fp16` | `torch.export`, `trtexec` |
| Intel Arc GPU | OpenVINO IR `.xml` + `.bin` | `fp16`, optional `int8` | Model Optimizer or equivalent OpenVINO export flow |
| CPU | ONNX `.onnx` | `fp32`, optional `int8` | `torch.onnx.export`, `onnxruntime` |

### 7.2 Hardware detection and fallback logic

```yaml
detection_order:
  - check: cuda_available() && vram_gb >= 8
    route: A100_BF16
  - check: openvino_runtime.has_device("GPU")
    route: INTEL_GPU_FP16
  - check: openvino_runtime.has_device("CPU") || cpu_supports_avx2()
    route: CPU_FP32_INT8
```

Runtime constraints:

- On NUC 14 Pro:
  - force `batch_size = 1`
  - monitor thermal throttling
  - pause sustained runs longer than 3 minutes if thermal thresholds are exceeded

- NPU:
  - explicitly excluded from deployment routing

Post-conversion verification:

- Run a 100-sample benchmark on each backend.
- If divergence exceeds 5 percent between backends, abort export and review:
  - operator mapping
  - tokenizer parity
  - preprocessing parity
  - numeric precision drift

## 8. Phase 6: Inference API and production monitoring

### 8.1 API contract

**Endpoint**
- `POST /analyze`

**Request**

```json
{
  "text": "..."
}
```

**Response**

```json
{
  "principles": ["AUTH", "LSD"],
  "probabilities": {
    "AUTH": 0.82,
    "SP": 0.11,
    "LSD": 0.71,
    "CRC": 0.08,
    "DIS": 0.15
  },
  "reasoning": {
    "AUTH": "...",
    "LSD": "..."
  },
  "hardware_used": "A100_BF16",
  "latency_ms": 142
}
```

Runtime rule:

- Timeout limit: `5000 ms`
- If GPU inference stalls, fall back gracefully to CPU.

### 8.2 Monitoring and drift detection

| Metric | Threshold | Action |
|---|---|---|
| Data drift (KL divergence) | `> 0.1` vs training distribution | Flag, log, and prepare retraining |
| Confidence shift | Median probability drops by more than 15 percent or variance spikes | Audit incoming text distribution |
| Human sampling | Review 5 percent random sample | If precision drops by more than 5 percent vs test, trigger retraining |
| Threshold recalibration | At least 1000 new annotated samples | Update `thresholds.json` and re-evaluate ECE |

### 8.3 Logging schema

```json
{
  "timestamp": "ISO8601",
  "input_hash": "sha256",
  "hardware": "A100|GPU|CPU",
  "precision": "bf16|fp16|fp32",
  "latency_ms": 142,
  "predicted": ["AUTH", "LSD"],
  "probabilities": {
    "AUTH": 0.82,
    "LSD": 0.71
  },
  "drift_score": 0.03
}
```

## 9. Hard constraints and acceptance criteria

| Constraint | Value | Verification |
|---|---|---|
| Data loss | `< 2%` | `audit_report_phase1.txt` |
| F1-macro (test) | `>= 0.70` | Final evaluation script |
| F1 per class | `>= 0.65` for `AUTH`, `LSD`, `DIS`; `>= 0.50` for `SP`, `CRC` | Per-class report |
| ECE (test) | `< 0.15` | Calibration report |
| Consistency (classifier ↔ generator) | `>= 0.85` | Cross-verifier log |
| Export divergence | `< 5%` between backends | Benchmark diff log |
| Runtime fallback | Deterministic A100 → GPU → CPU | Runtime detection log |

Failure protocol:

- If any hard constraint is violated:
  1. halt deployment
  2. log root cause
  3. adjust loss weights, thresholds, calibration, or dataset composition
  4. re-run validation
  5. do not proceed to production until all gates pass

## 10. Execution checklist

### Phase 1
- Implement CSV parsers
- Implement text normalization
- Implement language detection
- Implement JSONL export
- Generate audit report
- Verify row loss remains below 2 percent

### Phase 2
- Implement stratified split
- Configure dual training pipelines
- Apply inverse-frequency weighted BCE
- Log F1 and ECE per epoch

### Phase 3
- Implement model-selection logic
- Generate precision-recall curves
- Optimize thresholds
- Apply isotonic calibration if `ECE > 0.15`
- Evaluate on held-out test set

### Phase 4
- Set up LoRA training for the generative model
- Enforce strict JSON schema through constrained decoding
- Implement cross-verification engine
- Validate consistency at or above `0.85`

### Phase 5
- Export to:
  - PyTorch or TensorRT
  - OpenVINO IR
  - ONNX
- Verify post-conversion divergence remains below 5 percent
- Implement deterministic hardware router

### Phase 6
- Build REST or gRPC inference endpoint
- Implement drift and confidence monitoring
- Set up human review sampling
- Deploy monitoring dashboard

### Final audit
- Verify that all hard criteria are satisfied
- Document any deviations
- Freeze configuration for deployment

## 11. Current export contract for paper assets

The current implementation now writes reusable research artifacts automatically on every fresh run:

- `outputs/results/figures/`
  - English-language figures in `PNG` and `EPS`
  - companion `CSV` files with the plotted source data
- `outputs/results/tables/`
  - split summaries
  - training history
  - class metrics
  - summary metrics
  - threshold sweeps
  - multilabel confusion summaries
- `outputs/results/predictions/`
  - validation and test predictions with gold labels, scores, and binarized outputs
- `outputs/results/reports/`
  - JSON manifests and machine-readable summaries
- `outputs/splits/`
  - persisted train, validation, test, SLM-train, and SLM-eval splits

Practical note for continuation:

- The repository is now in a pre-final-run state.
- Old runtime outputs were intentionally cleaned so the next complete execution can become the canonical experiment for the paper.
- The recommended full command is:
  - `python main.py --fresh-all --fresh-slm --slm-run-tag paper_run_YYYYMMDD`
- The next engineering checkpoint after that run should be:
  - verify generated manifests
  - inspect summary metrics
  - decide whether ECE/calibration must be added before freezing paper results

Webapp continuity note:

- a Spanish-first web application prototype already exists under `webapp/`
- it currently serves as a product and UX validation layer
- real integration with classifier, SLM, and research outputs is intentionally deferred until after the current training cycle completes

Current strategic decision for this workstream:

- Finish the current full run first.
- Do not perform a large anti-leakage pipeline refactor before that run.
- Treat the classifier as the primary quantitative result.
- Treat the SLM as a support/explainability/generative component for the current iteration.
- Do not overclaim the current SLM split as strong leakage-free generalization evidence.

Deferred-for-next-iteration paper hardening tasks:

- split-first data flow across the full pipeline
- stricter leakage control for SLM training data
- deduplication and near-duplicate checks
- ECE and explicit calibration layer
- stronger final evaluation protocol for the SLM
