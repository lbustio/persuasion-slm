# <img src="webapp/logo.png" width="48" align="center"> Persuasion SLM

**Persuasion SLM** is a research platform for detecting and analyzing persuasion principles in text (emails, SMS, phishing messages). It combines a **multi-label classifier** (mDeBERTa / XLM-RoBERTa) with a **fine-tuned Small Language Model** (Qwen 2.5) to provide detection scores, human-like reasoning, textual evidence, and interactive conversation about analyzed messages.

Designed for **local-only execution** to ensure data privacy and sovereignty.

---

## Key Features

### Two-Stage Detection Architecture
- **Classifier (Encoder)**: Multi-label detection of 5 persuasion principles with quantitative scores
- **SLM (Decoder)**: Audits, explains, and converses about the message with textual evidence

### Persuasion Principles Detected
| Label | Principle | Signal |
|---|---|---|
| `AUTH` | Authority | Official status or authority cues |
| `SP` | Social Proof | References to what others do or validate |
| `LSD` | Liking/Similarity/Deception | Use of rapport, similarity, or personal deception |
| `CRC` | Commitment/Reciprocation | Pressure from prior commitment or reciprocity |
| `DIS` | Distraction | Urgency, fear, or distraction tactics |

### Pipeline Phases
1. **Data Harmonization** — Unifies phishing and persuasion datasets into a consistent format
2. **Master Split Creation** — Stable message-level splits (train/val/test/heldout) with leakage prevention
3. **Synthetic Augmentation** — Teacher LLM (Qwen 2.5 7B) generates explanations and reasoning for training data
4. **Classifier Training** — Multi-label fine-tuning with early stopping and quality gates
5. **SLM Fine-Tuning** — LoRA/QLoRA adaptation with hardware-aware configuration
6. **Model Export** — ONNX and OpenVINO formats for cross-platform deployment

### Hardware Adaptation
Automatically detects and configures for:
- **NVIDIA CUDA** (BF16/FP16)
- **Apple Silicon MPS**
- **Intel XPU**
- **CPU-only** fallback

| Profile | VRAM | Batch | Precision |
|---|---|---|---|
| HIGH_VRAM | 24GB+ | 32 | BF16 |
| MID_VRAM | 6-24GB | 8 (grad accum x4) | FP16 |
| LOW_VRAM | <6GB | 2 (grad accum x16) | FP16 |
| CPU_ONLY | N/A | 4 (grad accum x8) | FP32 |

### Interactive Webapp
- **Message Analysis** — Paste text, get principle scores and risk assessment
- **Contextual Chat** — Ask follow-up questions about the analyzed message
- **Evidence Highlighting** — See which text spans triggered each detection
- **Hardware Telemetry** — Live VRAM, RAM, and CPU monitoring
- **Research Dashboard** — Learning curves, metrics, and downloadable artifacts

### Intelligence Evaluation Framework
- Compares fine-tuned SLM against its base model
- Measures grounding, hallucination detection, and external drift
- Statistical significance testing (bootstrap CI, sign test)
- Acceptance gate ensures statistical superiority

---

## Tech Stack

| Category | Tools |
|---|---|
| **Models** | `microsoft/mdeberta-v3-base`, `FacebookAI/xlm-roberta-base`, `Qwen/Qwen2.5-7B-Instruct` |
| **ML Framework** | PyTorch, Transformers, PEFT (LoRA/QLoRA), Accelerate, BitsAndBytes, Datasets |
| **Data** | NumPy, Pandas, SciPy, Scikit-Learn, NLTK, SentencePiece |
| **Export** | ONNX, ONNX Runtime, OpenVINO |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Vanilla JS (ES6+), Modern CSS (Inter typography) |
| **Viz** | Matplotlib, Seaborn |
| **Monitoring** | psutil |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/lbustio/persuasion-slm.git
cd persuasion-slm
```

### 2. Set up the environment

**Conda (recommended):**
```bash
conda create -n tuning python=3.11
conda activate tuning
pip install -r configs/requirements.txt
```

**Or virtualenv:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r configs/requirements.txt
```

---

## Usage

### Run the Full Pipeline

```bash
# Full run with auto SLM selection
# Detects your hardware, picks the optimal SLM, and runs every phase:
# data harmonization, augmentation, classifier training, SLM fine-tuning, and model export
python main.py --fresh-all --slm auto

# SLM-only (skip classifier training and export)
# Runs only the phases needed to produce the SLM adapter:
# harmonization, splits, augmentation, and SLM fine-tuning.
# Useful when you already have a classifier or just want the LLM
python main.py --slm-only --slm auto

# Specific model selection
# Forces specific models instead of auto-detecting.
# Uses Qwen2.5-1.5B as SLM and XLM-RoBERTa as the classifier
python main.py --fresh-all --slm Qwen/Qwen2.5-1.5B-Instruct --classifier FacebookAI/xlm-roberta-base
```

#### CLI Arguments

| Flag | Description |
|---|---|
| `--fresh-all` | Rebuild every phase from scratch |
| `--fresh-harmonizer` | Regenerate harmonized data |
| `--fresh-augmenter` | Regenerate augmented data |
| `--fresh-classifier` | Retrain the classifier |
| `--fresh-slm` | Run fresh SLM fine-tuning |
| `--slm-only` | Run only phases needed for SLM |
| `--slm <model>` | SLM model ID or `auto` for hardware-based selection |
| `--slm-run-tag <tag>` | Optional suffix for the SLM adapter directory |
| `--classifier <model>` | Classifier model (default: `microsoft/mdeberta-v3-base`) |
| `--teacher <model>` | Teacher model for augmentation (default: `Qwen/Qwen2.5-7B-Instruct`) |
| `--data-dir <dir>` | Directory containing input CSV datasets |

### Launch the Webapp

```bash
python webapp.py
```

Open `http://127.0.0.1:8000` in your browser.

### Evaluate SLM Intelligence

```bash
# List detected SLM adapters
python evaluate_slm_intelligence.py --list-models

# Evaluate a specific adapter
python evaluate_slm_intelligence.py --model <adapter_name>

# Compare adapter vs base model
python evaluate_slm_intelligence.py --model <adapter_name> --compare-base

# JSON output for automation
python evaluate_slm_intelligence.py --model <adapter_name> --json
```

---

## Project Structure

```
persuasion-slm/
├── main.py                          # Pipeline orchestrator
├── webapp.py                        # Webapp launcher
├── evaluate_slm_intelligence.py     # SLM evaluation CLI
├── configs/
│   ├── training_defaults.yaml       # Hyperparameters, splits, hardware profiles
│   ├── architecture.yaml            # Directory paths and runtime config
│   ├── webapp.yaml                  # Thresholds, principles, prompts
│   └── requirements.txt             # Pip dependencies
├── src/
│   ├── pipeline/                    # ML pipeline modules
│   │   ├── harmonizer.py            # Phase 1: Data harmonization
│   │   ├── split_manager.py         # Master split creation
│   │   ├── augmenter.py             # Phase 2: Synthetic augmentation
│   │   ├── classifier_trainer.py    # Phase 3: Classifier training
│   │   ├── slm_finetuner.py         # Phase 4: SLM fine-tuning
│   │   ├── exporter.py              # Phase 5: ONNX/OpenVINO export
│   │   └── quality_gate.py          # Data validation
│   ├── intelligence_eval/           # SLM evaluation framework
│   ├── utils/                       # Hardware, logging, paths, state
│   ├── visualization/               # Plots and figures
│   └── reporting/                   # Research artifact generation
├── webapp/                          # FastAPI server + frontend
│   ├── server.py                    # API endpoints
│   ├── index.html                   # Main page
│   ├── app.js                       # Frontend logic
│   └── styles.css                   # Styling
├── data/                            # Input datasets (gitignored)
├── context/                         # Research notes and changelog
├── docs/                            # System documentation
├── caches/                          # Model downloads (gitignored)
├── outputs/                         # Generated artifacts (gitignored)
└── logs/                            # Execution logs (gitignored)
```

---

## Configuration

### Training (`configs/training_defaults.yaml`)

Key settings:
- **Learning rate**: `3.0e-5`
- **Max epochs**: `6` with early stopping patience `2`
- **Quality gate**: Minimum macro F1 of `0.65` at epoch 3
- **Max length**: `256` (classifier), `4096` (SLM)
- **Splits**: 70% train, 10% validation, 20% test, 10% heldout

### Webapp (`configs/webapp.yaml`)

- **Visible threshold**: `0.25` (shown in UI)
- **Positive threshold**: `0.40` (considered detected)
- **Strong threshold**: `0.60` (high confidence)
- **Languages**: Spanish, English, auto-detect

---

## License

This project is intended for research purposes.
