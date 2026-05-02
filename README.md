# <img src="webapp/logo.png" width="48" align="center"> Persuasion Lab

**Persuasion Lab** is an advanced research platform for the detection and analysis of persuasion principles in text. It combines a state-of-the-art **multi-label classifier** (mDeBERTa) with a **Small Language Model (SLM)** (Qwen-2.5) to provide not only detection scores but also human-like reasoning and textual evidence.

---

## 🚀 Key Features

- **Dynamic Hardware Discovery**: Automatically detects and utilizes available hardware (CUDA/CPU) without hardcoded paths.
- **DNA-based Model Discovery**: Scans and identifies trained classifiers and SLM adapters based on structural signatures (`config.json`/`adapter_config.json`).
- **Forensic Audit Workspace**: A professional web interface featuring:
  - **Live Hardware Telemetry**: VRAM, RAM, and CPU core monitoring.
  - **Contextual Chat**: Talk to the SLM about specific analyzed messages.
  - **Evidence Highlighting**: See exactly which parts of the text triggered each principle.
  - **Execution Logs**: Real-time technical trace of the analysis pipeline.
- **Privacy-First**: Designed for local-only execution to ensure data sovereignty.

---

## 🛠️ Tech Stack

- **Models**: `microsoft/mdeberta-v3-base` (Encoder) + `Qwen/Qwen2.5-1.5B-Instruct` (Adapter).
- **Backend**: Python, FastAPI, PyTorch, HuggingFace Transformers, PEFT.
- **Frontend**: Vanilla JS (ES6+), Modern CSS (Inter Typography).
- **Monitoring**: `psutil` for real-time system metrics.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/persuasion-lab.git
   cd persuasion-lab
   ```

2. **Set up the environment**:
   ```bash
   conda create -n tuning python=3.10
   conda activate tuning
   pip install -r requirements.txt
   ```

3. **Prepare Models**:
   Ensure your trained models are placed in `outputs/results/models/`. The system will automatically detect them.

---

## 🖥️ Usage

To start the research workbench:

```bash
python webapp.py
```

Then open your browser at `http://127.0.0.1:8000`.

---

## 📂 Project Structure

- `src/`: Core logic for training, reporting, and utilities.
- `webapp/`: FastAPI server and web interface.
- `configs/`: YAML configurations for the pipeline and webapp.
- `context/`: Project history, memory logs, and research notes.
- `outputs/`: (Ignored) Directory for model weights and results.

---

## 📄 License

This project is intended for research purposes. See the [LICENSE](LICENSE) file for details (if applicable).

---

Developed with ❤️ by the Persuasion Lab Team.
