import csv
import json
import logging
import os
import re
import psutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from peft import PeftConfig, PeftModel
from src.utils.paths import get_project_layout


BASE_DIR = Path(__file__).resolve().parent
LAYOUT = get_project_layout()
REPORTS_DIR = LAYOUT.outputs_reports
TABLES_DIR = LAYOUT.outputs_tables
FIGURES_DIR = LAYOUT.outputs_figures
SPLITS_DIR = LAYOUT.outputs_splits
MODELS_DIR = LAYOUT.outputs_models

os.environ["HF_HOME"] = str(LAYOUT.cache_downloads)
os.environ["TORCH_HOME"] = str(LAYOUT.cache_downloads)

LOGGER = logging.getLogger("webapp.server")


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


WEBAPP_CFG = _read_yaml(LAYOUT.root_dir / "configs" / "webapp.yaml")["webapp"]
PRINCIPLE_META: dict[str, dict[str, str]] = WEBAPP_CFG["principles"]
LABEL_KEYS = list(PRINCIPLE_META.keys())
LANGUAGE_LABELS: dict[str, str] = WEBAPP_CFG["languages"]
THRESHOLD = float(WEBAPP_CFG["thresholds"]["classifier_positive"])
VISIBLE_THRESHOLD = float(WEBAPP_CFG["thresholds"]["classifier_visible"])
STRONG_THRESHOLD = float(WEBAPP_CFG["thresholds"]["classifier_strong"])

_classifier_bundle: dict[str, Any] | None = None
_slm_bundle: dict[str, Any] | None = None
_bootstrap_cache: dict[str, Any] | None = None
_split_cache: list[dict[str, Any]] | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "auto"
    mode: str = "detailed"


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    language: str = "auto"
    mode: str = "detailed"
    analysis: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Persuasion Lab Webapp",
    description="Servidor local para la webapp del proyecto.",
    version="0.5.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _principle_catalog() -> list[dict[str, str]]:
    return [{"key": key, **meta} for key, meta in PRINCIPLE_META.items()]


def _status_from_score(score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return "Detectado"
    if score >= THRESHOLD:
        return "Senal debil"
    return "No detectado"


def _canonical_status(status: str | None) -> str:
    return _canonical_token(status or "")


def _is_active_status(status: str | None) -> bool:
    return _canonical_status(status) == "detectado"


def _is_visible_status(status: str | None) -> bool:
    token = _canonical_status(status)
    return token in {"detectado", "senaldebil"}


def _build_analysis_bundle(scores: dict[str, float], matched_record: dict[str, Any] | None) -> dict[str, Any]:
    status_by_label = {key: _status_from_score(value) for key, value in scores.items()}
    detected_keys = [key for key, status in status_by_label.items() if _is_active_status(status)]
    visible_keys = [key for key, status in status_by_label.items() if _is_visible_status(status)]
    gold_labels = matched_record.get("labels", {}) if matched_record else {}
    justifications = matched_record.get("justifications", {}) if matched_record else {}

    return {
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "status_by_label": status_by_label,
        "detected_keys": detected_keys,
        "visible_keys": visible_keys,
        "matched_record_lang": matched_record.get("lang") if matched_record else None,
        "matched_record_labels": gold_labels or None,
        "matched_record_justifications": {key: value for key, value in justifications.items() if value} or None,
    }


def _get_dir_size(path: Path) -> str:
    try:
        total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} GB"
    except Exception:
        return "N/A"


def _get_model_metadata(path: Path, is_adapter: bool = False) -> dict[str, Any]:
    if not path or not path.exists():
        return {"status": "missing", "name": "No disponible"}
    
    config_file = path / ("adapter_config.json" if is_adapter else "config.json")
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
        except Exception:
            pass
            
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    
    return {
        "name": path.name,
        "path": str(path.absolute()),
        "trained_at": mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "size": _get_dir_size(path),
        "base_model": config_data.get("base_model_name_or_path") or config_data.get("_name_or_path") or "N/A",
        "architecture": config_data.get("model_type") or config_data.get("peft_type") or "N/A",
        "status": "ready"
    }


def _discover_classifier_dir() -> Path:
    # Buscar modelos base que tengan config.json pero NO sean adaptadores (sin adapter_config.json)
    candidates = [
        path
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "config.json").exists() and not (path / "adapter_config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No se encontro ningun clasificador base (Encoder) en la carpeta oficial.")
    
    # Usar el mas reciente
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _discover_slm_adapter_dir() -> Path | None:
    # Buscar cualquier subdirectorio en MODELS_DIR que tenga un archivo adapter_config.json
    candidates = [
        path
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "adapter_config.json").exists()
    ]
    if not candidates:
        return None
    # Priorizar el que se haya modificado mas recientemente.
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _resolve_hf_snapshot(model_id: str) -> Path | None:
    cache_root = Path(os.environ.get("HF_HOME", str(LAYOUT.cache_downloads)))
    repo_dir = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return None

    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _load_classifier_bundle() -> dict[str, Any]:
    global _classifier_bundle
    if _classifier_bundle is not None:
        return _classifier_bundle

    model_dir = _discover_classifier_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            use_fast=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), use_safetensors=True, local_files_only=True)
    model.to(device)
    model.eval()

    _classifier_bundle = {
        "model_dir": model_dir,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }
    return _classifier_bundle


def _load_slm_bundle() -> dict[str, Any] | None:
    global _slm_bundle
    if _slm_bundle is not None:
        return _slm_bundle

    adapter_dir = _discover_slm_adapter_dir()
    if adapter_dir is None:
        return None

    try:
        peft_cfg = PeftConfig.from_pretrained(str(adapter_dir), local_files_only=True)
        base_snapshot_dir = _resolve_hf_snapshot(peft_cfg.base_model_name_or_path)
        if base_snapshot_dir is None:
            raise FileNotFoundError(
                f"No se encontro la base local requerida para el SLM: {peft_cfg.base_model_name_or_path}. "
                f"HF_HOME actual: {os.environ.get('HF_HOME')}"
            )

        tokenizer_source = str(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_snapshot_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"use_safetensors": True, "local_files_only": True}
        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["device_map"] = "auto"
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(str(base_snapshot_dir), **model_kwargs)
        
        # Qwen/Llama fix: ensure weights are tied before PEFT.
        if hasattr(base_model, "tie_weights"):
            base_model.tie_weights()
            
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), local_files_only=True)
        model.eval()

        _slm_bundle = {
            "adapter_dir": adapter_dir,
            "base_snapshot_dir": base_snapshot_dir,
            "tokenizer": tokenizer,
            "model": model,
        }
        return _slm_bundle
    except Exception as exc:
        _slm_bundle = {"error": str(exc), "adapter_dir": adapter_dir}
        return _slm_bundle


def _require_slm_bundle() -> dict[str, Any]:
    bundle = _load_slm_bundle()
    if not bundle or bundle.get("error"):
        detail = bundle.get("error") if bundle else "No se encontro un adaptador SLM entrenado."
        raise RuntimeError(
            "El sistema no puede trabajar sin el SLM local entrenado. "
            f"Detalle: {detail}"
        )
    return bundle


def _sigmoid_scores(text: str) -> dict[str, float]:
    bundle = _load_classifier_bundle()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    encoded = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        scores = torch.sigmoid(logits).cpu().tolist()[0]
    
    # Principles (first 5 labels)
    result = {label_key: float(score) for label_key, score in zip(LABEL_KEYS, scores[:5])}
    
    # 6th label is the binary phishing verdict (if trained)
    if len(scores) >= 6:
        result["is_phishing_direct"] = float(scores[5])
        
    return result


def _risk_profile(scores: dict[str, float]) -> tuple[str, str]:
    positives = sum(score >= THRESHOLD for score in scores.values())
    max_score = max(scores.values()) if scores else 0.0
    
    # Heuristics for specific threats
    has_authority = scores.get("authority", 0.0) >= THRESHOLD
    has_urgency = scores.get("distraction", 0.0) >= THRESHOLD
    has_deception = scores.get("liking_similarity_deception", 0.0) >= THRESHOLD
    
    # Combined signals that increase risk
    is_authoritative_urgency = has_authority and has_urgency
    is_deceptive_urgency = has_deception and has_urgency
    
    # Direct model verdict (6th label)
    direct_verdict_high = scores.get("is_phishing_direct", 0.0) >= 0.6

    if positives >= 3 or max_score >= 0.8 or is_authoritative_urgency or direct_verdict_high:
        return "Presion persuasiva alta (Riesgo de Phishing detectado)", "high"
    if positives >= 2 or max_score >= 0.55 or is_deceptive_urgency:
        return "Presion persuasiva moderada", "medium"
    return "Presion persuasiva baja o limitada", "low"


def _phishing_suspicion_analysis(scores: dict[str, float]) -> dict[str, Any]:
    auth = scores.get("authority", 0.0)
    dist = scores.get("distraction", 0.0)
    lsd = scores.get("liking_similarity_deception", 0.0)
    crc = scores.get("commitment_integrity_reciprocation", 0.0)
    
    suspicion_score = 0.0
    reasons = []
    
    # Weighting based on known phishing patterns
    if auth >= 0.4 and dist >= 0.4:
        suspicion_score += 0.4
        reasons.append("Combinacion critica de Autoridad y Urgencia.")
    elif auth >= 0.6:
        suspicion_score += 0.25
        reasons.append("Uso fuerte de Autoridad institucional.")
        
    if dist >= 0.6:
        suspicion_score += 0.2
        reasons.append("Presion de tiempo extrema (Urgencia).")
        
    if lsd >= 0.5:
        suspicion_score += 0.2
        reasons.append("Tecnicas de agrado o engano detectadas.")
        
    if crc >= 0.5:
        suspicion_score += 0.15
        reasons.append("Activacion de compromiso o reciprocidad.")
    # Normalize suspicion heuristic
    heuristic_score = min(suspicion_score + (sum(scores.values()) / 10.0), 1.0)
    
    # Integrate direct model verdict if available
    if "is_phishing_direct" in scores:
        direct_score = scores["is_phishing_direct"]
        # Blend: 40% heuristic, 60% direct model verdict
        final_score = (heuristic_score * 0.4) + (direct_score * 0.6)
        if direct_score >= 0.7:
            reasons.append("Veredicto directo del clasificador: Sospecha alta.")
    else:
        final_score = heuristic_score
    
    verdict = "Bajo"
    if final_score >= 0.7: verdict = "Muy Alto"
    elif final_score >= 0.5: verdict = "Alto"
    elif final_score >= 0.3: verdict = "Moderado"
    
    return {
        "score": round(final_score, 4),
        "verdict": verdict,
        "reasons": reasons
    }


def _load_split_records() -> list[dict[str, Any]]:
    global _split_cache
    if _split_cache is not None:
        return _split_cache

    split_path = _resolve_active_split_path()
    records: list[dict[str, Any]] = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    _split_cache = records
    return records


def _resolve_active_split_path() -> Path:
    manifest_path = SPLITS_DIR / "master_split_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            split_path = Path(manifest.get("split_files", {}).get("test", ""))
            if split_path.exists():
                return split_path
        except Exception:
            pass

    candidates = sorted(SPLITS_DIR.glob("*_test.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No se encontro ningun split de test para ejemplos de la webapp.")


def _find_matching_record(text: str) -> dict[str, Any] | None:
    compact = " ".join(text.strip().lower().split())
    try:
        records = _load_split_records()
    except FileNotFoundError:
        return None
    for record in records:
        candidate = " ".join(record["text"].strip().lower().split())
        if candidate == compact:
            return record
    return None


def _example_title(text: str, index: int) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else f"Caso {index}"
    title = " ".join(first_line.split()).strip()
    return title[:72] + ("..." if len(title) > 72 else "")


def _build_examples(limit: int = 8) -> list[dict[str, Any]]:
    try:
        records = _load_split_records()
    except FileNotFoundError:
        return []
    ranked = sorted(
        records,
        key=lambda row: (
            -int(row.get("is_phishing", 0)),
            row.get("lang", "es"),
            -sum(int(value) for value in row.get("labels", {}).values()),
        ),
    )
    examples: list[dict[str, Any]] = []
    for index, record in enumerate(ranked[:limit], start=1):
        examples.append(
            {
                "id": record["id"],
                "title": _example_title(record["text"], index),
                "language": record.get("lang", "es"),
                "source": record.get("source", "dataset"),
                "is_phishing": bool(record.get("is_phishing", 0)),
                "message": record["text"],
            }
        )
    return examples


def _compact_learning_curve(rows: list[dict[str, str]], train_field: str, eval_field: str) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in rows:
        if row.get(eval_field):
            result.append({"epoch": float(row["epoch"]), "value": float(row[eval_field]), "series": "eval"})
        elif row.get(train_field):
            result.append({"epoch": float(row["epoch"]), "value": float(row[train_field]), "series": "train"})
    return result


def _build_artifact_links() -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for folder_name, directory in (("figures", FIGURES_DIR), ("tables", TABLES_DIR), ("reports", REPORTS_DIR)):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                links.append({"label": path.name, "kind": folder_name, "url": f"/api/artifacts/{folder_name}/{path.name}"})
    return links


def _latest_file(directory: Path, pattern: str) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_json_optional(directory: Path, pattern: str) -> tuple[Any | None, str | None]:
    path = _latest_file(directory, pattern)
    if not path:
        return None, None
    return _read_json(path), path.name


def _read_csv_optional(directory: Path, pattern: str) -> tuple[list[dict[str, str]], str | None]:
    path = _latest_file(directory, pattern)
    if not path:
        return [], None
    return _read_csv(path), path.name


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _load_research_payload() -> dict[str, Any]:
    class_metrics, class_metrics_file = _read_json_optional(REPORTS_DIR, "*_paper_class_metrics.json")
    summary_metrics, summary_metrics_file = _read_json_optional(REPORTS_DIR, "*_paper_summary_metrics.json")
    split_summary, split_summary_file = _read_json_optional(REPORTS_DIR, "*_paper_split_summary.json")
    slm_summary, slm_summary_file = _read_json_optional(REPORTS_DIR, "*_slm_paper_summary.json")
    classifier_history, classifier_history_file = _read_csv_optional(TABLES_DIR, "*_paper_training_history.csv")
    slm_history, slm_history_file = _read_csv_optional(TABLES_DIR, "*_slm_paper_training_history.csv")

    class_metrics = class_metrics or []
    summary_metrics = summary_metrics or {}
    split_summary = split_summary or []
    slm_summary = slm_summary or {}

    return {
        "artifactStatus": {
            "class_metrics": class_metrics_file,
            "summary_metrics": summary_metrics_file,
            "split_summary": split_summary_file,
            "slm_summary": slm_summary_file,
            "classifier_history": classifier_history_file,
            "slm_history": slm_history_file,
        },
        "splitSummary": [{"split": item["split"], "samples": int(item["samples"])} for item in split_summary],
        "classMetrics": [
            {
                "label": PRINCIPLE_META[item["label_key"]]["label"],
                "precision": round(_safe_float(item.get("precision")), 4),
                "recall": round(_safe_float(item.get("recall")), 4),
                "f1": round(_safe_float(item.get("f1_score")), 4),
                "support": int(_safe_float(item.get("support"))),
                "roc_auc": round(_safe_float(item.get("roc_auc")), 4),
                "average_precision": round(_safe_float(item.get("average_precision")), 4),
            }
            for item in class_metrics
            if item.get("label_key") in PRINCIPLE_META
        ],
        "summaryMetrics": {key: round(float(value), 4) for key, value in summary_metrics.items()},
        "classifierLearningCurve": _compact_learning_curve(classifier_history, "loss", "eval_loss"),
        "slmLearningCurve": _compact_learning_curve(slm_history, "loss", "eval_loss"),
        "slmSummary": {
            "model_name": slm_summary.get("model_name", "No disponible"),
            "train_samples": int(_safe_float(slm_summary.get("train_samples"))),
            "eval_samples": int(_safe_float(slm_summary.get("eval_samples"))),
            "eval_ratio": _safe_float(slm_summary.get("eval_ratio")),
        },
        "artifactLinks": _build_artifact_links(),
    }


def _principle_guide_text() -> str:
    return "\n".join(f"- {key}: {meta['label']} | {meta['primer']}" for key, meta in PRINCIPLE_META.items())


def _classifier_profile(analysis_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "scores": analysis_bundle["scores"],
        "status_by_label": analysis_bundle["status_by_label"],
        "predicted_positive_labels": analysis_bundle["detected_keys"],
        "visible_labels": analysis_bundle["visible_keys"],
        "matched_record_lang": analysis_bundle.get("matched_record_lang"),
        "matched_record_labels": analysis_bundle.get("matched_record_labels"),
        "matched_record_justifications": analysis_bundle.get("matched_record_justifications"),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("El SLM no devolvio un objeto JSON valido.")

    for index in range(start, len(text)):
        if text[index] != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("El SLM no devolvio un objeto JSON decodificable.")


def _extract_json_string_field(text: str, field_name: str) -> str | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw_value = match.group(1)
    try:
        return json.loads(f'"{raw_value}"')
    except json.JSONDecodeError:
        return raw_value


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe_repeated_sentences(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalize_whitespace(part)
        if not normalized:
            continue
        fingerprint = normalized.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_parts.append(normalized)

    return " ".join(unique_parts) if unique_parts else cleaned


def _dedupe_repeated_blocks(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""

    words = cleaned.split()
    total = len(words)
    if total < 12:
        return cleaned

    for block_size in range(4, (total // 2) + 1):
        if total % block_size != 0:
            continue
        repeats = total // block_size
        if repeats < 2:
            continue
        block = words[:block_size]
        if all(words[index * block_size : (index + 1) * block_size] == block for index in range(repeats)):
            return " ".join(block)
    return cleaned


def _sanitize_generated_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = _dedupe_repeated_blocks(text)
    cleaned = _dedupe_repeated_sentences(cleaned)
    
    # Cut off common small-model drift into hashtag lists.
    if "#" in cleaned:
        # Si hay mAs de 3 hashtags, cortamos antes del primero
        if cleaned.count("#") > 2:
            cleaned = cleaned.split("#")[0].strip()
    
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def _extract_json_array_block(text: str, field_name: str) -> str | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*\['
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_json_object_block(text: str, field_name: str) -> str | None:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*\{{'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _salvage_suggested_questions(raw_text: str) -> list[Any]:
    array_block = _extract_json_array_block(raw_text, "suggested_questions")
    if not array_block:
        return []
    try:
        parsed = json.loads(array_block)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        recovered = []
        for item in re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', array_block, flags=re.IGNORECASE | re.DOTALL):
            try:
                recovered.append({"text": json.loads(f'"{item}"')})
            except json.JSONDecodeError:
                recovered.append({"text": item})
        return recovered


def _salvage_principles(raw_text: str) -> dict[str, Any]:
    object_block = _extract_json_object_block(raw_text, "principles")
    if not object_block:
        return {}
    try:
        parsed = json.loads(object_block)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        recovered: dict[str, Any] = {}
        principle_pattern = re.compile(
            r'"(?P<key>[^"]+)"\s*:\s*\{(?P<body>.*?)(?:\}(?=,\s*"[^"]+"\s*:)|\}\s*$)',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in principle_pattern.finditer(object_block):
            body = match.group("body")
            recovered[match.group("key")] = {
                "status": _extract_json_string_field(body, "status") or "",
                "score_rationale": _extract_json_string_field(body, "score_rationale") or "",
                "explanation": _extract_json_string_field(body, "explanation") or "",
                "evidence": _extract_json_string_field(body, "evidence") or "",
                "intensity": 0,
            }
        return recovered


def _salvage_slm_payload(raw_text: str) -> dict[str, Any]:
    payload = {
        "detected_language": _extract_json_string_field(raw_text, "detected_language"),
        "simple_summary": _sanitize_generated_text(_extract_json_string_field(raw_text, "simple_summary")),
        "technical_summary": _sanitize_generated_text(_extract_json_string_field(raw_text, "technical_summary")),
        "suggested_questions": _salvage_suggested_questions(raw_text),
        "principles": _salvage_principles(raw_text),
        "evidence_segments": [],
    }
    useful = any(
        [
            payload["detected_language"],
            payload["simple_summary"],
            payload["technical_summary"],
            payload["suggested_questions"],
            payload["principles"],
        ]
    )
    if not useful:
        raise ValueError("No se pudo recuperar estructura util desde la salida cruda del SLM.")
    return payload


def _sanitize_slm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["simple_summary"] = _sanitize_generated_text(payload.get("simple_summary"))
    payload["technical_summary"] = _sanitize_generated_text(payload.get("technical_summary"))
    return payload


def _generate_slm_json(messages: list[dict[str, str]], max_new_tokens: int = 520) -> dict[str, Any]:
    bundle = _require_slm_bundle()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=min(max_new_tokens, 360),
            do_sample=False,
            repetition_penalty=1.12,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("El SLM local cargo, pero no produjo salida.")

    try:
        return _sanitize_slm_payload(_extract_json_object(answer))
    except Exception:
        try:
            LOGGER.warning("El SLM devolvio JSON malformado; se intentara recuperar una estructura parcial.")
            return _sanitize_slm_payload(_salvage_slm_payload(answer))
        except Exception as salvage_exc:
            raise RuntimeError(f"El SLM no devolvio JSON analizable. Respuesta cruda: {answer[:600]}") from salvage_exc


def _normalize_detected_language(raw_language: str | None, matched_record: dict[str, Any] | None) -> str:
    if raw_language in {"es", "en"}:
        return raw_language
    if matched_record and matched_record.get("lang") in {"es", "en"}:
        return matched_record["lang"]
    return "es"

def _canonical_token(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^\w]+", "", without_accents, flags=re.IGNORECASE)


def _coerce_principle_key(raw_key: str) -> str | None:
    token = _canonical_token(raw_key)
    direct_map = {
        "authority": "authority",
        "autoridad": "authority",
        "socialproof": "social_proof",
        "pruebasocial": "social_proof",
        "social": "social_proof",
        "likingsimilaritydeception": "liking_similarity_deception",
        "agradoengano": "liking_similarity_deception",
        "engano": "liking_similarity_deception",
        "deception": "liking_similarity_deception",
        "commitmentintegrityreciprocation": "commitment_integrity_reciprocation",
        "compromisoreciprocidad": "commitment_integrity_reciprocation",
        "compromiso": "commitment_integrity_reciprocation",
        "reciprocidad": "commitment_integrity_reciprocation",
        "distraction": "distraction",
        "distraccion": "distraction",
        "urgencia": "distraction",
        "distraccionurgencia": "distraction",
    }
    if token in direct_map:
        return direct_map[token]

    for key, meta in PRINCIPLE_META.items():
        if token == _canonical_token(key) or token == _canonical_token(meta["label"]) or token == _canonical_token(meta["short"]):
            return key
    return None


def _coerce_principle_map(raw_principles: Any) -> dict[str, Any]:
    if not isinstance(raw_principles, dict):
        return {}

    normalized: dict[str, Any] = {}
    for raw_key, payload in raw_principles.items():
        canonical = _coerce_principle_key(str(raw_key))
        if canonical is None:
            continue
        normalized[canonical] = payload

    return normalized


def _normalize_evidence_segments(raw_segments: Any, principles: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            raw_text = segment.get("text", "")
            if isinstance(raw_text, list):
                text = " ".join(str(item).strip() for item in raw_text if str(item).strip()).strip()
            else:
                text = str(raw_text).strip()
            principle = _coerce_principle_key(segment.get("principle")) if segment.get("principle") else None
            if not text:
                continue
            segments.append({"text": text + " ", "principle": principle})

    if segments:
        return [
            segment
            for segment in segments
            if segment["principle"] is None or _is_visible_status(principles.get(segment["principle"], {}).get("status"))
        ]

    return [
        {"text": value["evidence"] + " ", "principle": key}
        for key, value in principles.items()
        if _is_visible_status(value["status"]) and value["evidence"] != "Sin evidencia textual aislada."
    ]


def _normalize_suggested_questions(raw_questions: Any) -> list[str]:
    if not isinstance(raw_questions, list):
        return []
    normalized: list[str] = []
    for item in raw_questions:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        cleaned = _sanitize_generated_text(text)
        if cleaned:
            normalized.append(cleaned)
    return normalized[:4]


def _build_logs(language: str, used_dataset_match: bool) -> list[dict[str, str]]:
    now = _timestamp()
    logs = [
        {"time": now, "stage": "Entrada", "text": "Mensaje recibido. Iniciando lectura contextual y segmentacion del contenido."},
        {"time": now, "stage": "Clasificador", "text": "Extrayendo senales de persuasion mediante el clasificador mDeBERTa optimizado."},
        {"time": now, "stage": "SLM", "text": "Razonamiento local activo: el SLM busca evidencia textual y redacta la explicacion."},
        {"time": now, "stage": "Idioma", "text": f"Deteccion automatica: se ha identificado el idioma {LANGUAGE_LABELS.get(language, language)}."},
    ]
    if used_dataset_match:
        logs.append({"time": now, "stage": "Dataset", "text": "Coincidencia exacta encontrada en el repositorio de entrenamiento. Enriqueciendo analisis con contexto historico."})
    logs.append({"time": now, "stage": "Finalizado", "text": "Analisis completado. El sistema esta listo para responder preguntas sobre el mensaje."})
    return logs


def _classifier_meta() -> dict[str, Any]:
    path = _discover_classifier_dir()
    return _get_model_metadata(path, is_adapter=False) if path else {"status": "missing"}

def _slm_meta() -> dict[str, Any]:
    path = _discover_slm_adapter_dir()
    return _get_model_metadata(path, is_adapter=True) if path else {"status": "missing"}

def _get_hardware_info():
    import subprocess
    import platform
    def _get_cpu_name():
        try:
            if os.name == "nt":
                cmd = "powershell -command \"Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name\""
                return subprocess.check_output(cmd, shell=True).decode().strip()
            return platform.processor()
        except:
            return platform.machine()

    return {
        "active": "CUDA GPU" if torch.cuda.is_available() else "CPU",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU",
        "vram_free_gb": round(torch.cuda.mem_get_info(0)[0] / (1024**3), 1) if torch.cuda.is_available() else 0,
        "vram_total_gb": round(torch.cuda.mem_get_info(0)[1] / (1024**3), 1) if torch.cuda.is_available() else 0,
        "ram_free_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cpu_count": os.cpu_count() or 0,
        "cpu_model": _get_cpu_name()
    }

def _build_bootstrap() -> dict[str, Any]:
    c_meta = _classifier_meta()
    s_meta = _slm_meta()
    
    slm_status = s_meta.get("status", "missing")
    slm_error = None
    if slm_status == "missing":
        slm_error = "No se encontro un adaptador SLM en la carpeta oficial."

    return {
        "app": {
            "title": "Persuasion Lab",
            "status": "ready" if slm_status == "ready" else "blocked",
            "classifier": c_meta,
            "slm": s_meta,
            "slm_status": slm_status,
            "slm_error": slm_error,
            "hardware": _get_hardware_info(),
        },
        "principleCatalog": _principle_catalog(),
        "examples": _build_examples(),
        "research": _load_research_payload(),
    }


@app.get("/api/hardware", tags=["webapp"])
def hardware_stats():
    return _get_hardware_info()


def _build_analysis_response(text: str, language: str, mode: str) -> dict[str, Any]:
    _require_slm_bundle()
    scores = _sigmoid_scores(text)
    matched_record = _find_matching_record(text)
    analysis_bundle = _build_analysis_bundle(scores, matched_record)
    slm_payload = _slm_analysis(text, language, analysis_bundle)
    normalized_language = _normalize_detected_language(slm_payload.get("detected_language"), matched_record)
    principles = _normalize_principles(slm_payload.get("principles", {}), analysis_bundle)
    evidence_segments = _normalize_evidence_segments(slm_payload.get("evidence_segments", []), principles)
    overall_risk, risk_level = _risk_profile(scores)
    phishing_suspicion = _phishing_suspicion_analysis(scores)
    analysis_snapshot = {"principles": principles, "scores": analysis_bundle["scores"]}
    detected = [PRINCIPLE_META[key]["label"] for key, _payload, _score in _active_principle_rows(analysis_snapshot)]
    visible = [PRINCIPLE_META[key]["label"] for key, _payload, _score in _visible_principle_rows(analysis_snapshot)]

    return {
        "overallRisk": overall_risk,
        "riskLevel": risk_level,
        "phishingSuspicion": phishing_suspicion,
        "detectedLanguage": LANGUAGE_LABELS.get(normalized_language, normalized_language),
        "analysisMode": {"quick": "Rapido", "detailed": "Detallado", "educational": "Educativo"}.get(mode, "Detallado"),
        "runtime": "clasificador + slm local",
        "classifierModel": _discover_classifier_dir().name,
        "slmModel": (_discover_slm_adapter_dir() or Path("No disponible")).name,
        "slmRequired": True,
        "principleCatalog": _principle_catalog(),
        "simpleSummary": slm_payload.get("simple_summary") or "El SLM no devolvio resumen simple.",
        "technicalSummary": slm_payload.get("technical_summary") or "El SLM no devolvio resumen tecnico.",
        "scores": analysis_bundle["scores"],
        "principles": principles,
        "evidenceSegments": evidence_segments,
        "suggestedQuestions": _normalize_suggested_questions(slm_payload.get("suggested_questions")),
        "chat": [
            {
                "role": "assistant",
                "text": "El analisis se genero con el clasificador del proyecto y tu SLM local entrenado. Pregunta sobre evidencia, diferencias o ambiguedad.",
            }
        ],
        "logs": _build_logs(normalized_language, matched_record is not None),
        "detectedPrinciples": detected,
        "visiblePrinciples": visible,
        "analysisBundle": analysis_bundle,
        "research": _load_research_payload(),
    }


def _normalize_for_similarity(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^\w]+", " ", without_accents, flags=re.IGNORECASE)
    return [token for token in cleaned.split() if token]


def _looks_like_prompt_echo(question: str, answer: str) -> bool:
    q_tokens = _normalize_for_similarity(question)
    a_tokens = _normalize_for_similarity(answer)
    if not q_tokens or not a_tokens:
        return False
    shared = set(q_tokens) & set(a_tokens)
    overlap = len(shared) / max(len(set(q_tokens)), 1)
    answer_starts_like_question = " ".join(a_tokens[: min(len(q_tokens), 10)]) in " ".join(q_tokens)
    return overlap >= 0.72 or answer_starts_like_question


def _top_detected_principles(analysis: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    principles = analysis.get("principles", {})
    scores = analysis.get("scores", {})
    rows = []
    for key, payload in principles.items():
        rows.append((key, payload, float(scores.get(key, 0.0))))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def _active_principle_rows(analysis: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    return [row for row in _top_detected_principles(analysis) if _is_active_status(row[1].get("status"))]
def _calculate_dynamic_max_tokens(question: str) -> int:
    lower_q = question.lower()
    # Keep answers flexible without giving the model unlimited room to drift.
    if any(kw in lower_q for kw in ["explica", "detalle", "resumen", "analiza", "por que", "porque", "evidencia"]):
        return 1024
    if len(question) < 30:
        return 192
    return 512


def _visible_principle_rows(analysis: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    return [row for row in _top_detected_principles(analysis) if _is_visible_status(row[1].get("status"))]


def _normalize_principles(raw_principles: dict[str, Any], analysis_bundle: dict[str, Any]) -> dict[str, Any]:
    raw_principles = _coerce_principle_map(raw_principles)
    normalized: dict[str, Any] = {}
    for key in LABEL_KEYS:
        status = analysis_bundle["status_by_label"].get(key, "No detectado")
        raw = (raw_principles.get(key) or {}) if _is_visible_status(status) else {}
        score = float(analysis_bundle["scores"].get(key, 0.0))
        raw_evidence = raw.get("evidence")
        if isinstance(raw_evidence, list):
            evidence_text = " | ".join(str(item).strip() for item in raw_evidence if str(item).strip()) or "Sin evidencia textual aislada."
        elif raw_evidence is None:
            evidence_text = "Sin evidencia textual aislada."
        else:
            evidence_text = str(raw_evidence).strip() or "Sin evidencia textual aislada."

        if status == "No detectado":
            default_explanation = f"No se observan senales claras de {PRINCIPLE_META[key]['label']}."
        elif evidence_text == "Sin evidencia textual aislada.":
            default_explanation = (
                f"El clasificador sugiere {PRINCIPLE_META[key]['label']}, "
                "pero el SLM no devolvio una justificacion textual suficiente."
            )
        else:
            default_explanation = (
                f"El clasificador sugiere {PRINCIPLE_META[key]['label']} y existe evidencia textual para auditar esa hipotesis."
            )

        normalized[key] = {
            "status": status,
            "explanation": raw.get("explanation") or raw.get("score_rationale") or default_explanation,
            "evidence": evidence_text,
            "intensity": int(raw.get("intensity", round(score * 10))),
        }
    return normalized


def _slm_analysis(text: str, language: str, analysis_bundle: dict[str, Any]) -> dict[str, Any]:
    profile = _classifier_profile(analysis_bundle)
    visible_keys = profile["visible_labels"]
    explanation_bundle = {
        **profile,
        "scores": {key: profile["scores"][key] for key in visible_keys},
        "status_by_label": {key: profile["status_by_label"][key] for key in visible_keys},
        "predicted_positive_labels": [key for key in visible_keys if _is_active_status(profile["status_by_label"][key])],
        "visible_labels": visible_keys,
    }

    prompts_cfg = WEBAPP_CFG.get("prompts", {})
    analysis_system = prompts_cfg.get("analysis_system", "").strip()
    analysis_user_template = prompts_cfg.get("analysis_user_template", "").strip()
    principle_guide = _principle_guide_text()
    classifier_profile = json.dumps(explanation_bundle, indent=2, ensure_ascii=False)
    if analysis_user_template:
        user_prompt = analysis_user_template.format(
            principle_guide=principle_guide,
            classifier_profile=classifier_profile,
            message_text=text,
        )
    else:
        user_prompt = (
            "Analiza el siguiente mensaje tomando el bundle del clasificador como hipotesis inicial, "
            "no como verdad final. Devuelve solo JSON valido.\n\n"
            f"Principios disponibles:\n{principle_guide}\n\n"
            f"Bundle estructurado del backend:\n{classifier_profile}\n\n"
            f"Mensaje:\n{text}"
        )

    messages = [
        {
            "role": "system",
            "content": analysis_system
            or (
                "Eres un analista forense de ciberseguridad. Tu trabajo es tomar el bundle del clasificador "
                "como hipotesis inicial, contrastarlo con el texto y devolver solo JSON valido con evidencia textual."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    return _generate_slm_json(messages)


def _deterministic_chat_answer(question: str, analysis: dict[str, Any]) -> str | None:
    lower = question.lower()
    simple_summary = analysis.get("simpleSummary", "")
    technical_summary = analysis.get("technicalSummary", "")
    top_rows = _top_detected_principles(analysis)
    positive_rows = _active_principle_rows(analysis)
    visible_rows = _visible_principle_rows(analysis)

    asks_summary = any(term in lower for term in ["habla", "resume", "resumen", "que dice", "trata"])
    asks_principles = any(term in lower for term in ["principio", "persua", "senal", "detect", "presente", "encontr", "encotr"])
    asks_evidence = any(term in lower for term in ["por que", "porque", "evidencia", "prueba", "sustento", "donde dice"])
    asks_risk = any(term in lower for term in ["riesgo", "peligro", "grave", "confiable", "confiar", "phishing", "estafa", "engano"])

    if asks_summary and asks_evidence:
        if not positive_rows:
            return (
                f"{simple_summary or technical_summary or 'No tengo un resumen claro del mensaje.'} "
                "No se han detectado principios persuasivos dominantes en este contenido."
            )
        key, payload, score = positive_rows[0]
        return (
            f"{simple_summary or technical_summary or 'No tengo un resumen claro del mensaje.'} "
            f'El factor principal es {PRINCIPLE_META[key]["label"]}. '
            f'Evidencia: "{payload.get("evidence", "Sin evidencia aislada")}". '
            f'Lectura: {payload.get("explanation", "Principio detectado.")}'
        )

    if asks_summary:
        return _sanitize_generated_text(simple_summary or technical_summary) or "No tengo un resumen fiable del mensaje actual."

    if asks_evidence:
        candidate_rows = positive_rows or visible_rows or top_rows
        if not candidate_rows:
            return "No veo evidencia suficiente para sostener un principio persuasivo fuerte en el mensaje actual."
        key, payload, score = candidate_rows[0]
        evidence = payload.get("evidence", "Sin evidencia textual aislada.")
        explanation = payload.get("explanation", "No tengo una lectura lista para este hallazgo.")
        if "Sin evidencia" in evidence:
            return (
                "Puedo comentar la hipotesis general del sistema, pero no tengo una cita textual suficientemente clara "
                "para sostenerla con confianza. Prefiero no inventar una justificacion."
            )
        return (
            f'He observado que el factor principal es {PRINCIPLE_META[key]["label"]}. '
            f'En el texto vemos: "{evidence}". '
            f"{explanation}"
        )

    if asks_principles:
        if not positive_rows:
            if visible_rows:
                parts = []
                for key, payload, score in visible_rows[:3]:
                    parts.append(
                        f'{PRINCIPLE_META[key]["label"]}: {payload.get("explanation", "Senal visible.")}'
                    )
                return "No hay principios dominantes, pero observo estas senales: " + " ".join(parts)
            return "Segun el analisis actual no hay principios de persuasion claramente activos."
        parts = []
        for key, payload, score in positive_rows[:3]:
            parts.append(
                f'{PRINCIPLE_META[key]["label"]}: {payload.get("explanation", "Principio activo.")}'
            )
        return "He detectado los siguientes factores de persuasion: " + " ".join(parts)

    if asks_risk:
        return _sanitize_generated_text(technical_summary or simple_summary) or "No tengo una valoracion de riesgo lista para este mensaje."

    if any(term in lower for term in ["dime", "cuenta", "explica", "habla"]):
        return _sanitize_generated_text(simple_summary or technical_summary) or "No tengo una explicacion lista para este mensaje."

    return None


def _slm_answer(question: str, text: str, language: str, analysis: dict[str, Any]) -> tuple[str, str]:
    try:
        bundle = _require_slm_bundle()
    except Exception:
        deterministic = _deterministic_chat_answer(question, analysis)
        return deterministic or "No tengo un motor de lenguaje activo.", "analysis"

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    analysis_bundle = analysis.get("analysisBundle") or {}
    hypothesis = {
        "scores": analysis.get("scores") or analysis_bundle.get("scores") or {},
        "principles": analysis.get("principles") or {},
        "detected_principles": analysis.get("detectedPrinciples") or [],
        "visible_principles": analysis.get("visiblePrinciples") or [],
        "classifier_bundle": {
            "status_by_label": analysis_bundle.get("status_by_label", {}),
            "detected_keys": analysis_bundle.get("detected_keys", []),
            "visible_keys": analysis_bundle.get("visible_keys", []),
            "matched_record_labels": analysis_bundle.get("matched_record_labels"),
            "matched_record_justifications": analysis_bundle.get("matched_record_justifications"),
        },
    }
    hypothesis_json = json.dumps(hypothesis, ensure_ascii=False, indent=2)
    chat_system = WEBAPP_CFG.get("prompts", {}).get("chat_system", "").strip() or (
        "Eres un Analista Senior de Ciberseguridad especializado en persuasion y phishing. "
        "El clasificador aporta una hipotesis inicial, no una verdad final. "
        "Responde con base en el texto; si no hay datos suficientes, dilo sin inventar."
    )

    messages = [
        {
            "role": "system",
            "content": chat_system,
        },
        {
            "role": "user",
            "content": (
                "Responde la pregunta del investigador usando el mensaje como fuente principal.\n"
                "Usa la hipotesis del clasificador solo como punto de partida: puedes confirmarla, "
                "matizarla o refutarla.\n"
                "Si afirmas evidencia, cita fragmentos exactos del mensaje.\n"
                "Si la pregunta requiere informacion que no esta en el mensaje ni en la hipotesis, "
                "di claramente que no puedes afirmarlo.\n\n"
                f"IDIOMA SOLICITADO: {language}\n\n"
                f"HIPOTESIS INICIAL DEL SISTEMA:\n{hypothesis_json}\n\n"
                f"MENSAJE A ANALIZAR:\n\"\"\"{text}\"\"\"\n\n"
                f"PREGUNTA DEL INVESTIGADOR: {question}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=_calculate_dynamic_max_tokens(question),
        do_sample=False,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        generated = model.generate(**encoded, generation_config=gen_config, use_cache=True)

    new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not answer or _looks_like_prompt_echo(question, answer):
        fallback = _deterministic_chat_answer(question, analysis)
        return fallback or "El modelo no pudo generar una respuesta coherente.", "analysis"

    return _sanitize_generated_text(answer), "slm"


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health", tags=["system"])
@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "app": "webapp"}


@app.get("/api/bootstrap", tags=["webapp"])
def bootstrap():
    return _build_bootstrap()


@app.get("/api/examples", tags=["webapp"])
def examples():
    return {"examples": _build_examples()}


@app.get("/api/research/summary", tags=["webapp"])
def research_summary():
    return _load_research_payload()


@app.post("/api/analyze", tags=["analysis"])
def analyze(request: AnalyzeRequest):
    LOGGER.info(
        "Inicio /api/analyze | chars=%s | language=%s | mode=%s",
        len(request.text),
        request.language,
        request.mode,
    )
    try:
        response = _build_analysis_response(request.text, request.language, request.mode)
        LOGGER.info(
            "Fin /api/analyze | detected_language=%s | risk=%s | detected=%s | visible=%s",
            response.get("detectedLanguage"),
            response.get("riskLevel"),
            response.get("detectedPrinciples"),
            response.get("visiblePrinciples"),
        )
        return response
    except RuntimeError as exc:
        LOGGER.exception("Fallo en /api/analyze: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/chat", tags=["analysis"])
def chat(request: ChatRequest):
    LOGGER.info(
        "Inicio /api/chat | question_chars=%s | has_analysis=%s",
        len(request.question),
        bool(request.analysis),
    )
    try:
        analysis = request.analysis or _build_analysis_response(request.text, request.language, request.mode)
        answer, source = _slm_answer(request.question, request.text, request.language, analysis)
    except RuntimeError as exc:
        LOGGER.exception("Fallo en /api/chat: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    LOGGER.info("Fin /api/chat | source=%s | answer_chars=%s", source, len(answer))

    return {
        "answer": answer,
        "source": source,
        "log": {
            "time": _timestamp(),
            "stage": "SLM" if source == "slm" else "Analisis",
            "text": "Respuesta generada por el SLM local entrenado para la conversacion contextual."
            if source == "slm"
            else "Respuesta generada desde la capa estructurada del analisis para evitar eco o alucinacion conversacional.",
        },
    }


@app.get("/api/artifacts/{kind}/{filename}", tags=["webapp"])
def artifact(kind: str, filename: str):
    allowed = {
        "figures": FIGURES_DIR,
        "tables": TABLES_DIR,
        "reports": REPORTS_DIR,
        "splits": SPLITS_DIR,
    }
    if kind not in allowed:
        raise HTTPException(status_code=404, detail="Tipo de artefacto no soportado.")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo invalido.")
    path = allowed[kind] / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    return FileResponse(path)
