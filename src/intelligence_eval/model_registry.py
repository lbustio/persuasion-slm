from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import get_project_layout


@dataclass
class AdapterCandidate:
    name: str
    path: Path
    source: str
    base_model_name: str
    trained_at: float


@dataclass
class BaseModelCandidate:
    name: str
    model_id: str
    path: Path


def discover_slm_candidates() -> list[AdapterCandidate]:
    layout = get_project_layout()
    candidates: list[AdapterCandidate] = []
    search_roots = [
        ("active", layout.outputs_models),
        ("tuned_models", layout.outputs_tuned_models),
    ]

    for source, root in search_roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            adapter_cfg = path / "adapter_config.json"
            if not adapter_cfg.exists():
                continue
            try:
                payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            candidates.append(
                AdapterCandidate(
                    name=path.name,
                    path=path,
                    source=source,
                    base_model_name=str(payload.get("base_model_name_or_path", "")),
                    trained_at=path.stat().st_mtime,
                )
            )

    candidates.sort(key=lambda item: item.trained_at, reverse=True)
    return candidates


def resolve_candidate(name_or_path: str | None = None) -> AdapterCandidate:
    candidates = discover_slm_candidates()
    if not candidates:
        raise FileNotFoundError(
            "No se encontraron adaptadores SLM ni en outputs/results/models ni en outputs/tuned_models."
        )

    if not name_or_path:
        return candidates[0]

    raw = Path(name_or_path)
    if raw.exists():
        adapter_cfg = raw / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(f"La ruta {raw} no contiene adapter_config.json.")
        payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        return AdapterCandidate(
            name=raw.name,
            path=raw,
            source="manual",
            base_model_name=str(payload.get("base_model_name_or_path", "")),
            trained_at=raw.stat().st_mtime,
        )

    for candidate in candidates:
        if candidate.name == name_or_path:
            return candidate

    available = ", ".join(candidate.name for candidate in candidates[:12])
    raise FileNotFoundError(
        f"No se encontro el adaptador '{name_or_path}'. Disponibles: {available}"
    )


def resolve_hf_snapshot(model_id: str) -> Path | None:
    layout = get_project_layout()
    cache_root = Path(os.environ.get("HF_HOME", str(layout.cache_downloads)))
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


def load_slm_adapter(candidate: AdapterCandidate) -> dict[str, Any]:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    peft_cfg = PeftConfig.from_pretrained(str(candidate.path), local_files_only=True)
    base_snapshot_dir = resolve_hf_snapshot(peft_cfg.base_model_name_or_path)
    if base_snapshot_dir is None:
        raise FileNotFoundError(
            f"No se encontro en cache local el modelo base requerido: {peft_cfg.base_model_name_or_path}"
        )

    tokenizer_source = str(candidate.path if (candidate.path / "tokenizer_config.json").exists() else base_snapshot_dir)
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
    if hasattr(base_model, "tie_weights"):
        base_model.tie_weights()
    model = PeftModel.from_pretrained(base_model, str(candidate.path), local_files_only=True)
    model.eval()

    return {
        "candidate": candidate,
        "tokenizer": tokenizer,
        "model": model,
        "base_snapshot_dir": base_snapshot_dir,
    }


def resolve_base_candidate(candidate: AdapterCandidate) -> BaseModelCandidate:
    if not candidate.base_model_name:
        raise FileNotFoundError(f"El adaptador {candidate.name} no declara base_model_name_or_path.")

    base_snapshot_dir = resolve_hf_snapshot(candidate.base_model_name)
    if base_snapshot_dir is None:
        raise FileNotFoundError(
            f"No se encontro en cache local el modelo base requerido: {candidate.base_model_name}"
        )

    safe_name = candidate.base_model_name.replace("/", "__")
    return BaseModelCandidate(
        name=f"base::{safe_name}",
        model_id=candidate.base_model_name,
        path=base_snapshot_dir,
    )


def load_base_model(candidate: BaseModelCandidate) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(candidate.path), use_fast=True, local_files_only=True)
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

    model = AutoModelForCausalLM.from_pretrained(str(candidate.path), **model_kwargs)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()

    return {
        "candidate": candidate,
        "tokenizer": tokenizer,
        "model": model,
        "base_snapshot_dir": candidate.path,
    }
