from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


INSTRUCT_MARKERS = ("instruct", "chat", "it")
BAD_MARKERS = ("embedding", "reranker", "vision", "vl", "coder", "gguf", "awq", "gptq")

# Quality bonus multipliers for known high-capability teacher model families.
# Higher = better reasoning and instruction following for synthetic audit generation.
QUALITY_FAMILIES: dict[str, float] = {
    "llama-3.3":     2.0,   # Meta's best open instruct (late 2024)
    "llama-3.1":     1.8,
    "llama-3":       1.6,
    "qwen2.5":       1.5,   # Alibaba, strong multilingual reasoning
    "qwen3":         1.7,
    "gemma-2":       1.3,   # Google, efficient and capable
    "mistral":       1.2,
    "mixtral":       1.4,   # MoE variant, very capable
    "command-r-plus":1.3,
    "command-r":     1.1,
    "phi-4":         1.2,   # Microsoft, surprisingly capable small models
    "phi-3":         0.9,
}


@dataclass
class ModelCandidateScore:
    model_id: str
    score: float
    params_b: float
    downloads: int
    likes: int
    context_length: int | None = None
    reason: str = ""
    gated: bool = False


class ModelAdvisor:
    def __init__(
        self,
        logger: logging.Logger,
        cache_dir: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
    ):
        self.logger = logger
        self.hf_api_url = "https://huggingface.co/api/models"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
        self._readme_cache: dict[str, str] = {}

    def discover_optimal_model(self, hw_specs: dict[str, Any], *, limit: int = 80) -> str:
        report = self.rank_models(hw_specs, limit=limit)
        self._write_report(report)
        if not report["winner"]:
            raise RuntimeError(
                "No SLM candidate could be selected automatically. Internet search failed or no cached model fits the hardware. "
                "Provide --slm explicitly or enable network access."
            )
        winner = report["winner"]["model_id"]
        self.logger.info("Selected SLM for hardware: %s", winner)
        return winner

    def discover_teacher_model(self, hw_specs: dict[str, Any], *, limit: int = 80) -> str:
        """Select the best instruction-following model for Teacher/augmentation.

        Teacher models are used for *inference only*, so we can apply 4-bit
        quantization and select models that are 2-3x larger than what we would
        choose for LoRA fine-tuning. This maximises the quality of synthetic
        audit data without any training-time memory overhead.

        Capacity formula (4-bit, 0.5 GB per B params):
            max_params = (vram_gb - 4_overhead) * 2

        Examples:
            A100 40 GB → up to ~72 B params  (selects Llama-3.1-70B-Instruct)
            RTX 16 GB  → up to ~24 B params  (selects a 13-14 B model)
            RTX  8 GB  → up to  ~8 B params  (selects a 7 B model)
        """
        self.logger.info(
            "Searching for the best Teacher model | backend=%s device=%s vram=%.1fGB",
            hw_specs.get("backend"),
            hw_specs.get("device"),
            float(hw_specs.get("vram_gb", 0.0)),
        )
        report = self.rank_teacher_models(hw_specs, limit=limit)
        self._write_teacher_report(report)
        if not report["winner"]:
            fallback = "Qwen/Qwen2.5-7B-Instruct"
            self.logger.warning(
                "No Teacher candidate found (no internet or no cached model). "
                "Falling back to default: %s", fallback,
            )
            return fallback
        winner = report["winner"]["model_id"]
        self.logger.info(
            "Selected Teacher model: %s  (%.1f B params, max capacity %.1f B)",
            winner,
            report["winner"]["params_b"],
            self._max_teacher_params_for_hw(hw_specs),
        )
        return winner

    def rank_teacher_models(self, hw_specs: dict[str, Any], *, limit: int = 80) -> dict[str, Any]:
        """Rank all viable teacher candidates and return a scored report."""
        try:
            raw_models = self._fetch_hf_models(limit=limit)
            candidates = self._score_teacher_models(raw_models, hw_specs)
        except Exception as exc:
            self.logger.warning("Remote teacher search failed: %s. Trying local HF cache.", exc)
            raw_models = self._fetch_local_cached_models()
            candidates = self._score_teacher_models(raw_models, hw_specs)
            return {
                "hardware": hw_specs,
                "source": "local_cache" if candidates else "unresolved",
                "winner": max(candidates, key=lambda c: c.score).__dict__ if candidates else None,
                "candidates": [c.__dict__ for c in sorted(candidates, key=lambda c: c.score, reverse=True)[:20]],
                "error": str(exc),
            }

        if not candidates:
            local = self._score_teacher_models(self._fetch_local_cached_models(), hw_specs)
            if local:
                local.sort(key=lambda c: c.score, reverse=True)
                return {"hardware": hw_specs, "source": "local_cache",
                        "winner": local[0].__dict__,
                        "candidates": [c.__dict__ for c in local[:20]]}
            return {"hardware": hw_specs, "source": "unresolved", "winner": None,
                    "candidates": [], "error": "no suitable teacher candidates found"}

        candidates.sort(key=lambda c: c.score, reverse=True)
        return {
            "hardware": hw_specs,
            "source": "huggingface_api",
            "winner": candidates[0].__dict__,
            "candidates": [c.__dict__ for c in candidates[:20]],
        }

    def _max_teacher_params_for_hw(self, hw_specs: dict[str, Any]) -> float:
        """Maximum teacher params assuming 4-bit inference (0.5 GB per B param).

        vram_gb is the *free* VRAM available (from nvidia-smi), not total.
        We only need to reserve for KV-cache, activations, and driver overhead.

        Formula: (free_vram_gb - 3_overhead) * 2
        Examples:
            40 GB free → up to ~74 B params  (can fit 70B safely)
            24 GB free → up to ~42 B params  (fits 32B comfortably)
            16 GB free → up to ~26 B params  (fits 13B comfortably)
             8 GB free → up to ~10 B params  (fits 7B safely)
        """
        device = hw_specs.get("device")
        vram = float(hw_specs.get("vram_gb", 0.0))
        ram  = float(hw_specs.get("ram_gb", 0.0))
        if device == "cuda":
            return max(1.0, (vram - 3.0) * 2.0)   # 4-bit: 0.5 GB/B, 3GB overhead
        if device in {"mps", "xpu"}:
            usable = min(vram, ram) if vram > 0 else ram * 0.6
            return max(1.0, (usable - 3.0) * 2.0)
        # CPU: full-precision (2 GB/B BF16), conservative
        return max(1.0, (ram * 0.45 - 2.0) / 2.0)

    def _score_teacher_models(
        self, models: list[dict[str, Any]], hw_specs: dict[str, Any]
    ) -> list[ModelCandidateScore]:
        """Score models for teacher use: maximise size + quality family bonus."""
        max_params = self._max_teacher_params_for_hw(hw_specs)
        rows: list[ModelCandidateScore] = []

        for model in models:
            model_id = str(model.get("modelId", ""))
            lowered  = model_id.lower()

            if not model_id or not any(m in lowered for m in INSTRUCT_MARKERS):
                continue
            if any(m in lowered for m in BAD_MARKERS):
                continue

            params_b = self._extract_params_b(model)
            if params_b <= 0.0 or params_b > max_params:
                continue

            is_gated = model.get("gated") is True
            is_cached = model.get("local_cached") is True

            context_length = self._extract_context_length(model)
            downloads = int(model.get("downloads") or 0)
            likes     = int(model.get("likes") or 0)
            score     = 1.0   # base: passes instruct gate
            reasons   = ["instruction_tuned"]

            # --- Size score: maximise within capacity (teacher = bigger is better)
            size_score = params_b / max(max_params, 1.0)
            score += size_score * 5.0
            reasons.append(f"{params_b:.1f}B")

            # --- Quality family bonus
            for family, bonus in QUALITY_FAMILIES.items():
                if family in lowered:
                    score += bonus
                    reasons.append(f"family:{family}")
                    break   # take the highest-priority match only

            # --- Long context bonus (teacher generates up to 2048 tokens)
            if context_length and context_length >= 8192:
                score += min(1.5, context_length / 65536.0)
                reasons.append(f"context={context_length}")
            elif not context_length:
                score -= 0.5   # unknown context — mild penalty

            # --- Popularity & freshness signals
            score += min(1.5, downloads / 3_000_000.0)
            score += min(0.8, likes / 2000.0)
            score += self._recency_bonus(model)

            if model.get("local_cached"):
                score += 0.5   # prefer already-downloaded models
                reasons.append("local_cache")

            rows.append(ModelCandidateScore(
                model_id=model_id,
                score=round(score, 4),
                params_b=params_b,
                downloads=downloads,
                likes=likes,
                context_length=context_length,
                reason=", ".join(reasons),
                gated=is_gated,
            ))
        return rows

    def _write_teacher_report(self, report: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / "teacher_advisor_last_report.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("Could not write teacher advisor report: %s", exc)

    def rank_models(self, hw_specs: dict[str, Any], *, limit: int = 80) -> dict[str, Any]:
        self.logger.info(
            "Searching Hugging Face for SLM candidates | backend=%s device=%s vram=%.1fGB",
            hw_specs.get("backend"),
            hw_specs.get("device"),
            float(hw_specs.get("vram_gb", 0.0)),
        )
        try:
            raw_models = self._fetch_hf_models(limit=limit)
            candidates = self._score_models(raw_models, hw_specs)
        except Exception as exc:
            self.logger.warning("Remote model search failed: %s. Trying local HF cache discovery.", exc)
            raw_models = self._fetch_local_cached_models()
            candidates = self._score_models(raw_models, hw_specs)
            return {
                "hardware": hw_specs,
                "source": "local_cache" if candidates else "unresolved",
                "winner": max(candidates, key=lambda item: item.score).__dict__ if candidates else None,
                "candidates": [item.__dict__ for item in sorted(candidates, key=lambda item: item.score, reverse=True)[:20]],
                "error": str(exc),
            }

        if not candidates:
            local_candidates = self._score_models(self._fetch_local_cached_models(), hw_specs)
            if local_candidates:
                local_candidates.sort(key=lambda item: item.score, reverse=True)
                return {
                    "hardware": hw_specs,
                    "source": "local_cache",
                    "winner": local_candidates[0].__dict__,
                    "candidates": [item.__dict__ for item in local_candidates[:20]],
                }
            return {
                "hardware": hw_specs,
                "source": "unresolved",
                "winner": None,
                "candidates": [],
                "error": "no suitable Hugging Face or local cached candidates",
            }

        candidates.sort(key=lambda item: item.score, reverse=True)
        winner = candidates[0]
        return {
            "hardware": hw_specs,
            "source": "huggingface_api",
            "winner": winner.__dict__,
            "candidates": [item.__dict__ for item in candidates[:20]],
        }

    def _fetch_hf_models(self, *, limit: int) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        searches = ["instruct", "chat", "text-generation"]
        for query in searches:
            response = requests.get(
                self.hf_api_url,
                params={
                    "search": query,
                    "filter": "text-generation",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": limit,
                    "full": "true",
                },
                timeout=20,
                headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else None
            )
            response.raise_for_status()
            for model in response.json():
                model_id = str(model.get("modelId", ""))
                if model_id:
                    seen[model_id] = model
        return list(seen.values())

    def _fetch_local_cached_models(self) -> list[dict[str, Any]]:
        if not self.model_cache_dir:
            return []
        hub_dir = self.model_cache_dir / "hub"
        if not hub_dir.exists():
            return []

        rows: list[dict[str, Any]] = []
        for model_dir in hub_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
                continue
            model_id = model_dir.name.replace("models--", "").replace("--", "/")
            config = self._load_cached_config(model_dir)
            rows.append(
                {
                    "modelId": model_id,
                    "downloads": 0,
                    "likes": 0,
                    "lastModified": None,
                    "config": config,
                    "cardData": {},
                    "local_cached": True,
                }
            )
        return rows

    def _load_cached_config(self, model_dir: Path) -> dict[str, Any]:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return {}
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        snapshots.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            config_path = snapshot / "config.json"
            if config_path.exists():
                try:
                    return json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    return {}
        return {}

    def _score_models(self, models: list[dict[str, Any]], hw_specs: dict[str, Any]) -> list[ModelCandidateScore]:
        max_params = self._max_params_for_hw(hw_specs)
        min_context = 4096
        rows: list[ModelCandidateScore] = []
        for model in models:
            model_id = str(model.get("modelId", ""))
            lowered = model_id.lower()
            if not model_id or not any(marker in lowered for marker in INSTRUCT_MARKERS):
                continue
            if any(marker in lowered for marker in BAD_MARKERS):
                continue

            params_b = self._extract_params_b(model)
            if params_b <= 0.0 or params_b > max_params:
                continue

            is_gated = model.get("gated") is True
            is_cached = model.get("local_cached") is True

            context_length = self._extract_context_length(model)
            downloads = int(model.get("downloads") or 0)
            likes = int(model.get("likes") or 0)
            score = 0.0
            reasons = []

            if any(marker in lowered for marker in INSTRUCT_MARKERS):
                score += 1.0
                reasons.append("instruction_tuned")

            size_score = 1.0 - abs(params_b - self._target_params_for_hw(hw_specs)) / max(max_params, 1.0)
            score += max(0.0, size_score) * 3.0

            if context_length and context_length >= min_context:
                score += min(2.0, context_length / 32768.0)
                reasons.append(f"context={context_length}")
            else:
                score -= 2.0
                reasons.append("context_unknown_or_short")

            if params_b < 2.0 and float(hw_specs.get("vram_gb", 0.0)) >= 7.0:
                score -= 1.5
                reasons.append("underuses_available_vram")

            score += min(2.0, downloads / 2_000_000.0)
            score += min(1.0, likes / 1000.0)
            score += self._recency_bonus(model)
            if model.get("local_cached"):
                score += 0.25
                reasons.append("local_cache")

            rows.append(
                ModelCandidateScore(
                    model_id=model_id,
                    score=round(score, 4),
                    params_b=params_b,
                    downloads=downloads,
                    likes=likes,
                    context_length=context_length,
                    reason=", ".join(reasons) or "fits_hardware",
                    gated=is_gated,
                )
            )
        return rows

    def _max_params_for_hw(self, hw_specs: dict[str, Any]) -> float:
        device = hw_specs.get("device")
        vram = float(hw_specs.get("vram_gb", 0.0))
        ram = float(hw_specs.get("ram_gb", 0.0))
        if device == "cuda":
            if vram >= 20:
                return 14.0
            if vram >= 10:
                return 7.5
            if vram >= 7:
                return 4.5
            return 2.0
        if device in {"mps", "xpu"}:
            return 7.5 if ram >= 32 else 4.5
        return 3.0 if ram >= 24 else 1.5

    def _target_params_for_hw(self, hw_specs: dict[str, Any]) -> float:
        max_params = self._max_params_for_hw(hw_specs)
        return max(1.0, max_params * 0.75)

    def _extract_params_b(self, model: dict[str, Any]) -> float:
        model_id = str(model.get("modelId", ""))
        match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", model_id)
        if match:
            return float(match.group(1))
        card_data = model.get("cardData") or {}
        for key in ("model_size", "parameters", "params"):
            value = card_data.get(key)
            if value:
                match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", str(value))
                if match:
                    return float(match.group(1))
        return 0.0

    def _extract_context_length(self, model: dict[str, Any]) -> int | None:
        card_data = model.get("cardData") or {}
        config = model.get("config") or {}
        model_id = str(model.get("modelId", ""))
        candidates = [
            card_data.get("context_length"),
            card_data.get("context length"),
            card_data.get("max_position_embeddings"),
            card_data.get("max_sequence_length"),
            config.get("max_position_embeddings"),
            config.get("seq_length"),
            config.get("max_seq_len"),
        ]
        for value in candidates:
            if value is None:
                continue
            match = re.search(r"(\d+)", str(value).replace(",", ""))
            if match:
                return int(match.group(1))
        readme = self._fetch_readme(model_id)
        if readme:
            patterns = [
                r"context length[^\d]{0,80}([\d,]+)",
                r"context-length[^\d]{0,80}([\d,]+)",
                r"max-model-len[^\d]{0,80}([\d,]+)",
                r"context[:\s]+(?:length[:\s]+)?[^\d]{0,80}([\d,]+)",
                r"long-context.*?([\d,]+)\s*tokens",
                r"support(?:s)? up to ([\d,]+)\s*tokens",
            ]
            lowered = readme.lower()
            matches: list[int] = []
            for pattern in patterns:
                for match in re.finditer(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                    try:
                        matches.append(int(match.group(1).replace(",", "")))
                    except Exception:
                        continue
            if matches:
                return max(matches)
        return None

    def _fetch_readme(self, model_id: str) -> str:
        if not model_id:
            return ""
        if model_id in self._readme_cache:
            return self._readme_cache[model_id]
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                self._readme_cache[model_id] = response.text[:120_000]
            else:
                self._readme_cache[model_id] = ""
        except Exception:
            self._readme_cache[model_id] = ""
        return self._readme_cache[model_id]

    def _recency_bonus(self, model: dict[str, Any]) -> float:
        modified = model.get("lastModified")
        if not modified:
            return 0.0
        try:
            timestamp = datetime.fromisoformat(str(modified).replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - timestamp).days
            if age_days <= 180:
                return 1.0
            if age_days <= 365:
                return 0.5
        except Exception:
            return 0.0
        return 0.0

    def _write_report(self, report: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / "model_advisor_last_report.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("Could not write model advisor report: %s", exc)
