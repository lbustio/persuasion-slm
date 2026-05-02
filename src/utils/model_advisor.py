from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


INSTRUCT_MARKERS = ("instruct", "chat", "it")
BAD_MARKERS = ("embedding", "reranker", "vision", "vl", "coder", "gguf", "awq", "gptq")


@dataclass
class ModelCandidateScore:
    model_id: str
    score: float
    params_b: float
    downloads: int
    likes: int
    context_length: int | None
    reason: str


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
