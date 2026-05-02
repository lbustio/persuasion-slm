import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from src.utils.logger import setup_logger
from src.utils.paths import get_project_layout
from src.utils.state import StateManager


MASTER_SPLIT_SCHEMA_VERSION = "master_split_v1"
SPLIT_NAMES = ("train", "validation", "test", "heldout_final")


class MasterSplitManager:
    """Create one stable message-level split shared by all downstream phases."""

    def __init__(self, run_id: str = None):
        self.logger = setup_logger("split_manager", run_id=run_id)
        self.layout = get_project_layout()
        with open(self.layout.root_dir / "configs" / "training_defaults.yaml", "r", encoding="utf-8") as handle:
            self.training_defaults = yaml.safe_load(handle)
        self.state_mgr = StateManager(self.layout.outputs_checkpoints)
        self.output_dir = self.layout.outputs_splits
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, dataset_path: str | Path, force_restart: bool = False) -> Path:
        task_name = "master_split"
        manifest_path = self.output_dir / "master_split_manifest.json"

        if not force_restart and self.state_mgr.is_completed(task_name) and self._manifest_is_valid(manifest_path):
            self.logger.info("Master split already exists. Reusing %s.", manifest_path)
            return manifest_path

        if force_restart:
            self.logger.info("Fresh master split requested. Rebuilding split files.")

        records = self._load_records(Path(dataset_path))
        if not records:
            raise ValueError("Cannot create master split from an empty dataset.")

        ratios = self._load_ratios()
        grouped = self._group_records(records)
        groups = list(grouped.values())
        seed = int(self.training_defaults.get("seed", 42))
        rng = random.Random(seed)
        groups.sort(key=lambda rows: self._group_key(rows[0]))
        rng.shuffle(groups)

        split_groups = self._assign_groups(groups, ratios)
        split_files = {}
        split_counts = {}
        for split_name in SPLIT_NAMES:
            rows = []
            for group in split_groups[split_name]:
                rows.extend(group)
            rows.sort(key=lambda item: str(item.get("id", "")))
            split_path = self.output_dir / f"master_{split_name}.jsonl"
            self._write_jsonl(split_path, rows, split_name)
            split_files[split_name] = str(split_path)
            split_counts[split_name] = len(rows)

        manifest = {
            "schema_version": MASTER_SPLIT_SCHEMA_VERSION,
            "dataset_path": str(Path(dataset_path)),
            "seed": seed,
            "ratios": ratios,
            "split_files": split_files,
            "split_counts": split_counts,
            "total_records": len(records),
            "group_count": len(groups),
            "leakage_guard": "Records with identical normalized text are assigned to the same split.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        self.state_mgr.mark_completed(task_name, {"manifest_path": str(manifest_path), **manifest})
        self.logger.info("Master split created: %s", manifest_path)
        return manifest_path

    def load_manifest(self, manifest_path: str | Path | None = None) -> dict:
        path = Path(manifest_path) if manifest_path else self.output_dir / "master_split_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _manifest_is_valid(self, manifest_path: Path) -> bool:
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != MASTER_SPLIT_SCHEMA_VERSION:
                return False
            return all(Path(path).exists() for path in manifest.get("split_files", {}).values())
        except Exception:
            return False

    def _load_records(self, dataset_path: Path) -> list[dict]:
        records = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _load_ratios(self) -> dict[str, float]:
        data_cfg = self.training_defaults["data"]
        heldout = float(data_cfg.get("heldout_final_ratio", 0.0))
        if heldout < 0.0 or heldout >= 1.0:
            raise ValueError("heldout_final_ratio must be >= 0 and < 1.")

        base = {
            "train": float(data_cfg["train_ratio"]),
            "validation": float(data_cfg["validation_ratio"]),
            "test": float(data_cfg["test_ratio"]),
        }
        base_total = sum(base.values())
        if abs(base_total - 1.0) > 1e-9:
            raise ValueError(f"train/validation/test ratios must sum to 1.0, got {base_total}.")

        available = 1.0 - heldout
        ratios = {key: value * available for key, value in base.items()}
        ratios["heldout_final"] = heldout
        return ratios

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).lower().split())

    def _group_key(self, record: dict) -> str:
        normalized = self._normalize_text(record.get("text", ""))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _group_records(self, records: list[dict]) -> dict[str, list[dict]]:
        grouped = defaultdict(list)
        for record in records:
            grouped[self._group_key(record)].append(record)
        return grouped

    def _assign_groups(self, groups: list[list[dict]], ratios: dict[str, float]) -> dict[str, list[list[dict]]]:
        total = sum(len(group) for group in groups)
        targets = {name: total * ratios[name] for name in SPLIT_NAMES}
        assigned = {name: [] for name in SPLIT_NAMES}
        counts = {name: 0 for name in SPLIT_NAMES}

        ordered_splits = ("heldout_final", "test", "validation", "train")
        for group in groups:
            eligible = [
                name
                for name in ordered_splits
                if targets[name] > 0 and counts[name] < targets[name]
            ]
            if not eligible:
                eligible = ["train"]
            split_name = min(eligible, key=lambda name: counts[name] / max(targets[name], 1.0))
            assigned[split_name].append(group)
            counts[split_name] += len(group)

        return assigned

    def _write_jsonl(self, path: Path, records: list[dict], split_name: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                payload = {**record, "split": split_name, "split_schema_version": MASTER_SPLIT_SCHEMA_VERSION}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
