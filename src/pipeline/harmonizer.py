import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

from src.utils.logger import setup_logger
from src.utils.paths import get_project_layout
from src.utils.state import StateManager

PERSUASION_PRINCIPLES = [
    "authority",
    "social_proof",
    "liking_similarity_deception",
    "commitment_integrity_reciprocation",
    "distraction",
]

HARMONIZED_SCHEMA_VERSION = "harmonized_v2"


class DataHarmonizer:
    def __init__(self, run_id: str = None):
        self.logger = setup_logger("harmonizer", run_id=run_id)
        self.layout = get_project_layout()
        self.state_mgr = StateManager(self.layout.outputs_checkpoints)
        self.output_dir = self.layout.outputs_artifacts
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        data_dir: str | Path = "data",
        force_restart: bool = False,
        manual_paths: list[str | Path] | None = None,
    ) -> Path:
        task_name = "data_harmonization"

        dataset_paths = self._discover_dataset_paths(data_dir, manual_paths)
        input_fingerprint = self._input_fingerprint(dataset_paths)

        if not force_restart and self.state_mgr.is_completed(task_name):
            state = self.state_mgr.load_state(task_name) or {}
            result = state.get("result", {})
            out_file = Path(result.get("output_file", ""))
            is_current = (
                result.get("schema_version") == HARMONIZED_SCHEMA_VERSION
                and result.get("input_fingerprint") == input_fingerprint
            )
            if is_current and out_file.exists() and out_file.stat().st_size > 0:
                self.logger.info(
                    "Data harmonization already completed (%s records). Resuming from checkpoint.",
                    result["total_records"],
                )
                return out_file
            self.logger.warning("Harmonization checkpoint is stale or invalid. Reprocessing source CSV files.")
        elif force_restart:
            self.logger.info("Fresh harmonization requested. Rebuilding artifacts from source datasets.")

        self.logger.info("=== PHASE 1: Dataset harmonization ===")
        records: list[dict] = []

        for dataset_path in dataset_paths:
            dataset_kind, df = self._load_and_identify_dataset(dataset_path)
            if dataset_kind == "iwspa":
                records.extend(self._process_iwspa(dataset_path, df))
            elif dataset_kind == "spaphish":
                records.extend(self._process_spaphish(dataset_path, df))
            else:
                self.logger.warning("CSV is not a supported dataset: %s. Skipping.", dataset_path)

        if not records:
            self.logger.error("No records were loaded. Check the dataset directory.")
            raise FileNotFoundError("No data available for processing.")

        total = len(records)
        phishing = sum(1 for r in records if r["is_phishing"])
        self.logger.info(
            "Unified records: %s | Phishing: %s | Legitimate: %s",
            total,
            phishing,
            total - phishing,
        )
        for principle in PERSUASION_PRINCIPLES:
            count = sum(1 for r in records if r["labels"][principle] == 1)
            self.logger.info("  [%s]: %s positives (%.1f%%)", principle, count, count / total * 100)

        output_file = self.output_dir / "harmonized_dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        df_out = pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "schema_version": r.get("schema_version", HARMONIZED_SCHEMA_VERSION),
                    "source": r["source"],
                    "dataset_file": r.get("dataset_file", ""),
                    "lang": r["lang"],
                    "is_phishing": r["is_phishing"],
                    "is_phishing_inferred": r.get("is_phishing_inferred", 0),
                    "phishing_label_source": r.get("phishing_label_source", "unknown"),
                    "text_len": len(r["text"]),
                    **r["labels"],
                }
                for r in records
            ]
        )
        df_out.to_csv(self.output_dir / "harmonized_dataset_summary.csv", index=False)

        self.logger.info("Artifacts saved in: %s", self.output_dir)
        self.state_mgr.mark_completed(
            task_name,
            {
                "output_file": str(output_file),
                "schema_version": HARMONIZED_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "total_records": total,
                "phishing": phishing,
                "legitimate": total - phishing,
                "input_files": [str(path) for path in dataset_paths],
            },
        )
        return output_file

    def _to_ascii(self, value: object) -> str:
        text = str(value)
        for _ in range(3):
            try:
                candidate = text.encode("cp1252").decode("utf-8")
            except UnicodeError:
                break
            if candidate == text:
                break
            text = candidate
        replacements = {
            "\u201c": "\"",
            "\u201d": "\"",
            "\u2018": "'",
            "\u2019": "'",
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u2022": "*",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        return text.encode("ascii", "ignore").decode("ascii")

    def _discover_dataset_paths(
        self,
        data_dir: str | Path,
        manual_paths: list[str | Path] | None = None,
    ) -> list[Path]:
        if manual_paths:
            paths = [Path(path) for path in manual_paths]
        else:
            root = Path(data_dir)
            paths = sorted(path for path in root.glob("*.csv") if path.is_file())

        existing = []
        for path in paths:
            if path.exists():
                existing.append(path)
            else:
                self.logger.warning("Dataset CSV no encontrado: %s. Saltando.", path)
        if not existing:
            raise FileNotFoundError(f"No CSV datasets found in {data_dir}.")
        self.logger.info("Dataset CSVs discovered: %s", [str(path) for path in existing])
        return existing

    def _input_fingerprint(self, paths: list[Path]) -> str:
        hasher = hashlib.sha256()
        for path in sorted(paths, key=lambda item: str(item).lower()):
            stat = path.stat()
            hasher.update(str(path.resolve()).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        return hasher.hexdigest()

    def _read_csv_auto(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip", encoding="utf-8")
            return self._normalize_columns(df)
        except Exception:
            for sep in (",", ";"):
                try:
                    df = pd.read_csv(path, sep=sep, on_bad_lines="skip", encoding="utf-8")
                    return self._normalize_columns(df)
                except Exception:
                    continue
        raise ValueError(f"Could not read CSV: {path}")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._to_ascii(str(column).replace("\ufeff", "")).strip() for column in df.columns]
        return df

    def _load_and_identify_dataset(self, path: Path) -> tuple[str, pd.DataFrame]:
        df = self._read_csv_auto(path)
        columns = set(df.columns)
        if {"subject", "txt", *PERSUASION_PRINCIPLES}.issubset(columns):
            return "iwspa", df
        if {"subject", "body", "Label", *PERSUASION_PRINCIPLES}.issubset(columns):
            return "spaphish", df
        return "unknown", df

    def _process_iwspa(self, path: Path, df: pd.DataFrame | None = None) -> list[dict]:
        self.logger.info("Loading IWSPA from %s ...", path)
        df = df if df is not None else self._read_csv_auto(path)
        self.logger.info("  IWSPA: %s rows loaded. Columns: %s", len(df), df.columns.tolist())

        records = []
        skipped = 0
        for _, row in df.iterrows():
            subject = self._to_ascii(row.get("subject", "")).strip()
            body = self._to_ascii(row.get("txt", "")).strip()
            text = f"{subject}\n\n{body}".strip()

            if not text or text == "nan":
                skipped += 1
                continue

            labels = {}
            for principle in PERSUASION_PRINCIPLES:
                raw = row.get(principle, 0)
                labels[principle] = int(raw) if not pd.isna(raw) else 0

            is_phishing, phishing_source, label_text = self._extract_iwspa_phishing_label(row, labels)
            records.append(
                {
                    "schema_version": HARMONIZED_SCHEMA_VERSION,
                    "id": self._to_ascii(row.get("id", "")).strip(),
                    "source": "IWSPA",
                    "dataset_file": path.name,
                    "lang": "en",
                    "is_phishing": is_phishing,
                    "is_phishing_inferred": 1 if phishing_source == "inferred_from_persuasion_labels" else 0,
                    "phishing_label_source": phishing_source,
                    "phishing_label_text": label_text,
                    "text": text,
                    "labels": labels,
                    "justifications": {},
                    "annotation_details": {},
                }
            )

        self.logger.info("  IWSPA: %s valid records processed (%s skipped).", len(records), skipped)

        df_clean = pd.DataFrame(
            [
                {
                    **{
                        "id": r["id"],
                        "schema_version": r.get("schema_version", HARMONIZED_SCHEMA_VERSION),
                        "source": r["source"],
                        "dataset_file": r.get("dataset_file", ""),
                        "lang": r["lang"],
                        "is_phishing": r["is_phishing"],
                        "is_phishing_inferred": r.get("is_phishing_inferred", 0),
                        "phishing_label_source": r.get("phishing_label_source", "unknown"),
                    },
                    **r["labels"],
                }
                for r in records
            ]
        )
        df_clean.to_csv(self.output_dir / "iwspa_cleaned.csv", index=False)
        return records

    def _extract_iwspa_phishing_label(self, row, labels: dict) -> tuple[int, str, str]:
        if "class" in row and not pd.isna(row.get("class")):
            return int(row.get("class")), "dataset_label", self._to_ascii(row.get("label", row.get("class", "")))
        if "is_phishing" in row and not pd.isna(row.get("is_phishing")):
            return int(row.get("is_phishing")), "dataset_label", self._to_ascii(row.get("is_phishing"))
        if "Label" in row and not pd.isna(row.get("Label")):
            raw = row.get("Label")
            if str(raw).strip().isdigit():
                return int(raw), "dataset_label", self._to_ascii(raw)
        inferred = int(any(v == 1 for v in labels.values()))
        return inferred, "inferred_from_persuasion_labels", "inferred"

    def _process_spaphish(self, path: Path, df: pd.DataFrame | None = None) -> list[dict]:
        self.logger.info("Loading Spaphish from %s ...", path)
        df = df if df is not None else self._read_csv_auto(path)
        self.logger.info("  Spaphish: %s rows loaded. Columns: %s", len(df), df.columns.tolist())

        records = []
        skipped = 0
        for _, row in df.iterrows():
            subject = self._to_ascii(row.get("subject", "")).strip()
            body = self._to_ascii(row.get("body", "")).strip()
            text = f"{subject}\n\n{body}".strip()

            if not text or text in ("nan", "\n\n"):
                skipped += 1
                continue

            labels = {}
            for principle in PERSUASION_PRINCIPLES:
                raw = row.get(principle, 0)
                labels[principle] = int(raw) if not pd.isna(raw) else 0

            justifications = {}
            annotation_details = {}
            for principle in PERSUASION_PRINCIPLES:
                final_label = int(labels.get(principle, 0))
                matching_texts = []
                all_annotations = []
                for annotator in ["A", "B", "C"]:
                    vote_raw = row.get(f"{principle}_{annotator}", None)
                    vote = None if pd.isna(vote_raw) else int(vote_raw)
                    val = self._to_ascii(row.get(f"justif_{principle}_{annotator}", "")).strip()
                    if val and val.lower() not in ("nan", ""):
                        all_annotations.append({"annotator": annotator, "vote": vote, "justification": val})
                        if vote == final_label:
                            matching_texts.append((annotator, val))
                selected = max(matching_texts, key=lambda item: len(item[1])) if matching_texts else ("", "")
                fallback_used = False
                if not selected[1] and all_annotations:
                    fallback_used = True
                    fallback_candidates = [
                        (item["annotator"], item["justification"]) for item in all_annotations if item["justification"]
                    ]
                    selected = max(fallback_candidates, key=lambda item: len(item[1])) if fallback_candidates else ("", "")
                justifications[principle] = selected[1]
                annotation_details[principle] = {
                    "final_label": final_label,
                    "selected_annotator": selected[0],
                    "selected_justification": selected[1],
                    "selected_vote_matches_final": not fallback_used and bool(selected[1]),
                    "matching_justification_count": len(matching_texts),
                    "annotations": all_annotations,
                }

            label_raw = row.get("Label", 1)
            is_phishing = int(label_raw) if not pd.isna(label_raw) else 1
            records.append(
                {
                    "schema_version": HARMONIZED_SCHEMA_VERSION,
                    "id": self._to_ascii(row.get("hash", ""))[:16],
                    "source": "Spaphish",
                    "dataset_file": path.name,
                    "lang": "es",
                    "is_phishing": is_phishing,
                    "is_phishing_inferred": 0,
                    "phishing_label_source": "dataset_label",
                    "phishing_label_text": self._to_ascii(label_raw),
                    "text": text,
                    "labels": labels,
                    "justifications": justifications,
                    "annotation_details": annotation_details,
                }
            )

        self.logger.info("  Spaphish: %s valid records processed (%s skipped).", len(records), skipped)

        df_clean = pd.DataFrame(
            [
                {
                    **{
                        "id": r["id"],
                        "schema_version": r.get("schema_version", HARMONIZED_SCHEMA_VERSION),
                        "source": r["source"],
                        "dataset_file": r.get("dataset_file", ""),
                        "lang": r["lang"],
                        "is_phishing": r["is_phishing"],
                        "is_phishing_inferred": r.get("is_phishing_inferred", 0),
                        "phishing_label_source": r.get("phishing_label_source", "unknown"),
                    },
                    **r["labels"],
                }
                for r in records
            ]
        )
        df_clean.to_csv(self.output_dir / "spaphish_cleaned.csv", index=False)
        return records
