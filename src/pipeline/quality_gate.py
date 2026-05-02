from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QualityGateReport:
    schema_version: str
    total_records: int = 0
    failed_records: int = 0
    warnings: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    sample_failures: list[dict[str, Any]] = field(default_factory=list)
    failed_record_indices: set[int] = field(default_factory=set, repr=False)

    @property
    def passed(self) -> bool:
        return self.failed_records == 0 and self.total_records > 0

    def add_warning(self, key: str) -> None:
        self.warnings[key] = self.warnings.get(key, 0) + 1

    def add_failure(self, key: str, record_index: int, source_id: str = "") -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if record_index not in self.failed_record_indices:
            self.failed_record_indices.add(record_index)
            self.failed_records += 1
        if len(self.sample_failures) < 25:
            self.sample_failures.append(
                {
                    "record_index": record_index,
                    "source_id": source_id,
                    "reason": key,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_records": self.total_records,
            "failed_records": self.failed_records,
            "passed": self.passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "sample_failures": self.sample_failures,
        }


class AugmentedDatasetQualityGate:
    def __init__(self, expected_schema: str):
        self.expected_schema = expected_schema

    def validate_file(self, dataset_path: str | Path, report_path: str | Path | None = None) -> QualityGateReport:
        dataset_path = Path(dataset_path)
        report = QualityGateReport(schema_version=self.expected_schema)

        if not dataset_path.exists() or dataset_path.stat().st_size == 0:
            report.add_failure("dataset_missing_or_empty", 0)
            self._write_report(report, report_path)
            return report

        with open(dataset_path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                report.total_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    report.add_failure("invalid_json", index)
                    continue
                self.validate_record(record, report, index)

        self._write_report(report, report_path)
        return report

    def validate_record(self, record: dict[str, Any], report: QualityGateReport, index: int) -> None:
        source_id = str(record.get("source_id", ""))
        failure_before = report.failed_records

        if record.get("schema_version") != self.expected_schema:
            report.add_failure("wrong_schema_version", index, source_id)

        if not self._is_ascii_json(record):
            report.add_failure("non_ascii_payload", index, source_id)

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            report.add_failure("invalid_messages_shape", index, source_id)
        else:
            expected_roles = ["system", "user", "assistant"]
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if roles != expected_roles:
                report.add_failure("invalid_message_roles", index, source_id)
            assistant_text = str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else ""
            for marker in ("Conclusion:", "Juicio de ciberseguridad:", "Principios evaluados:", "Limite:"):
                if marker not in assistant_text:
                    report.add_failure(f"missing_marker_{marker.rstrip(':').lower().replace(' ', '_')}", index, source_id)

        for key in (
            "source_id",
            "source",
            "source_split",
            "is_phishing",
            "phishing_label_source",
            "generation_source",
            "classifier_hypothesis",
            "quality",
        ):
            if key not in record:
                report.add_failure(f"missing_field_{key}", index, source_id)

        if record.get("source_split") not in {"train", "validation"}:
            report.add_failure("invalid_source_split", index, source_id)

        if record.get("is_phishing") not in {0, 1}:
            report.add_failure("invalid_is_phishing", index, source_id)

        quality = record.get("quality", {})
        if isinstance(quality, dict):
            if quality.get("passed") is not True:
                report.add_failure("record_quality_failed", index, source_id)
        else:
            report.add_failure("invalid_quality_payload", index, source_id)

        if record.get("generation_source") == "dummy_fallback":
            report.add_failure("dummy_fallback_record", index, source_id)

        if report.failed_records == failure_before and record.get("generation_source") == "teacher_generated":
            report.add_warning("teacher_generated_record")

    def _is_ascii_json(self, payload: object) -> bool:
        try:
            return json.dumps(payload, ensure_ascii=False).isascii()
        except TypeError:
            return False

    def _write_report(self, report: QualityGateReport, report_path: str | Path | None) -> None:
        if report_path is None:
            return
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
