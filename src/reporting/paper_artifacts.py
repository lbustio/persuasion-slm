from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.utils.paths import get_project_layout
from src.visualization.plots import PlotManager


class PaperArtifactManager:
    def __init__(self, logger, plot_manager: PlotManager | None = None):
        self.logger = logger
        self.layout = get_project_layout()
        self.plot_mgr = plot_manager or PlotManager(self.layout.outputs_figures)
        self.tables_dir = self.layout.outputs_tables
        self.predictions_dir = self.layout.outputs_predictions
        self.reports_dir = self.layout.outputs_reports
        self.splits_dir = self.layout.outputs_splits

    def _artifact_stem(self, model_name: str) -> str:
        return model_name.replace("/", "_")

    def _write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, path: Path, records: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_csv_rows(self, path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["empty"])
            return

        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_scalar_csv(self, path: Path, metrics: dict):
        rows = [{"metric": key, "value": value} for key, value in metrics.items()]
        self._write_csv_rows(path, rows)

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits, -50, 50)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _labels_to_matrix(self, records: list[dict], label_keys: list[str]) -> np.ndarray:
        return np.array(
            [[float(record.get("labels", {}).get(label_key, 0)) for label_key in label_keys] for record in records],
            dtype=np.float32,
        )

    def _export_split_records(self, model_stem: str, split_name: str, records: list[dict]):
        split_path = self.splits_dir / f"{model_stem}_{split_name}.jsonl"
        self._write_jsonl(split_path, records)

    def _export_training_history(self, artifact_prefix: str, log_history: list[dict]) -> list[str]:
        history_rows = []
        for row in log_history:
            normalized = {}
            for key, value in row.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    normalized[key] = value
                else:
                    normalized[key] = str(value)
            history_rows.append(normalized)

        json_path = self.reports_dir / f"{artifact_prefix}_training_history.json"
        csv_path = self.tables_dir / f"{artifact_prefix}_training_history.csv"
        self._write_json(json_path, history_rows)
        self._write_csv_rows(csv_path, history_rows)
        return [str(json_path), str(csv_path)]

    def build_classifier_artifacts(
        self,
        model_name: str,
        label_keys: list[str],
        label_display_names: list[str],
        raw_train_records: list[dict],
        raw_validation_records: list[dict],
        raw_test_records: list[dict],
        eval_predictions,
        test_predictions,
        log_history: list[dict],
        threshold: float,
        split_ratios: dict,
        seed: int,
    ) -> dict:
        model_stem = self._artifact_stem(model_name)
        artifact_prefix = f"{model_stem}_paper"

        self._export_split_records(model_stem, "train", raw_train_records)
        self._export_split_records(model_stem, "validation", raw_validation_records)
        self._export_split_records(model_stem, "test", raw_test_records)

        generated_files = self._export_training_history(artifact_prefix, log_history)
        generated_files.extend(
            [
                str(self.splits_dir / f"{model_stem}_train.jsonl"),
                str(self.splits_dir / f"{model_stem}_validation.jsonl"),
                str(self.splits_dir / f"{model_stem}_test.jsonl"),
            ]
        )

        split_rows = []
        split_label_series = {"Train": [], "Validation": [], "Test": []}
        for split_name, records in (
            ("train", raw_train_records),
            ("validation", raw_validation_records),
            ("test", raw_test_records),
        ):
            label_matrix = self._labels_to_matrix(records, label_keys)
            split_rows.append(
                {
                    "split": split_name,
                    "samples": len(records),
                    "positive_labels_total": int(label_matrix.sum()),
                    "labels_per_sample_mean": float(label_matrix.sum(axis=1).mean()) if len(records) else 0.0,
                }
            )
            split_label_series[split_name.title()] = [int(label_matrix[:, idx].sum()) for idx in range(len(label_keys))]

        split_csv = self.tables_dir / f"{artifact_prefix}_split_summary.csv"
        split_json = self.reports_dir / f"{artifact_prefix}_split_summary.json"
        self._write_csv_rows(split_csv, split_rows)
        self._write_json(split_json, split_rows)
        generated_files.extend([str(split_csv), str(split_json)])

        self.plot_mgr.plot_grouped_bar(
            categories=label_display_names,
            series=split_label_series,
            title="Label Distribution by Split",
            xlabel="Persuasion Principle",
            ylabel="Positive Samples",
            name=f"{artifact_prefix}_label_distribution_by_split",
        )
        generated_files.extend(
            [
                str(self.layout.outputs_figures / f"{artifact_prefix}_label_distribution_by_split.png"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_label_distribution_by_split.eps"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_label_distribution_by_split_data.csv"),
            ]
        )

        eval_payload = self._build_classifier_prediction_payload(
            records=raw_validation_records,
            predictions=eval_predictions,
            label_keys=label_keys,
            label_display_names=label_display_names,
            threshold=threshold,
            split_name="validation",
            artifact_prefix=artifact_prefix,
        )
        test_payload = self._build_classifier_prediction_payload(
            records=raw_test_records,
            predictions=test_predictions,
            label_keys=label_keys,
            label_display_names=label_display_names,
            threshold=threshold,
            split_name="test",
            artifact_prefix=artifact_prefix,
        )
        generated_files.extend(eval_payload["files"])
        generated_files.extend(test_payload["files"])

        test_scores = test_payload["scores"]
        test_gold = test_payload["gold"]
        test_binary = test_payload["binary"]

        class_metrics_rows, summary_metrics = self._build_classifier_metrics(
            label_keys=label_keys,
            label_display_names=label_display_names,
            gold=test_gold,
            binary=test_binary,
            scores=test_scores,
        )

        class_metrics_csv = self.tables_dir / f"{artifact_prefix}_class_metrics.csv"
        class_metrics_json = self.reports_dir / f"{artifact_prefix}_class_metrics.json"
        summary_csv = self.tables_dir / f"{artifact_prefix}_summary_metrics.csv"
        summary_json = self.reports_dir / f"{artifact_prefix}_summary_metrics.json"
        self._write_csv_rows(class_metrics_csv, class_metrics_rows)
        self._write_json(class_metrics_json, class_metrics_rows)
        self._write_scalar_csv(summary_csv, summary_metrics)
        self._write_json(summary_json, summary_metrics)
        generated_files.extend([str(class_metrics_csv), str(class_metrics_json), str(summary_csv), str(summary_json)])

        self.plot_mgr.plot_grouped_bar(
            categories=label_display_names,
            series={
                "Precision": [row["precision"] for row in class_metrics_rows],
                "Recall": [row["recall"] for row in class_metrics_rows],
                "F1-score": [row["f1_score"] for row in class_metrics_rows],
            },
            title="Per-Class Classification Metrics",
            xlabel="Persuasion Principle",
            ylabel="Score",
            name=f"{artifact_prefix}_per_class_metrics",
        )
        generated_files.extend(
            [
                str(self.layout.outputs_figures / f"{artifact_prefix}_per_class_metrics.png"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_per_class_metrics.eps"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_per_class_metrics_data.csv"),
            ]
        )

        self._export_roc_pr_curves(
            artifact_prefix=artifact_prefix,
            label_display_names=label_display_names,
            gold=test_gold,
            scores=test_scores,
            generated_files=generated_files,
        )
        self._export_threshold_sweeps(
            artifact_prefix=artifact_prefix,
            label_keys=label_keys,
            label_display_names=label_display_names,
            gold=test_gold,
            scores=test_scores,
            generated_files=generated_files,
        )
        self._export_multilabel_confusions(
            artifact_prefix=artifact_prefix,
            label_display_names=label_display_names,
            gold=test_gold,
            binary=test_binary,
            generated_files=generated_files,
        )

        manifest = {
            "artifact_type": "classifier_paper_assets",
            "model_name": model_name,
            "seed": seed,
            "threshold": threshold,
            "split_ratios": split_ratios,
            "labels": label_keys,
            "files": sorted(set(generated_files)),
        }
        manifest_path = self.reports_dir / f"{artifact_prefix}_manifest.json"
        self._write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _build_classifier_prediction_payload(
        self,
        records: list[dict],
        predictions,
        label_keys: list[str],
        label_display_names: list[str],
        threshold: float,
        split_name: str,
        artifact_prefix: str,
    ) -> dict:
        logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        logits = np.asarray(logits)
        gold = np.asarray(predictions.label_ids, dtype=np.float32)
        scores = self._sigmoid(logits)
        binary = (scores >= threshold).astype(int)

        prediction_rows = []
        for idx, record in enumerate(records):
            row = {
                "sample_id": idx,
                "split": split_name,
                "text": record.get("text", ""),
            }
            for class_idx, label_key in enumerate(label_keys):
                row[f"gold_{label_key}"] = int(gold[idx, class_idx])
                row[f"score_{label_key}"] = float(scores[idx, class_idx])
                row[f"pred_{label_key}"] = int(binary[idx, class_idx])
            prediction_rows.append(row)

        csv_path = self.predictions_dir / f"{artifact_prefix}_{split_name}_predictions.csv"
        json_path = self.predictions_dir / f"{artifact_prefix}_{split_name}_predictions.json"
        self._write_csv_rows(csv_path, prediction_rows)
        self._write_json(json_path, prediction_rows)

        probability_rows = []
        for idx, label_name in enumerate(label_display_names):
            probability_rows.append(
                {
                    "label": label_name,
                    "mean_score": float(scores[:, idx].mean()),
                    "std_score": float(scores[:, idx].std()),
                    "positive_rate_at_threshold": float(binary[:, idx].mean()),
                }
            )
        summary_csv = self.tables_dir / f"{artifact_prefix}_{split_name}_prediction_summary.csv"
        summary_json = self.reports_dir / f"{artifact_prefix}_{split_name}_prediction_summary.json"
        self._write_csv_rows(summary_csv, probability_rows)
        self._write_json(summary_json, probability_rows)

        return {
            "gold": gold.astype(int),
            "scores": scores,
            "binary": binary,
            "files": [str(csv_path), str(json_path), str(summary_csv), str(summary_json)],
        }

    def _build_classifier_metrics(self, label_keys, label_display_names, gold, binary, scores):
        precision, recall, f1_score, support = precision_recall_fscore_support(
            gold, binary, average=None, zero_division=0
        )

        rows = []
        for idx, label_key in enumerate(label_keys):
            row = {
                "label_key": label_key,
                "label": label_display_names[idx],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1_score[idx]),
                "support": int(support[idx]),
                "roc_auc": None,
                "average_precision": None,
            }
            if len(np.unique(gold[:, idx])) > 1:
                row["roc_auc"] = float(roc_auc_score(gold[:, idx], scores[:, idx]))
                row["average_precision"] = float(average_precision_score(gold[:, idx], scores[:, idx]))
            rows.append(row)

        report = classification_report(gold, binary, target_names=label_display_names, output_dict=True, zero_division=0)

        summary = {
            "subset_accuracy": float((gold == binary).all(axis=1).mean()),
            "hamming_loss": float(hamming_loss(gold, binary)),
            "jaccard_samples": float(jaccard_score(gold, binary, average="samples", zero_division=0)),
            "f1_micro": float(report["micro avg"]["f1-score"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
            "precision_micro": float(report["micro avg"]["precision"]),
            "precision_macro": float(report["macro avg"]["precision"]),
            "precision_weighted": float(report["weighted avg"]["precision"]),
            "recall_micro": float(report["micro avg"]["recall"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "recall_weighted": float(report["weighted avg"]["recall"]),
            "support_total": int(gold.shape[0]),
        }

        valid_roc = []
        valid_ap = []
        for idx in range(gold.shape[1]):
            if len(np.unique(gold[:, idx])) > 1:
                valid_roc.append(float(roc_auc_score(gold[:, idx], scores[:, idx])))
                valid_ap.append(float(average_precision_score(gold[:, idx], scores[:, idx])))
        if valid_roc:
            summary["roc_auc_macro_valid"] = float(np.mean(valid_roc))
        if valid_ap:
            summary["average_precision_macro_valid"] = float(np.mean(valid_ap))

        return rows, summary

    def _export_roc_pr_curves(self, artifact_prefix, label_display_names, gold, scores, generated_files):
        roc_curves = {}
        pr_curves = {}
        for idx, label_name in enumerate(label_display_names):
            if len(np.unique(gold[:, idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(gold[:, idx], scores[:, idx])
            precision, recall, _ = precision_recall_curve(gold[:, idx], scores[:, idx])
            roc_curves[label_name] = (fpr.tolist(), tpr.tolist())
            pr_curves[label_name] = (recall.tolist(), precision.tolist())

        if roc_curves:
            self.plot_mgr.plot_curve_series(
                curves=roc_curves,
                title="ROC Curves by Persuasion Principle",
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                name=f"{artifact_prefix}_roc_curves",
            )
            generated_files.extend(
                [
                    str(self.layout.outputs_figures / f"{artifact_prefix}_roc_curves.png"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_roc_curves.eps"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_roc_curves_data.csv"),
                ]
            )

        if pr_curves:
            self.plot_mgr.plot_curve_series(
                curves=pr_curves,
                title="Precision-Recall Curves by Persuasion Principle",
                xlabel="Recall",
                ylabel="Precision",
                name=f"{artifact_prefix}_pr_curves",
            )
            generated_files.extend(
                [
                    str(self.layout.outputs_figures / f"{artifact_prefix}_pr_curves.png"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_pr_curves.eps"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_pr_curves_data.csv"),
                ]
            )

    def _export_threshold_sweeps(self, artifact_prefix, label_keys, label_display_names, gold, scores, generated_files):
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

        for idx, label_key in enumerate(label_keys):
            rows = []
            precision_values = []
            recall_values = []
            f1_values = []

            for threshold in thresholds:
                binary = (scores[:, idx] >= threshold).astype(int)
                precision, recall, f1_score, support = precision_recall_fscore_support(
                    gold[:, idx], binary, average="binary", zero_division=0
                )
                rows.append(
                    {
                        "label_key": label_key,
                        "label": label_display_names[idx],
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1_score),
                        "support": int(gold[:, idx].sum()),
                    }
                )
                precision_values.append(float(precision))
                recall_values.append(float(recall))
                f1_values.append(float(f1_score))

            csv_path = self.tables_dir / f"{artifact_prefix}_{label_key}_threshold_sweep.csv"
            json_path = self.reports_dir / f"{artifact_prefix}_{label_key}_threshold_sweep.json"
            self._write_csv_rows(csv_path, rows)
            self._write_json(json_path, rows)
            generated_files.extend([str(csv_path), str(json_path)])

            self.plot_mgr.plot_multi_series_line(
                x_values=thresholds.tolist(),
                series={
                    "Precision": precision_values,
                    "Recall": recall_values,
                    "F1-score": f1_values,
                },
                title=f"Threshold Sweep for {label_display_names[idx]}",
                xlabel="Decision Threshold",
                ylabel="Score",
                name=f"{artifact_prefix}_{label_key}_threshold_sweep",
            )
            generated_files.extend(
                [
                    str(self.layout.outputs_figures / f"{artifact_prefix}_{label_key}_threshold_sweep.png"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_{label_key}_threshold_sweep.eps"),
                    str(self.layout.outputs_figures / f"{artifact_prefix}_{label_key}_threshold_sweep_data.csv"),
                ]
            )

    def _export_multilabel_confusions(self, artifact_prefix, label_display_names, gold, binary, generated_files):
        matrices = multilabel_confusion_matrix(gold, binary)
        rows = []
        for idx, label_name in enumerate(label_display_names):
            tn, fp, fn, tp = matrices[idx].ravel()
            rows.append(
                {
                    "label": label_name,
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )
            self.plot_mgr.plot_binary_confusion_matrix(
                cm=matrices[idx],
                labels=["Negative", "Positive"],
                title=f"Binary Confusion Matrix: {label_name}",
                name=f"{artifact_prefix}_{idx + 1:02d}_{label_name.lower().replace(' ', '_').replace(',', '')}_confusion",
            )
            generated_files.extend(
                [
                    str(
                        self.layout.outputs_figures
                        / f"{artifact_prefix}_{idx + 1:02d}_{label_name.lower().replace(' ', '_').replace(',', '')}_confusion.png"
                    ),
                    str(
                        self.layout.outputs_figures
                        / f"{artifact_prefix}_{idx + 1:02d}_{label_name.lower().replace(' ', '_').replace(',', '')}_confusion.eps"
                    ),
                    str(
                        self.layout.outputs_figures
                        / f"{artifact_prefix}_{idx + 1:02d}_{label_name.lower().replace(' ', '_').replace(',', '')}_confusion_data.csv"
                    ),
                ]
            )

        csv_path = self.tables_dir / f"{artifact_prefix}_multilabel_confusions.csv"
        json_path = self.reports_dir / f"{artifact_prefix}_multilabel_confusions.json"
        self._write_csv_rows(csv_path, rows)
        self._write_json(json_path, rows)
        generated_files.extend([str(csv_path), str(json_path)])

    def build_slm_artifacts(
        self,
        model_name: str,
        raw_train_records: list[dict],
        raw_eval_records: list[dict],
        log_history: list[dict],
        split_ratio: dict,
    ) -> dict:
        model_stem = self._artifact_stem(model_name)
        artifact_prefix = f"{model_stem}_slm_paper"

        self._export_split_records(model_stem, "slm_train", raw_train_records)
        self._export_split_records(model_stem, "slm_eval", raw_eval_records)
        generated_files = self._export_training_history(artifact_prefix, log_history)
        generated_files.extend(
            [
                str(self.splits_dir / f"{model_stem}_slm_train.jsonl"),
                str(self.splits_dir / f"{model_stem}_slm_eval.jsonl"),
            ]
        )

        train_lengths = [len(json.dumps(record.get("messages", []), ensure_ascii=False)) for record in raw_train_records]
        eval_lengths = [len(json.dumps(record.get("messages", []), ensure_ascii=False)) for record in raw_eval_records]

        summary = {
            "model_name": model_name,
            "train_samples": len(raw_train_records),
            "eval_samples": len(raw_eval_records),
            "train_message_chars_mean": float(np.mean(train_lengths)) if train_lengths else 0.0,
            "eval_message_chars_mean": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
            "train_message_chars_median": float(np.median(train_lengths)) if train_lengths else 0.0,
            "eval_message_chars_median": float(np.median(eval_lengths)) if eval_lengths else 0.0,
            "eval_ratio": split_ratio.get("eval"),
        }

        summary_csv = self.tables_dir / f"{artifact_prefix}_summary.csv"
        summary_json = self.reports_dir / f"{artifact_prefix}_summary.json"
        self._write_scalar_csv(summary_csv, summary)
        self._write_json(summary_json, summary)
        generated_files.extend([str(summary_csv), str(summary_json)])

        self.plot_mgr.plot_grouped_bar(
            categories=["Train", "Evaluation"],
            series={
                "Samples": [len(raw_train_records), len(raw_eval_records)],
                "Mean Message Characters": [
                    float(np.mean(train_lengths)) if train_lengths else 0.0,
                    float(np.mean(eval_lengths)) if eval_lengths else 0.0,
                ],
            },
            title="SLM Data Split Overview",
            xlabel="Dataset Split",
            ylabel="Value",
            name=f"{artifact_prefix}_data_overview",
        )
        generated_files.extend(
            [
                str(self.layout.outputs_figures / f"{artifact_prefix}_data_overview.png"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_data_overview.eps"),
                str(self.layout.outputs_figures / f"{artifact_prefix}_data_overview_data.csv"),
            ]
        )

        sample_rows = []
        for idx, record in enumerate(raw_eval_records[:50]):
            sample_rows.append(
                {
                    "sample_id": idx,
                    "messages_json": json.dumps(record.get("messages", []), ensure_ascii=False),
                }
            )
        samples_csv = self.tables_dir / f"{artifact_prefix}_eval_samples.csv"
        samples_json = self.reports_dir / f"{artifact_prefix}_eval_samples.json"
        self._write_csv_rows(samples_csv, sample_rows)
        self._write_json(samples_json, sample_rows)
        generated_files.extend([str(samples_csv), str(samples_json)])

        manifest = {
            "artifact_type": "slm_paper_assets",
            "model_name": model_name,
            "split_ratio": split_ratio,
            "files": sorted(set(generated_files)),
        }
        manifest_path = self.reports_dir / f"{artifact_prefix}_manifest.json"
        self._write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
