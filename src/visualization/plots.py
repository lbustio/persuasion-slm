import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils.paths import get_project_layout


class PlotManager:
    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = get_project_layout().outputs_figures
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

    def _save_all_formats(self, fig, name: str, data: dict | None):
        png_path = self.output_dir / f"{name}.png"
        eps_path = self.output_dir / f"{name}.eps"
        csv_path = self.output_dir / f"{name}_data.csv"

        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

        if data:
            keys = list(data.keys())
            length = len(data[keys[0]])
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for i in range(length):
                    row = [data[k][i] for k in keys]
                    writer.writerow(row)

    def plot_learning_curve(
        self,
        epochs: list,
        train_loss: list,
        val_loss: list,
        name: str = "learning_curve",
        train_x: list | None = None,
        val_x: list | None = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        train_axis = train_x if train_x is not None else epochs
        val_axis = val_x if val_x is not None else epochs[: len(val_loss)]

        ax.plot(train_axis, train_loss, label="Train Loss", marker="o")
        if val_loss:
            ax.plot(val_axis, val_loss, label="Validation Loss", marker="s")
        ax.set_title("Learning Curve")
        ax.set_xlabel("Epochs / Steps")
        ax.set_ylabel("Loss")
        ax.legend()

        data = {
            "series": [],
            "x": [],
            "loss": [],
        }
        for x_value, loss_value in zip(train_axis, train_loss):
            data["series"].append("train")
            data["x"].append(x_value)
            data["loss"].append(loss_value)
        for x_value, loss_value in zip(val_axis, val_loss):
            data["series"].append("validation")
            data["x"].append(x_value)
            data["loss"].append(loss_value)

        self._save_all_formats(fig, name, data)

    def plot_confusion_matrix(self, cm: np.ndarray, labels: list, name: str = "confusion_matrix"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        data = {
            "actual_label": [],
            "predicted_label": [],
            "count": [],
        }
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                data["actual_label"].append(actual)
                data["predicted_label"].append(predicted)
                data["count"].append(cm[i, j])

        self._save_all_formats(fig, name, data)

    def plot_binary_confusion_matrix(self, cm: np.ndarray, labels: list, title: str, name: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        data = {
            "true_label": [],
            "predicted_label": [],
            "count": [],
        }
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                data["true_label"].append(actual)
                data["predicted_label"].append(predicted)
                data["count"].append(int(cm[i, j]))

        self._save_all_formats(fig, name, data)

    def plot_metric_bar(self, classes: list, metric_values: list, metric_name: str, name: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=classes, y=metric_values, ax=ax, palette="viridis")
        ax.set_title(f"{metric_name} per Class")
        ax.set_xlabel("Classes")
        ax.set_ylabel(metric_name)

        data = {
            "class": classes,
            metric_name: metric_values,
        }
        self._save_all_formats(fig, name, data)

    def plot_grouped_bar(self, categories: list, series: dict[str, list], title: str, xlabel: str, ylabel: str, name: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.8 / max(len(series), 1)

        for idx, (series_name, values) in enumerate(series.items()):
            offset = (idx - (len(series) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=series_name)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.legend()

        data = {"category": categories}
        data.update(series)
        self._save_all_formats(fig, name, data)

    def plot_multi_series_line(self, x_values: list, series: dict[str, list], title: str, xlabel: str, ylabel: str, name: str):
        fig, ax = plt.subplots(figsize=(12, 6))

        for series_name, values in series.items():
            ax.plot(x_values, values, marker="o", label=series_name)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        data = {"x": x_values}
        data.update(series)
        self._save_all_formats(fig, name, data)

    def plot_curve_series(self, curves: dict[str, tuple[list, list]], title: str, xlabel: str, ylabel: str, name: str):
        fig, ax = plt.subplots(figsize=(10, 6))

        data = {"series": [], xlabel: [], ylabel: []}
        for series_name, (x_values, y_values) in curves.items():
            ax.plot(x_values, y_values, label=series_name)
            data["series"].extend([series_name] * len(x_values))
            data[xlabel].extend(x_values)
            data[ylabel].extend(y_values)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        self._save_all_formats(fig, name, data)
