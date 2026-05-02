from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


class ProjectLayout:
    def __init__(self, root_dir: Path, config: dict):
        self.root_dir = root_dir
        self.config = config

        paths_cfg = config["paths"]
        self.cache_downloads = self.resolve(paths_cfg["caches"]["downloads"])
        self.cache_partials = self.resolve(paths_cfg["caches"]["partials"])
        self.outputs_checkpoints = self.resolve(paths_cfg["outputs"]["checkpoints"])
        self.outputs_artifacts = self.resolve(paths_cfg["outputs"]["artifacts"])
        self.outputs_splits = self.resolve(paths_cfg["outputs"]["splits"])
        self.outputs_results = self.root_dir / "outputs" / "results"
        self.outputs_models = self.resolve(paths_cfg["outputs"]["results"].get("models", "outputs/results/models"))
        self.outputs_tuned_models = self.root_dir / "outputs" / "tuned_models"
        self.outputs_figures = self.resolve(paths_cfg["outputs"]["results"].get("figures", paths_cfg["outputs"]["results"].get("metrics", "outputs/results/figures")))
        self.outputs_tables = self.resolve(paths_cfg["outputs"]["results"].get("tables", "outputs/results/tables"))
        self.outputs_predictions = self.resolve(paths_cfg["outputs"]["results"].get("predictions", "outputs/results/predictions"))
        self.outputs_reports = self.resolve(paths_cfg["outputs"]["results"].get("reports", "outputs/results/reports"))
        self.outputs_metrics = self.outputs_tables
        self.logs_runs = self.resolve(paths_cfg["logs"])

    def resolve(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else self.root_dir / path

    def rel(self, path_value: str | Path) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            path = self.root_dir / path

        try:
            return path.resolve().relative_to(self.root_dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    def ensure_runtime_dirs(self):
        for path in (
            self.cache_downloads,
            self.cache_partials,
            self.outputs_checkpoints,
            self.outputs_artifacts,
            self.outputs_splits,
            self.outputs_models,
            self.outputs_tuned_models,
            self.outputs_figures,
            self.outputs_tables,
            self.outputs_predictions,
            self.outputs_reports,
            self.outputs_metrics,
            self.logs_runs,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_project_layout() -> ProjectLayout:
    root_dir = Path(__file__).resolve().parents[2]
    config_path = root_dir / "configs" / "architecture.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ProjectLayout(root_dir=root_dir, config=config)
