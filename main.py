import argparse
import importlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.paths import get_project_layout


def _model_key(model_name: str, run_tag: str | None = None) -> str:
    base = model_name.replace("/", "_")
    return f"{base}_{run_tag}" if run_tag else base


def bootstrap_runtime():
    layout = get_project_layout()
    layout.ensure_runtime_dirs()
    os.environ["HF_HOME"] = str(layout.cache_downloads)
    os.environ["TORCH_HOME"] = str(layout.cache_downloads)
    
    secrets_path = layout.root_dir / "secrets" / "secrets.txt"
    if secrets_path.exists():
        token = secrets_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            
    return layout


def preload_modules():
    try:
        from tqdm import tqdm

        modules_to_preload = [
            ("numpy", "Loading NumPy"),
            ("pandas", "Loading Pandas"),
            ("scipy", "Loading SciPy"),
            ("sklearn", "Loading Scikit-Learn"),
            ("torch", "Loading PyTorch"),
            ("transformers", "Loading Transformers"),
            ("peft", "Loading PEFT"),
            ("datasets", "Loading Datasets"),
        ]

        print()
        for mod_name, _ in tqdm(
            modules_to_preload,
            desc="Starting engines",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ):
            importlib.import_module(mod_name)
        print()
    except ImportError:
        pass


def build_parser():
    parser = argparse.ArgumentParser(description="Persuasion Principle Detection Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing every CSV dataset to ingest.")
    parser.add_argument("--iwspa", type=str, default=None, help="Deprecated manual IWSPA path override.")
    parser.add_argument("--spaphish", type=str, default=None, help="Deprecated manual Spaphish path override.")
    parser.add_argument("--classifier", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--slm", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="SLM model id, local path, or 'auto' for hardware-based online selection.")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Teacher LLM for augmentation. Model id, local path, or 'auto' for hardware-based selection.")
    parser.add_argument("--fresh-all", action="store_true", help="Rebuild every phase without reusing previous checkpoints.")
    parser.add_argument("--fresh-harmonizer", action="store_true", help="Regenerate harmonized data.")
    parser.add_argument("--fresh-augmenter", action="store_true", help="Regenerate augmented data.")
    parser.add_argument("--fresh-classifier", action="store_true", help="Retrain the classifier.")
    parser.add_argument("--fresh-slm", action="store_true", help="Run a fresh SLM fine-tuning job.")
    parser.add_argument("--slm-run-tag", type=str, default=None, help="Optional suffix for the SLM adapter directory.")
    parser.add_argument("--slm-only", action="store_true", help="Run only the phases needed to create the SLM.")
    return parser


def print_summary(run_id, hw_specs, layout, logger, slm_model_path: str | None):
    summary_lines = [
        "\n" + "=" * 60,
        f" EXECUTION SUMMARY - {run_id}",
        "=" * 60,
        f" HARDWARE: {hw_specs['name']} ({hw_specs['device'].upper()})",
        f" BF16:     {'SUPPORTED' if hw_specs['bf16_supported'] else 'NOT SUPPORTED'}",
        "-" * 60,
        " COMPLETED PHASES:",
        " [X] Phase 1: Data harmonization",
        " [X] Phase 2: Synthetic audit augmentation",
        " [X] Phase 3: Classifier training or reuse",
        " [X] Phase 4: SLM fine-tuning or reuse",
        " [X] Phase 5: ONNX / OpenVINO export or skip",
        "-" * 60,
        " FINAL ARTIFACTS:",
        f"  - Dataset: {layout.rel(layout.outputs_artifacts / 'augmented_dataset.jsonl')}",
        f"  - SLM:     {layout.rel(slm_model_path) if slm_model_path else 'not available'}",
        "=" * 60 + "\n",
    ]

    for line in summary_lines:
        print(line)
        logger.info(line.strip())

    try:
        with open(ROOT_DIR / "context" / "bitacora.md", "a", encoding="utf-8") as handle:
            handle.write(f"\n### Corrida: {run_id}\n")
            handle.write(f"- **HW:** {hw_specs['name']} | **BF16:** {hw_specs['bf16_supported']}\n")
            handle.write("- **Estado:** EXITO\n")
            handle.write(f"- **SLM:** {slm_model_path or 'no disponible'}\n")
    except Exception as exc:
        logger.warning("No se pudo actualizar la bitacora: %s", exc)


def archive_non_target_adapters(layout, target_key: str, logger):
    active_models_dir = layout.outputs_models
    if not active_models_dir.exists():
        return

    for item in active_models_dir.iterdir():
        if not item.is_dir() or not (item / "adapter_config.json").exists():
            continue
        if item.name == target_key:
            continue

        backup_target = layout.outputs_tuned_models / item.name
        layout.outputs_tuned_models.mkdir(parents=True, exist_ok=True)
        if not backup_target.exists():
            logger.info("Archiving adapter '%s' to outputs/tuned_models.", item.name)
            shutil.move(str(item), str(backup_target))
        else:
            logger.info("Adapter '%s' already exists in outputs/tuned_models. Removing duplicate active copy.", item.name)
            shutil.rmtree(item)


def archive_previous_run_outputs(layout, run_id: str, logger, enabled: bool):
    if not enabled:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = layout.outputs_tuned_models / "runs" / f"run_{timestamp}_{run_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for source_dir, backup_name in (
        (layout.outputs_results, "results"),
        (layout.outputs_checkpoints, "checkpoints"),
        (layout.outputs_splits, "splits"),
        (layout.outputs_artifacts, "artifacts"),
    ):
        if not source_dir.exists():
            continue
        has_content = any(source_dir.iterdir())
        if not has_content:
            continue

        destination = backup_dir / backup_name
        logger.info("Archiving previous %s to outputs/tuned_models backup: %s", backup_name, destination)
        shutil.move(str(source_dir), str(destination))
        source_dir.mkdir(parents=True, exist_ok=True)

    layout.ensure_runtime_dirs()


def main():
    layout = bootstrap_runtime()

    from src.utils.compat import patch_accelerate_unwrap_model
    from src.utils.hardware import detect_hardware_profile
    from src.utils.logger import get_run_id, setup_logger

    run_id = get_run_id()
    logger = setup_logger("orchestrator", run_id=run_id)
    logger.info("Starting execution. Run ID: %s", run_id)
    patch_accelerate_unwrap_model()
    preload_modules()

    hw_specs = detect_hardware_profile(logger)

    from src.pipeline.augmenter import DataAugmenter
    from src.pipeline.classifier_trainer import ClassifierTrainer
    from src.pipeline.exporter import ModelExporter
    from src.pipeline.harmonizer import DataHarmonizer
    from src.pipeline.split_manager import MasterSplitManager
    from src.pipeline.slm_finetuner import SLMFineTuner
    from src.utils.model_advisor import ModelAdvisor

    parser = build_parser()
    args = parser.parse_args()
    selected_slm = args.slm
    advisor = None
    if selected_slm.lower() == "auto":
        advisor = ModelAdvisor(logger, cache_dir=layout.outputs_reports, model_cache_dir=layout.cache_downloads)
        selected_slm = advisor.discover_optimal_model(hw_specs)

    selected_teacher = args.teacher
    if selected_teacher.lower() == "auto":
        if advisor is None:
            advisor = ModelAdvisor(logger, cache_dir=layout.outputs_reports, model_cache_dir=layout.cache_downloads)
        selected_teacher = advisor.discover_teacher_model(hw_specs)

    try:
        fresh_harmonizer = args.fresh_all or args.fresh_harmonizer
        fresh_augmenter = args.fresh_all or args.fresh_augmenter
        fresh_classifier = args.fresh_all or args.fresh_classifier
        fresh_slm = args.fresh_all or args.fresh_slm
        slm_run_tag = args.slm_run_tag or (run_id if fresh_slm else None)
        target_key = _model_key(selected_slm, slm_run_tag)

        archive_previous_run_outputs(layout, run_id, logger, enabled=args.fresh_all)
        archive_non_target_adapters(layout, target_key, logger)

        harmonizer = DataHarmonizer(run_id=run_id)
        dataset_path = harmonizer.execute(
            data_dir=args.data_dir,
            force_restart=fresh_harmonizer,
            manual_paths=[path for path in (args.iwspa, args.spaphish) if path],
        )

        split_manager = MasterSplitManager(run_id=run_id)
        split_manifest_path = split_manager.execute(dataset_path, force_restart=fresh_harmonizer)

        augmenter = DataAugmenter(run_id=run_id)
        augmented_dataset_path = augmenter.execute(
            dataset_path,
            selected_teacher,
            force_restart=fresh_augmenter,
            split_manifest_path=split_manifest_path,
        )

        if not args.slm_only:
            classifier_trainer = ClassifierTrainer(run_id=run_id)
            classifier_trainer.train(
                args.classifier,
                str(dataset_path),
                force_restart=fresh_classifier,
                split_manifest_path=split_manifest_path,
            )

        slm_trainer = SLMFineTuner(run_id=run_id)
        active_path = layout.outputs_models / target_key
        if active_path.exists() and not fresh_slm:
            logger.info("Using active SLM adapter: %s", active_path)
            slm_model_path = str(active_path)
        else:
            logger.info("Training SLM adapter target: %s", target_key)
            slm_model_path = slm_trainer.train(
                selected_slm,
                str(augmented_dataset_path),
                force_restart=fresh_slm,
                run_tag=slm_run_tag,
                split_manifest_path=split_manifest_path,
            )

        if not args.slm_only:
            exporter = ModelExporter(run_id=run_id)
            exporter.export_classifier_onnx(args.classifier)
            exporter.export_classifier_openvino(args.classifier)

        logger.info("Pipeline execution completed successfully.")
        print_summary(run_id, hw_specs, layout, logger, slm_model_path)
    except Exception as exc:
        logger.critical("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
