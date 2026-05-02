import json
import os
from pathlib import Path
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset

from src.utils.logger import setup_logger
from src.utils.state import StateManager
from src.utils.hardware import detect_hardware_profile
from src.utils.paths import get_project_layout
from src.utils.turbo import apply_turbo_runtime, build_turbo_settings
from src.reporting.paper_artifacts import PaperArtifactManager
from src.visualization.plots import PlotManager

class ClassifierTrainer:
    def __init__(self, run_id: str = None):
        self.logger = setup_logger("classifier_trainer", run_id=run_id)

        self.layout = get_project_layout()
        self.config = self.layout.config
        with open(self.layout.root_dir / "configs" / "training_defaults.yaml", "r", encoding="utf-8") as f:
            self.training_defaults = yaml.safe_load(f)
        self.state_mgr = StateManager(self.layout.outputs_checkpoints)
        self.plot_mgr = PlotManager(self.layout.outputs_figures)
        self.artifact_mgr = PaperArtifactManager(self.logger, self.plot_mgr)
        self.hw_specs = detect_hardware_profile(self.logger)
        self.hardware_profile = self.hw_specs["profile"]
        apply_turbo_runtime(self.hw_specs, self.logger)

    def _extract_learning_curve_series(self, log_history: list[dict]) -> tuple[list[float], list[float], list[float], list[float]]:
        train_x = []
        train_loss = []
        val_x = []
        val_loss = []

        for entry in log_history:
            if "loss" in entry:
                train_loss.append(float(entry["loss"]))
                train_x.append(float(entry.get("epoch", len(train_loss))))
            if "eval_loss" in entry:
                val_loss.append(float(entry["eval_loss"]))
                val_x.append(float(entry.get("epoch", len(val_loss))))

        return train_x, train_loss, val_x, val_loss
        
    def _checkpoint_num_labels(self, checkpoint_path: str | Path) -> int | None:
        config_path = Path(checkpoint_path) / "config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return None

        if "num_labels" in cfg:
            return int(cfg["num_labels"])
        if "id2label" in cfg:
            return len(cfg["id2label"])
        if "label2id" in cfg:
            return len(cfg["label2id"])
        return None

    def train(
        self,
        model_name: str,
        dataset_path: str,
        force_restart: bool = False,
        split_manifest_path: str | Path | None = None,
    ):
        task_name = f"train_classifier_{model_name.replace('/', '_')}"
        out_dir = self.layout.outputs_models / model_name.replace('/', '_')
        checkpoint_dir = self.layout.outputs_checkpoints / model_name.replace('/', '_')
        data_cfg = self.training_defaults["data"]
        train_ratio = float(data_cfg["train_ratio"])
        validation_ratio = float(data_cfg["validation_ratio"])
        test_ratio = float(data_cfg["test_ratio"])
        seed = int(self.training_defaults.get("seed", 42))
        
        if not force_restart and self.state_mgr.is_completed(task_name):
            state = self.state_mgr.load_state(task_name) or {}
            saved_model_dir = state.get("result", {}).get("model_dir")
            saved_split_ratios = state.get("result", {}).get("split_ratios")
            expected_ratios = self._expected_split_ratios(split_manifest_path, train_ratio, validation_ratio, test_ratio)
            ratios_match = saved_split_ratios == expected_ratios
            if saved_model_dir and self.layout.resolve(saved_model_dir).exists() and ratios_match:
                self.logger.info(f"Training for {model_name} already completed.")
                return saved_model_dir
            if not ratios_match:
                self.logger.warning(
                    f"Checkpoint for {model_name} uses a different split configuration. Retraining with the current ratios."
                )
            else:
                self.logger.warning(
                    f"Checkpoint for {model_name} exists but the saved model directory is missing. Retraining."
                )
        elif force_restart:
            self.logger.info(f"Fresh classifier run requested for {model_name}. Existing checkpoints will be ignored.")
            
        self.logger.info(f"Starting training for {model_name} on {self.hardware_profile}")
        
        hw_cfg = self.config["hardware"]["profiles"].get(self.hardware_profile, {})
        training_cfg = self.training_defaults["training"]
        
        # Download caching is handled by transformers automatically if env vars are set,
        # but we can force it here:
        os.environ["HF_HOME"] = str(self.layout.cache_downloads)
        
        turbo = build_turbo_settings(
            self.hw_specs,
            workload="classifier",
            base_cfg=hw_cfg,
            max_length=int(training_cfg.get("max_length", 256)),
        )
        self.logger.info(
            "Turbo classifier: batch=%s eval_batch=%s grad_acc=%s workers=%s pin_memory=%s bf16=%s fp16=%s tf32=%s",
            turbo.train_batch_size,
            turbo.eval_batch_size,
            turbo.gradient_accumulation_steps,
            turbo.dataloader_num_workers,
            turbo.dataloader_pin_memory,
            turbo.use_bf16,
            turbo.use_fp16,
            turbo.use_tf32,
        )
        
        save_strategy = training_cfg.get("save_strategy", "steps")
        eval_strategy = training_cfg.get("eval_strategy", "steps")
        save_steps = int(training_cfg.get("save_steps", 200))
        eval_steps = int(training_cfg.get("eval_steps", 200))
        save_total_limit = int(training_cfg.get("save_total_limit", 3))
        logging_steps = int(training_cfg.get("logging_steps", 25))

        args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=turbo.train_batch_size,
            per_device_eval_batch_size=turbo.eval_batch_size,
            gradient_accumulation_steps=turbo.gradient_accumulation_steps,
            num_train_epochs=float(training_cfg.get("max_epochs", 3)),
            learning_rate=float(training_cfg.get("learning_rate", 3.0e-5)),
            fp16=turbo.use_fp16,
            bf16=turbo.use_bf16,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            logging_dir=str(self.layout.logs_runs),
            logging_steps=logging_steps,
            report_to="none",
            dataloader_num_workers=turbo.dataloader_num_workers,
            dataloader_pin_memory=turbo.dataloader_pin_memory,
            group_by_length=turbo.group_by_length,
        )
        
        self.logger.info(f"Cargando dataset armonizado para entrenamiento: {dataset_path}")
        train_ds, eval_ds, test_ds, split_ratios = self._load_classifier_splits(
            dataset_path=dataset_path,
            split_manifest_path=split_manifest_path,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        raw_train_records = [dict(row) for row in train_ds]
        raw_eval_records = [dict(row) for row in eval_ds]
        raw_test_records = [dict(row) for row in test_ds]

        self.logger.info(
            "Split aplicado: train=%s, validation=%s, test=%s",
            len(train_ds),
            len(eval_ds),
            len(test_ds),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Labels order according to harmonizer.py
        label_keys = [
            "authority", "social_proof", "liking_similarity_deception",
            "commitment_integrity_reciprocation", "distraction", "is_phishing"
        ]

        def tokenize(batch):
            tokens = tokenizer(batch["text"], truncation=True, max_length=turbo.max_length)
            labels_batch = []
            raw_labels = batch["labels"]
            # HF Datasets may return Struct columns as list-of-dicts OR dict-of-lists
            # depending on the version and Arrow schema inference.
            is_col_oriented = isinstance(raw_labels, dict)
            for i in range(len(batch["text"])):
                if is_col_oriented:
                    labels_dict = {k: raw_labels[k][i] for k in raw_labels}
                else:
                    labels_dict = raw_labels[i]
                is_phishing = batch.get("is_phishing", [0] * len(batch["text"]))[i]
                vector = [float(labels_dict.get(k, 0)) for k in label_keys[:-1]]
                vector.append(float(is_phishing))
                labels_batch.append(vector)
            tokens["labels"] = labels_batch
            return tokens
            
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)
        test_ds = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)
        
        # num_labels=6 for the 5 persuasion principles + 1 binary phishing verdict
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=6, 
            problem_type="multi_label_classification",
            use_safetensors=True
        )
        
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=turbo.pad_to_multiple_of,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )
        
        self.logger.info("Checking for existing checkpoints...")
        last_checkpoint = None
        if not force_restart and checkpoint_dir.exists():
            checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
            if checkpoints:
                candidate_checkpoint = max(checkpoints, key=os.path.getctime)
                checkpoint_num_labels = self._checkpoint_num_labels(candidate_checkpoint)
                if checkpoint_num_labels not in (None, 6):
                    self.logger.warning(
                        f"Ignorando checkpoint incompatible {candidate_checkpoint} porque fue entrenado con {checkpoint_num_labels} etiquetas."
                    )
                else:
                    last_checkpoint = str(candidate_checkpoint)
                    self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Save final model
            trainer.save_model(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            
            # Extract logs for plotting
            log_history = trainer.state.log_history
            train_x, train_loss, val_x, val_loss = self._extract_learning_curve_series(log_history)
            epochs = train_x if train_x else list(range(1, len(train_loss) + 1))
            
            artifact_manifest_path = None
            try:
                if len(train_loss) > 0:
                    self.plot_mgr.plot_learning_curve(
                        epochs,
                        train_loss,
                        val_loss,
                        name=f"lc_{model_name.replace('/', '_')}",
                        train_x=train_x,
                        val_x=val_x,
                    )

                evaluation_threshold = float(training_cfg.get("eval_threshold", 0.5))
                eval_predictions = trainer.predict(eval_ds)
                test_predictions = trainer.predict(test_ds)
                artifact_manifest = self.artifact_mgr.build_classifier_artifacts(
                    model_name=model_name,
                    label_keys=label_keys,
                    label_display_names=[
                        "Authority",
                        "Social Proof",
                        "Liking, Similarity, and Deception",
                        "Commitment, Integrity, and Reciprocation",
                        "Distraction",
                        "Is Phishing (Binary Verdict)",
                    ],
                    raw_train_records=raw_train_records,
                    raw_validation_records=raw_eval_records,
                    raw_test_records=raw_test_records,
                    eval_predictions=eval_predictions,
                    test_predictions=test_predictions,
                    log_history=log_history,
                    threshold=evaluation_threshold,
                    split_ratios=split_ratios,
                    seed=seed,
                )
                artifact_manifest_path = artifact_manifest["manifest_path"]
            except Exception as reporting_error:
                self.logger.warning(
                    "El entrenamiento termino, pero la fase de reporting/exportacion secundaria fallo: %s",
                    reporting_error,
                    exc_info=True,
                )
            
            self.state_mgr.mark_completed(task_name, {
                "model_dir": str(out_dir),
                "split_counts": {
                    "train": len(train_ds),
                    "validation": len(eval_ds),
                    "test": len(test_ds),
                },
                "split_ratios": split_ratios,
                "paper_artifacts_manifest": artifact_manifest_path,
            })
            self.logger.info("Training completed successfully.")
            
            return str(out_dir)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def _expected_split_ratios(
        self,
        split_manifest_path: str | Path | None,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> dict[str, float]:
        if split_manifest_path and Path(split_manifest_path).exists():
            manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
            ratios = manifest.get("ratios", {})
            return {
                "train": float(ratios.get("train", 0.0)),
                "validation": float(ratios.get("validation", 0.0)),
                "test": float(ratios.get("test", 0.0)),
                "heldout_final": float(ratios.get("heldout_final", 0.0)),
            }
        return {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}

    def _load_classifier_splits(
        self,
        *,
        dataset_path: str,
        split_manifest_path: str | Path | None,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        seed: int,
    ):
        if split_manifest_path and Path(split_manifest_path).exists():
            manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
            split_files = manifest.get("split_files", {})
            train_ds = Dataset.from_json(split_files["train"])
            eval_ds = Dataset.from_json(split_files["validation"])
            test_ds = Dataset.from_json(split_files["test"])
            split_ratios = {
                "train": float(manifest["ratios"].get("train", 0.0)),
                "validation": float(manifest["ratios"].get("validation", 0.0)),
                "test": float(manifest["ratios"].get("test", 0.0)),
                "heldout_final": float(manifest["ratios"].get("heldout_final", 0.0)),
            }
            self.logger.info("Using master split manifest: %s", split_manifest_path)
            return train_ds, eval_ds, test_ds, split_ratios

        ds = Dataset.from_json(dataset_path)
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(f"Las proporciones de split deben sumar 1.0 y hoy suman {total_ratio}.")
        split_ds = ds.train_test_split(test_size=(1.0 - train_ratio), seed=seed)
        train_ds = split_ds["train"]
        remaining_ds = split_ds["test"]
        validation_share = validation_ratio / (validation_ratio + test_ratio)
        remaining_split = remaining_ds.train_test_split(test_size=(1.0 - validation_share), seed=seed)
        return train_ds, remaining_split["train"], remaining_split["test"], {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        }
