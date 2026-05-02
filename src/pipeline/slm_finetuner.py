import os
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.reporting.paper_artifacts import PaperArtifactManager
from src.utils.logger import setup_logger
from src.utils.state import StateManager
from src.utils.hardware import detect_hardware_profile
from src.utils.paths import get_project_layout
from src.utils.turbo import apply_turbo_runtime, build_turbo_settings
from src.visualization.plots import PlotManager
from src.pipeline.augmenter import AUGMENTATION_SCHEMA_VERSION
from src.pipeline.quality_gate import AugmentedDatasetQualityGate

class SLMFineTuner:
    def __init__(self, run_id: str = None):
        self.logger = setup_logger("slm_finetuner", run_id=run_id)

        self.layout = get_project_layout()
        self.config = self.layout.config
        with open(self.layout.root_dir / "configs" / "training_defaults.yaml", "r", encoding="utf-8") as f:
            self.training_defaults = yaml.safe_load(f)
        self.state_mgr = StateManager(self.layout.outputs_checkpoints)
        self.plot_mgr = PlotManager(self.layout.outputs_figures)
        self.artifact_mgr = PaperArtifactManager(self.logger, self.plot_mgr)
        self.hw_specs = detect_hardware_profile(self.logger)
        self.hardware_profile = self.hw_specs["profile"]
        self.quality_gate = AugmentedDatasetQualityGate(AUGMENTATION_SCHEMA_VERSION)
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

    def _tokenize_supervised_chat(self, messages: list[dict], tokenizer, max_length: int) -> dict[str, list[int]]:
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_text = messages[-1]["content"].strip()
        if tokenizer.eos_token:
            assistant_text = assistant_text + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]

        answer_cap = min(max(256, int(max_length * 0.65)), max_length)
        answer_slice = answer_ids[:answer_cap]
        remaining = max_length - len(answer_slice)
        if len(prompt_ids) <= remaining:
            prompt_slice = prompt_ids
        else:
            # Keep the system prompt and the tail of the user message, where the analyzed text ends.
            keep_first = 150
            keep_last = remaining - keep_first
            prompt_slice = prompt_ids[:keep_first] + prompt_ids[-keep_last:] if keep_last > 0 else prompt_ids[:remaining]

        input_ids = prompt_slice + answer_slice
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_slice)) + answer_slice.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
    def train(
        self,
        model_name: str,
        dataset_path: str,
        force_restart: bool = False,
        run_tag: str | None = None,
        split_manifest_path: str | Path | None = None,
    ):
        model_key = model_name.replace('/', '_')
        artifact_key = f"{model_key}_{run_tag}" if run_tag else model_key
        task_name = f"train_slm_{artifact_key}"
        out_dir = self.layout.outputs_models / artifact_key
        checkpoint_dir = self.layout.outputs_checkpoints / artifact_key
        
        if not force_restart and self.state_mgr.is_completed(task_name):
            state = self.state_mgr.load_state(task_name) or {}
            saved_model_dir = state.get("result", {}).get("model_dir")

            if saved_model_dir == model_name and self.hardware_profile != "CPU_ONLY":
                self.logger.warning(
                    f"Found a CPU fallback checkpoint for {model_name}. GPU is available now, so fine-tuning will run."
                )
            elif saved_model_dir and saved_model_dir != model_name and self.layout.resolve(saved_model_dir).exists():
                self.logger.info(f"SLM Training for {model_name} already completed.")
                return saved_model_dir
            elif saved_model_dir == model_name:
                self.logger.info(f"SLM Training for {model_name} already completed with base-model fallback.")
                return saved_model_dir
            else:
                self.logger.warning(
                    f"Checkpoint for {model_name} exists but the saved model directory is missing. Re-running fine-tuning."
                )
        elif force_restart:
            self.logger.info(
                f"Fresh SLM run requested for {model_name}. A new adapter will be written to {out_dir}."
            )
            
        if self.hardware_profile == "CPU_ONLY":
            self.logger.warning("SLM Training requires a GPU for QLoRA. Skipping fine-tuning and using base model for downstream tasks.")
            self.state_mgr.mark_completed(task_name, {"model_dir": model_name})
            return model_name
            
        hw_cfg = self.config["hardware"]["profiles"].get(self.hardware_profile, {})
        training_cfg = self.training_defaults["training"]
        slm_max_length = int(training_cfg.get("slm_max_length", max(512, int(training_cfg.get("max_length", 256)))))
        turbo = build_turbo_settings(
            self.hw_specs,
            workload="slm",
            base_cfg=hw_cfg,
            max_length=slm_max_length,
        )
        self.logger.info(
            "Turbo SLM: max_length=%s batch=%s eval_batch=%s grad_acc=%s workers=%s pin_memory=%s bf16=%s fp16=%s tf32=%s qlora=%s grad_ckpt=%s",
            turbo.max_length,
            turbo.train_batch_size,
            turbo.eval_batch_size,
            turbo.gradient_accumulation_steps,
            turbo.dataloader_num_workers,
            turbo.dataloader_pin_memory,
            turbo.use_bf16,
            turbo.use_fp16,
            turbo.use_tf32,
            turbo.qlora_4bit,
            turbo.gradient_checkpointing,
        )
        
        os.environ["HF_HOME"] = str(self.layout.cache_downloads)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {"device_map": "auto"}
        
        # Seleccion dinamica de precision
        use_bf16 = turbo.use_bf16
        if self.hw_specs["device"] == "cpu":
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
        else:
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        if turbo.qlora_4bit and self.hw_specs["device"] == "cuda":
            self.logger.info(f"Enabling QLoRA 4-bit quantization with {compute_dtype}.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )
        else:
            model_kwargs["torch_dtype"] = compute_dtype
            
        model_kwargs["use_safetensors"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        if turbo.qlora_4bit:
            model = prepare_model_for_kbit_training(model)
        if turbo.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        quality_report_path = self.layout.outputs_reports / "slm_input_quality_report.json"
        quality_report = self.quality_gate.validate_file(dataset_path, quality_report_path)
        if not quality_report.passed:
            raise ValueError(
                f"SLM input dataset failed quality gate: {quality_report.failed_records}/"
                f"{quality_report.total_records} records failed. See {quality_report_path}"
            )

        # Load real augmented data
        self.logger.info(f"Loading augmented training dataset: {dataset_path}")
        ds = Dataset.from_json(dataset_path)
        
        def tokenize(batch):
            encoded_rows = [self._tokenize_supervised_chat(msg, tokenizer, turbo.max_length) for msg in batch["messages"]]
            return {
                "input_ids": [row["input_ids"] for row in encoded_rows],
                "attention_mask": [row["attention_mask"] for row in encoded_rows],
                "labels": [row["labels"] for row in encoded_rows],
            }
            
        split_ds, split_ratio = self._build_slm_split(ds, split_manifest_path)
        raw_train_records = [dict(row) for row in split_ds["train"]]
        raw_eval_records = [dict(row) for row in split_ds["eval"]]
        train_ds = split_ds["train"].map(tokenize, batched=True, remove_columns=split_ds["train"].column_names)
        eval_ds = split_ds["eval"].map(tokenize, batched=True, remove_columns=split_ds["eval"].column_names)
        
        args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=turbo.train_batch_size,
            per_device_eval_batch_size=turbo.eval_batch_size,
            gradient_accumulation_steps=turbo.gradient_accumulation_steps,
            num_train_epochs=float(training_cfg.get("max_epochs", 3)),
            learning_rate=float(training_cfg.get("learning_rate", 3.0e-5)),
            fp16=turbo.use_fp16,
            bf16=use_bf16,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_dir=str(self.layout.logs_runs),
            logging_steps=10,
            report_to="none",
            optim="paged_adamw_8bit" if turbo.qlora_4bit else "adamw_torch",
            dataloader_num_workers=turbo.dataloader_num_workers,
            dataloader_pin_memory=turbo.dataloader_pin_memory,
            group_by_length=turbo.group_by_length,
            gradient_checkpointing=turbo.gradient_checkpointing,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
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
                last_checkpoint = str(max(checkpoints, key=os.path.getctime))
                self.logger.info(f"Resuming SLM from checkpoint: {last_checkpoint}")
        
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Save PEFT adapter
            trainer.save_model(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            
            # Save metrics
            log_history = trainer.state.log_history
            train_x, train_loss, val_x, eval_loss = self._extract_learning_curve_series(log_history)
            epochs = train_x if train_x else list(range(1, len(train_loss) + 1))
            
            artifact_manifest_path = None
            try:
                if len(train_loss) > 0:
                    self.plot_mgr.plot_learning_curve(
                        epochs,
                        train_loss,
                        eval_loss,
                        name=f"slm_lc_{model_name.replace('/', '_')}",
                        train_x=train_x,
                        val_x=val_x,
                    )

                artifact_manifest = self.artifact_mgr.build_slm_artifacts(
                    model_name=model_name,
                    raw_train_records=raw_train_records,
                    raw_eval_records=raw_eval_records,
                    log_history=log_history,
                    split_ratio=split_ratio,
                )
                artifact_manifest_path = artifact_manifest["manifest_path"]
            except Exception as reporting_error:
                self.logger.warning(
                    "El fine-tuning termino, pero la fase de reporting/exportacion secundaria fallo: %s",
                    reporting_error,
                    exc_info=True,
                )
            
            self.state_mgr.mark_completed(task_name, {"model_dir": str(out_dir), "paper_artifacts_manifest": artifact_manifest_path})
            self.logger.info("SLM Training completed successfully.")
            
            return str(out_dir)
            
        except Exception as e:
            self.logger.error(f"SLM Training failed: {e}", exc_info=True)
            raise

    def _build_slm_split(self, dataset: Dataset, split_manifest_path: str | Path | None):
        source_splits = dataset.unique("source_split") if "source_split" in dataset.column_names else []
        if split_manifest_path and "train" in source_splits and "validation" in source_splits:
            train_ds = dataset.filter(lambda row: row.get("source_split") == "train")
            eval_ds = dataset.filter(lambda row: row.get("source_split") == "validation")
            if len(train_ds) > 0 and len(eval_ds) > 0:
                total = len(train_ds) + len(eval_ds)
                self.logger.info(
                    "Using master train/validation split for SLM: train=%s eval=%s.",
                    len(train_ds),
                    len(eval_ds),
                )
                return {"train": train_ds, "eval": eval_ds}, {
                    "train": len(train_ds) / total,
                    "eval": len(eval_ds) / total,
                    "source": "master_split",
                }
            self.logger.warning("Master split metadata exists but SLM train/eval records are incomplete. Falling back.")

        split_ds = dataset.train_test_split(test_size=0.1, seed=42)
        total = len(split_ds["train"]) + len(split_ds["test"])
        return {"train": split_ds["train"], "eval": split_ds["test"]}, {
            "train": len(split_ds["train"]) / max(total, 1),
            "eval": len(split_ds["test"]) / max(total, 1),
            "source": "random_fallback",
        }
