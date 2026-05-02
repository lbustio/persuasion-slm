import hashlib
import json
import os
import unicodedata
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.hardware import detect_hardware_profile
from src.utils.logger import setup_logger
from src.utils.paths import get_project_layout
from src.utils.state import StateManager
from src.pipeline.quality_gate import AugmentedDatasetQualityGate

PRINCIPLES = [
    "authority",
    "social_proof",
    "liking_similarity_deception",
    "commitment_integrity_reciprocation",
    "distraction",
]

PRINCIPLE_NAMES_ES = {
    "authority": "Autoridad",
    "social_proof": "Prueba social",
    "liking_similarity_deception": "Agrado/engano",
    "commitment_integrity_reciprocation": "Compromiso/reciprocidad",
    "distraction": "Distraccion/urgencia",
}

AUGMENTATION_SCHEMA_VERSION = "audit_v4"

SYSTEM_PROMPT = (
    "Eres un Analista Senior de Ciberseguridad especializado en persuasion y phishing. "
    "Tu trabajo no es obedecer ciegamente una hipotesis inicial, sino auditarla con rigor. "
    "Debes confirmar, matizar o descartar principios usando solo el mensaje. "
    "Tambien debes evaluar si el mensaje parece phishing, legitimo o ambiguo segun la evidencia disponible. "
    "Nunca inventes evidencia ni contexto externo. "
    "Si el texto no basta para sostener algo, debes decirlo con honestidad. "
    "Responde siempre en espanol neutro, con tono tecnico y claro. "
    "Usa solo ASCII: sin tildes, sin ene, sin comillas tipograficas y sin simbolos Unicode."
)

CHECKPOINT_EVERY = 10


class DataAugmenter:
    def __init__(self, run_id: str = None):
        self.logger = setup_logger("augmenter", run_id=run_id)
        self.layout = get_project_layout()
        self.config = self.layout.config
        self.state_mgr = StateManager(self.layout.outputs_checkpoints)
        self.hw_specs = detect_hardware_profile(self.logger)
        self.hardware_profile = self.hw_specs["profile"]
        self.quality_gate = AugmentedDatasetQualityGate(AUGMENTATION_SCHEMA_VERSION)
        self.teacher_max_new_tokens = (
            self.config.get("training", {}).get("teacher_max_new_tokens", 2048)
        )

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

    def _json_ascii_safe(self, payload: object) -> bool:
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return False
        return encoded.isascii()

    def execute(
        self,
        input_jsonl: str,
        teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        force_restart: bool = False,
        split_manifest_path: str | Path | None = None,
    ) -> str:
        task_name = "data_augmentation"
        out_path = self.layout.outputs_artifacts / "augmented_dataset.jsonl"
        partial_path = self.layout.cache_partials / "augmented_partial.jsonl"
        quality_report_path = self.layout.outputs_reports / "augmentation_quality_report.json"

        if not force_restart and self.state_mgr.is_completed(task_name):
            if self._is_current_augmented_dataset(out_path):
                self.logger.info(
                    "Reusable augmented training dataset already exists with schema %s. Reusing it.",
                    AUGMENTATION_SCHEMA_VERSION,
                )
                return str(out_path)
            self.logger.warning("Existing augmentation is missing or uses an old schema. Regenerating.")

        if force_restart:
            self.logger.info("Fresh augmentation requested. Existing final and partial artifacts will be rebuilt.")
            if partial_path.exists():
                partial_path.unlink()
            if out_path.exists():
                out_path.unlink()
                self.logger.info("Deleted stale augmented dataset: %s", out_path)

        if not Path(input_jsonl).exists():
            self.logger.error("Input file not found: %s", input_jsonl)
            return self._create_dummy_augmented(out_path)

        if self.hardware_profile == "CPU_ONLY":
            self.logger.warning("Data augmentation with Teacher requires GPU. Creating test dummy data.")
            return self._create_dummy_augmented(out_path)

        self.logger.info("Loading Teacher Model: %s on %s...", teacher_model_name, self.hardware_profile)
        os.environ["HF_HOME"] = str(self.layout.cache_downloads)

        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {"use_safetensors": True}
        if self.hw_specs["device"] == "cpu":
            compute_dtype = torch.bfloat16 if self.hw_specs["bf16_supported"] else torch.float32
        else:
            compute_dtype = torch.bfloat16 if self.hw_specs["bf16_supported"] else torch.float16

        if self.hw_specs["device"] == "cuda":
            # Always use 4-bit NF4 on any CUDA device.
            # Teacher is inference-only: quantisation loss is negligible, but
            # it allows selecting models 2-3x larger than the raw VRAM would
            # otherwise permit (e.g. Llama-3.1-70B on an A100 40 GB).
            self.logger.info(
                "Loading teacher in 4-bit NF4 (%s) — enables larger models on %s.",
                compute_dtype, self.hardware_profile,
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "38GiB", "cpu": "48GiB"}
        else:
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        model.eval()
        if torch.cuda.is_available():
            self.logger.info("Teacher ready. GPU memory used: %.2f GB", torch.cuda.memory_allocated() / 1024**3)
        else:
            self.logger.info("Teacher ready on CPU.")

        records = self._load_training_records(input_jsonl, split_manifest_path)

        partial_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = self._count_current_partial_records(partial_path)
        if already_done:
            self.logger.info(
                "Resuming augmentation from record %s/%s (valid partial checkpoint found).",
                already_done,
                len(records),
            )

        self.logger.info("Starting synthetic audit distillation: %s pending records.", len(records) - already_done)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if already_done > 0 and out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8") as out_handle, open(
            partial_path, "a", encoding="utf-8"
        ) as partial_handle:
            for offset, record in enumerate(
                tqdm(records[already_done:], desc="Generating audits", initial=already_done, total=len(records))
            ):
                augmented_record = self._build_augmented_record(
                    record=record,
                    fallback_id=str(already_done + offset),
                    model=model,
                    tokenizer=tokenizer,
                )

                line = json.dumps(augmented_record, ensure_ascii=False) + "\n"
                out_handle.write(line)
                out_handle.flush()
                partial_handle.write(line)
                partial_handle.flush()

                completed = already_done + offset + 1
                if completed % CHECKPOINT_EVERY == 0:
                    pct = completed / len(records) * 100
                    self.logger.info("Checkpoint [%s/%s (%.1f%%)] flushed to disk.", completed, len(records), pct)

        report = self.quality_gate.validate_file(out_path, quality_report_path)
        if not report.passed:
            raise ValueError(
                f"Augmented dataset failed quality gate: {report.failed_records}/{report.total_records} records failed. "
                f"See {quality_report_path}"
            )

        self.state_mgr.mark_completed(
            task_name,
            {
                "output_file": str(out_path),
                "quality_report": str(quality_report_path),
                "total_records": len(records),
                "schema_version": AUGMENTATION_SCHEMA_VERSION,
            },
        )
        self.logger.info("Reusable augmented dataset complete. %s records saved to: %s", len(records), out_path)
        return str(out_path)

    def _build_augmented_record(self, record: dict, fallback_id: str, model, tokenizer) -> dict:
        text = self._to_ascii(record.get("text", "")).strip()
        labels = record.get("labels", {})
        justifications = record.get("justifications", {})
        is_phishing = int(record.get("is_phishing", 0))
        record_id = self._to_ascii(record.get("id", fallback_id))
        classifier_hypothesis = self._build_classifier_hypothesis(labels, record_id)
        phishing_label_source = self._to_ascii(record.get("phishing_label_source", "unknown"))
        phishing_label_text = self._to_ascii(record.get("phishing_label_text", ""))
        user_prompt = self._build_training_user_prompt(
            text,
            classifier_hypothesis,
            is_phishing,
            phishing_label_source,
        )

        has_real_justifs = any(justifications.get(p, "").strip() for p in PRINCIPLES)
        if has_real_justifs:
            assistant_response = self._build_response_from_justifications(
                labels,
                justifications,
                classifier_hypothesis,
                is_phishing,
                phishing_label_source,
            )
            generation_source = "human_justifications_template"
        else:
            assistant_response = self._generate_with_teacher(
                model,
                tokenizer,
                text,
                classifier_hypothesis,
                is_phishing,
                phishing_label_source,
            )
            generation_source = "teacher_generated"

        quality = self._assess_record_quality(
            assistant_response=assistant_response,
            text=text,
            labels=labels,
            is_phishing=is_phishing,
            generation_source=generation_source,
        )

        return {
            "schema_version": AUGMENTATION_SCHEMA_VERSION,
            "source_id": record_id,
            "source": self._to_ascii(record.get("source", "")),
            "dataset_file": self._to_ascii(record.get("dataset_file", "")),
            "source_split": self._to_ascii(record.get("split", "unknown")),
            "teacher_model_name": self._to_ascii(getattr(model, "name_or_path", "")),
            "generation_source": generation_source,
            "phishing_label_source": phishing_label_source,
            "phishing_label_text": phishing_label_text,
            "is_phishing": is_phishing,
            "labels": {key: int(labels.get(key, 0)) for key in PRINCIPLES},
            "classifier_hypothesis": classifier_hypothesis,
            "justifications": {key: self._to_ascii(justifications.get(key, "")) for key in PRINCIPLES},
            "annotation_details": record.get("annotation_details", {}),
            "quality": quality,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
        }

    def _load_training_records(self, input_jsonl: str, split_manifest_path: str | Path | None) -> list[dict]:
        if split_manifest_path:
            manifest_path = Path(split_manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split_files = manifest.get("split_files", {})
            records: list[dict] = []
            for split_name in ("train", "validation"):
                split_path = Path(split_files.get(split_name, ""))
                if not split_path.exists():
                    raise FileNotFoundError(f"Master split file missing for SLM augmentation: {split_path}")
                with open(split_path, "r", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle if line.strip())
            self.logger.info(
                "Loaded %s records for SLM augmentation from master train+validation splits.",
                len(records),
            )
            return records

        with open(input_jsonl, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.logger.warning(
            "No master split manifest was provided. SLM augmentation will use the full dataset."
        )
        return records

    def _generate_with_teacher(
        self,
        model,
        tokenizer,
        text: str,
        classifier_hypothesis: dict,
        is_phishing: int,
        phishing_label_source: str,
    ) -> str:
        hypothesis_text = self._format_classifier_hypothesis(classifier_hypothesis)
        cyber_label = "phishing" if int(is_phishing) == 1 else "legitimo"
        label_basis = (
            "label supervisado del dataset"
            if phishing_label_source == "dataset_label"
            else "senal inferida o desconocida; debes tratarla con cautela"
        )
        instruction = (
            "Audita criticamente la hipotesis inicial del sistema. "
            "Algunos principios propuestos pueden ser correctos, otros debiles y otros incorrectos. "
            "Debes separar principios de persuasion del juicio de phishing. "
            "Un principio de persuasion no implica por si solo que el mensaje sea phishing. "
            "No inventes contexto ni intenciones que el texto no sostenga.\n\n"
            "Responde con este formato exacto:\n"
            "Conclusion: ...\n"
            "Juicio de ciberseguridad: phishing|legitimo|ambiguo | evidencia: \"...\" o \"label supervisado del dataset\" o \"No encuentro evidencia textual suficiente.\" | analisis: ...\n"
            "Principios evaluados:\n"
            "- <principio>: confirmado|matizado|descartado | evidencia: \"...\" o \"No encuentro evidencia textual suficiente.\" | analisis: ... | intensidad: 0-10\n"
            "Preguntas utiles:\n"
            "- ...\n"
            "- ...\n"
            "Limite: ...\n\n"
            f"LABEL DE CIBERSEGURIDAD: {cyber_label} ({label_basis})\n\n"
            f"HIPOTESIS INICIAL DEL SISTEMA:\n{hypothesis_text}\n\n"
            f"MENSAJE:\n{text[:1800]}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.teacher_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = output_ids[0][input_ids.shape[1] :]
        return self._to_ascii(tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def _build_response_from_justifications(
        self,
        labels: dict,
        justifications: dict,
        classifier_hypothesis: dict,
        is_phishing: int,
        phishing_label_source: str,
    ) -> str:
        evaluated_lines: list[str] = []
        for principle, meta in classifier_hypothesis.items():
            score = float(meta.get("score", 0.0))
            is_positive = int(labels.get(principle, 0)) == 1
            justif = str(justifications.get(principle, "")).strip()

            if not is_positive and score < 0.30:
                continue

            if is_positive:
                verdict = "confirmado" if score >= 0.55 else "matizado"
                evidence = self._to_ascii(justif) if justif else "No encuentro una cita literal fiable en las notas del anotador."
                analysis = (
                    self._to_ascii(justif)
                    if justif
                    else "La anotacion humana sugiere este principio, aunque la evidencia disponible en la nota es limitada."
                )
                intensity = 7 if score >= 0.55 else 5
            else:
                verdict = "descartado"
                evidence = "No encuentro evidencia textual suficiente."
                analysis = "La hipotesis inicial lo sugiere de forma debil, pero no veo base suficiente en el mensaje para sostenerlo."
                intensity = 0

            evaluated_lines.append(
                f"- {principle}: {verdict} | evidencia: {evidence} | analisis: {analysis} | intensidad: {intensity}"
            )

        if not evaluated_lines:
            evaluated_lines.append(
                "- evaluacion_global: descartado | evidencia: No encuentro evidencia textual suficiente. | analisis: El mensaje no ofrece senales claras de persuasion de ingenieria social. | intensidad: 0"
            )

        confirmed = [PRINCIPLE_NAMES_ES.get(p, p) for p in PRINCIPLES if int(labels.get(p, 0)) == 1]
        if confirmed:
            conclusion = (
                "La hipotesis inicial apunta a principios plausibles, pero la lectura final confirma sobre todo: "
                + ", ".join(confirmed)
                + "."
            )
        else:
            conclusion = "La hipotesis inicial no queda respaldada por evidencia suficiente en el mensaje."

        cyber_verdict = "phishing" if int(is_phishing) == 1 else "legitimo"
        cyber_evidence = (
            "label supervisado del dataset"
            if phishing_label_source == "dataset_label"
            else "No encuentro evidencia textual suficiente."
        )
        if int(is_phishing) == 1:
            cyber_analysis = (
                "El label supervisado marca el mensaje como phishing. La explicacion no debe convertir cada "
                "principio de persuasion en prueba de phishing; si faltan datos tecnicos, debe reconocer el limite."
            )
        else:
            cyber_analysis = (
                "El label supervisado marca el mensaje como legitimo. La respuesta debe evitar acusarlo de phishing "
                "solo porque aparezcan principios de persuasion."
            )

        return (
            f"Conclusion: {conclusion}\n"
            f"Juicio de ciberseguridad: {cyber_verdict} | evidencia: {cyber_evidence}. | analisis: {cyber_analysis}\n"
            "Principios evaluados:\n"
            f"{chr(10).join(evaluated_lines)}\n"
            "Preguntas utiles:\n"
            "- Que frase exacta del mensaje sostiene mas claramente esta lectura?\n"
            "- Que parte de la hipotesis inicial parece mas debil o discutible?\n"
            "Limite: Esta auditoria se basa en el texto del mensaje y en notas humanas resumidas, no en contexto externo."
        )

    def _build_classifier_hypothesis(self, labels: dict, record_id: str) -> dict:
        seed = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        positives = [p for p in PRINCIPLES if int(labels.get(p, 0)) == 1]
        negatives = [p for p in PRINCIPLES if int(labels.get(p, 0)) == 0]

        def score_for(key: str, low: float, high: float) -> float:
            chunk = hashlib.sha256(f"{record_id}:{key}".encode("utf-8")).hexdigest()[:8]
            ratio = int(chunk, 16) / 0xFFFFFFFF
            return round(low + (high - low) * ratio, 2)

        missed_positive = None
        if len(positives) >= 2 and int(seed[:2], 16) % 3 == 0:
            missed_positive = positives[int(seed[2:4], 16) % len(positives)]

        noisy_negative = negatives[int(seed[4:6], 16) % len(negatives)] if negatives else None
        hypothesis = {}
        for principle in PRINCIPLES:
            is_positive = int(labels.get(principle, 0)) == 1
            if is_positive:
                if principle == missed_positive:
                    score = score_for(principle, 0.18, 0.29)
                    status = "No detectado"
                else:
                    score = score_for(principle, 0.63, 0.89)
                    status = "Detectado"
            else:
                if principle == noisy_negative:
                    score = score_for(principle, 0.33, 0.49)
                    status = "Senal debil"
                else:
                    score = score_for(principle, 0.05, 0.22)
                    status = "No detectado"
            hypothesis[principle] = {"score": score, "status": status}
        return hypothesis

    def _format_classifier_hypothesis(self, hypothesis: dict) -> str:
        lines = []
        for principle in PRINCIPLES:
            meta = hypothesis.get(principle, {})
            lines.append(
                f"- {principle}: score={meta.get('score', 0.0)} | status={meta.get('status', 'No detectado')}"
            )
        return "\n".join(lines)

    def _build_training_user_prompt(
        self,
        text: str,
        classifier_hypothesis: dict,
        is_phishing: int,
        phishing_label_source: str,
    ) -> str:
        cyber_label = "phishing" if int(is_phishing) == 1 else "legitimo"
        label_basis = (
            "label supervisado del dataset"
            if phishing_label_source == "dataset_label"
            else "senal inferida o desconocida"
        )
        return (
            "Analiza este mensaje a partir de la hipotesis inicial del sistema. "
            "No la tomes como verdad final: auditala criticamente, confirma lo que tenga sustento y descarta lo que no. "
            "Separa siempre principios de persuasion y juicio de phishing.\n\n"
            f"LABEL DE CIBERSEGURIDAD: {cyber_label} ({label_basis})\n\n"
            f"HIPOTESIS INICIAL DEL SISTEMA:\n{self._format_classifier_hypothesis(classifier_hypothesis)}\n\n"
            f"MENSAJE:\n{text[:1800]}"
        )

    def _assess_record_quality(
        self,
        assistant_response: str,
        text: str,
        labels: dict,
        is_phishing: int,
        generation_source: str,
    ) -> dict:
        checks = {
            "ascii": assistant_response.isascii(),
            "has_conclusion": "Conclusion:" in assistant_response,
            "has_cyber_judgment": "Juicio de ciberseguridad:" in assistant_response,
            "has_principles_section": "Principios evaluados:" in assistant_response,
            "has_limit": "Limite:" in assistant_response,
            "valid_phishing_label": int(is_phishing) in {0, 1},
            "has_labels": all(key in labels for key in PRINCIPLES),
            "has_text": bool(text.strip()),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "generation_source": generation_source,
        }

    def _is_current_augmented_dataset(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size <= 1024:
            return False
        try:
            report = self.quality_gate.validate_file(path)
            return report.passed
        except Exception:
            return False

    def _count_current_partial_records(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            count = 0
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    if payload.get("schema_version") != AUGMENTATION_SCHEMA_VERSION:
                        return 0
                    if not self._json_ascii_safe(payload):
                        return 0
                    count += 1
            return count
        except Exception:
            return 0

    def _create_dummy_augmented(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_record = {
            "schema_version": AUGMENTATION_SCHEMA_VERSION,
            "source_id": "dummy",
            "source": "dummy",
            "source_split": "train",
            "dataset_file": "dummy",
            "teacher_model_name": "dummy",
            "generation_source": "dummy_fallback",
            "phishing_label_source": "dataset_label",
            "phishing_label_text": "1",
            "is_phishing": 1,
            "labels": {key: 0 for key in PRINCIPLES},
            "classifier_hypothesis": {},
            "justifications": {},
            "annotation_details": {},
            "quality": {"passed": True, "checks": {}, "generation_source": "dummy_fallback"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analiza este mensaje a partir de la hipotesis inicial del sistema.\n\n"
                        "HIPOTESIS INICIAL DEL SISTEMA:\n"
                        "- distraction: score=0.82 | status=Detectado\n"
                        "- authority: score=0.21 | status=No detectado\n\n"
                        "LABEL SUPERVISADO DE CIBERSEGURIDAD: phishing\n\n"
                        "MENSAJE:\nHaz clic ahora o perderas tu cuenta."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Conclusion: La hipotesis inicial queda bien respaldada en distraction por la presion temporal y la amenaza de perdida.\n"
                        "Juicio de ciberseguridad: phishing | evidencia: \"Haz clic ahora\" y \"perderas tu cuenta\" | analisis: El mensaje muestra presion para actuar y amenaza perdida de cuenta, senales compatibles con phishing.\n"
                        "Principios evaluados:\n"
                        "- distraction: confirmado | evidencia: \"Haz clic ahora\" y \"perderas tu cuenta\" | analisis: El mensaje empuja a actuar sin reflexion mediante urgencia y miedo a la perdida. | intensidad: 9\n"
                        "Preguntas utiles:\n"
                        "- Que efecto busca producir la amenaza de perdida inmediata?\n"
                        "- Hay alguna senal adicional de suplantacion o autoridad?\n"
                        "Limite: La conclusion se basa solo en el texto disponible."
                    ),
                },
            ],
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            for _ in range(100):
                handle.write(json.dumps(dummy_record, ensure_ascii=False) + "\n")

        self.state_mgr.mark_completed(
            "data_augmentation",
            {
                "output_file": str(out_path),
                "total_records": 100,
                "schema_version": AUGMENTATION_SCHEMA_VERSION,
            },
        )
        return str(out_path)
