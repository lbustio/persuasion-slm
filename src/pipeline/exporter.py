import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import setup_logger
from src.utils.paths import get_project_layout

class ModelExporter:
    def __init__(self, run_id: str = None):
        self.logger = setup_logger("exporter", run_id=run_id)
        self.layout = get_project_layout()
        self.config = self.layout.config
            
    def export_classifier_onnx(self, model_name: str):
        self.logger.info(f"Starting ONNX export for {model_name}")
        
        in_dir = self.layout.outputs_models / model_name.replace('/', '_')
        if not in_dir.exists():
            self.logger.error(f"Model not found at {in_dir}")
            return
            
        out_path = self.layout.outputs_models / f"{model_name.replace('/', '_')}_onnx" / "model.onnx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(str(in_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(in_dir))
        model.eval()

        dummy_text = "Dummy text for tracing"
        inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"}
        }
        
        if "token_type_ids" in inputs:
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

        torch.onnx.export(
            model,
            tuple(inputs.values()),
            str(out_path),
            input_names=list(inputs.keys()),
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=17,
        )
        self.logger.info(f"ONNX export successful: {out_path}")
        
    def export_classifier_openvino(self, model_name: str):
        self.logger.info(f"Starting OpenVINO export for {model_name}")
        
        onnx_path = self.layout.outputs_models / f"{model_name.replace('/', '_')}_onnx" / "model.onnx"
        if not onnx_path.exists():
            self.logger.error("ONNX model required for OpenVINO export")
            return
            
        out_dir = self.layout.outputs_models / f"{model_name.replace('/', '_')}_openvino"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import openvino as ov
            core = ov.Core()
            ov_model = core.read_model(model=str(onnx_path))
            ov.save_model(ov_model, str(out_dir / "model.xml"))
            self.logger.info(f"OpenVINO export successful: {out_dir / 'model.xml'}")
        except ImportError:
            self.logger.warning("OpenVINO is not installed. Skipping OpenVINO export.")
        except Exception as e:
            self.logger.error(f"OpenVINO export failed: {e}")
