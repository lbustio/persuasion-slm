import logging
import os
import psutil
import torch
import importlib.util
import subprocess
import json


def _get_free_vram_gb(gpu_index: int) -> float:
    """Get free VRAM in GB for a specific GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader", "--id", str(gpu_index)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 0.0
        free_mi = result.stdout.strip()
        if not free_mi:
            return 0.0
        free_value = free_mi.split()[0]
        return int(free_value) / 1024
    except Exception:
        return 0.0


def detect_hardware_profile(logger: logging.Logger | None = None) -> dict:
    """
    Detects hardware capabilities and returns a specs dictionary:
    - profile: HIGH_VRAM, MID_VRAM, LOW_VRAM, CPU_ONLY
    - bf16_supported: bool
    - vram_gb: float (FREE VRAM on the assigned GPU)
    - vram_total_gb: float (total GPU memory)
    - ram_gb: float
    - name: str
    - device: str (cuda, mps, xpu, or cpu)
    - backend: str (nvidia, apple, intel, generic)
    """
    specs = {
        "profile": "CPU_ONLY",
        "bf16_supported": False,
        "vram_gb": 0.0,
        "vram_total_gb": 0.0,
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "name": "CPU Generic",
        "device": "cpu",
        "backend": "generic",
        "logical_cores": os.cpu_count() or 1,
        "physical_cores": psutil.cpu_count(logical=False) or (os.cpu_count() or 1),
        "compute_capability": None,
        "gpu_count": 0,
        "intel_extensions": False,
        "openvino_available": importlib.util.find_spec("openvino") is not None,
    }

    # 1. Detectar NVIDIA (CUDA) - Prioridad Alta
    if torch.cuda.is_available():
        specs["device"] = "cuda"
        specs["backend"] = "nvidia"
        device_count = torch.cuda.device_count()
        specs["gpu_count"] = device_count

        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        specs["vram_total_gb"] = total_vram_gb
        specs["compute_capability"] = (props.major, props.minor)

        # Query free VRAM on the assigned GPU (GPU 0 from PyTorch perspective)
        free_vram_gb = _get_free_vram_gb(0)
        if free_vram_gb > 0:
            specs["vram_gb"] = free_vram_gb
            specs["name"] = f"{props.name} ({free_vram_gb:.1f}GB free / {total_vram_gb:.1f}GB total)"
        else:
            # Fallback: assume all VRAM is free if nvidia-smi fails
            specs["vram_gb"] = total_vram_gb
            specs["name"] = props.name

        try:
            specs["bf16_supported"] = torch.cuda.is_bf16_supported()
        except Exception:
            specs["bf16_supported"] = False

        free_vram = specs["vram_gb"]
        if free_vram >= 24.0:
            specs["profile"] = "HIGH_VRAM"
        elif free_vram >= 6.0:
            specs["profile"] = "MID_VRAM"
        else:
            specs["profile"] = "LOW_VRAM"

    # 2. Detectar Intel (XPU / IPEX)
    elif importlib.util.find_spec("intel_extension_for_pytorch") is not None:
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(ipex, "xpu") and ipex.xpu.is_available():
                specs["device"] = "xpu"
                specs["backend"] = "intel"
                specs["intel_extensions"] = True
                specs["name"] = ipex.xpu.get_device_name(0)
                # Estimacion de VRAM para Intel (depende de la implementacion de IPEX)
                # Por ahora usamos perfil MID si detectamos IPEX con GPU activa
                specs["profile"] = "MID_VRAM"
        except Exception:
            pass

    # 3. Detectar Apple Silicon (MPS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        specs["device"] = "mps"
        specs["backend"] = "apple"
        specs["name"] = "Apple Silicon GPU"
        if specs["ram_gb"] >= 32:
            specs["profile"] = "HIGH_VRAM"
        elif specs["ram_gb"] >= 16:
            specs["profile"] = "MID_VRAM"
        else:
            specs["profile"] = "LOW_VRAM"
            
    # 4. Logica para CPU (Si no hay GPU detectada)
    if specs["device"] == "cpu":
        try:
            torch.zeros(1, dtype=torch.bfloat16)
            specs["bf16_supported"] = True 
        except Exception:
            specs["bf16_supported"] = False

        if specs["ram_gb"] >= 32:
            specs["profile"] = "MID_VRAM"
        elif specs["ram_gb"] >= 16:
            specs["profile"] = "LOW_VRAM"
        else:
            specs["profile"] = "CPU_ONLY"

    if logger:
        bf16_status = "Soportado" if specs["bf16_supported"] else "No soportado"
        ov_status = "Disponible" if specs["openvino_available"] else "No detectado"
        mem_info = f"{specs['vram_gb']:.1f}GB VRAM" if specs['vram_gb'] > 0 else f"{specs['ram_gb']:.1f}GB RAM"
        
        logger.info("-" * 40)
        logger.info(" AUDITORIA DE HARDWARE")
        logger.info("-" * 40)
        logger.info(f" DISPOSITIVO: {specs['name']}")
        logger.info(f" BACKEND:     {specs['backend'].upper()} ({specs['device']})")
        logger.info(f" MEMORIA:     {mem_info}")
        logger.info(f" ESTRATEGIA:  {specs['profile']}")
        logger.info(f" BF16:        {bf16_status}")
        logger.info(f" OPENVINO:    {ov_status}")
        logger.info("-" * 40)

    return specs
