import logging
import os
import psutil
import torch
import importlib.util

def detect_hardware_profile(logger: logging.Logger | None = None) -> dict:
    """
    Detects hardware capabilities and returns a specs dictionary:
    - profile: HIGH_VRAM, MID_VRAM, LOW_VRAM, CPU_ONLY
    - bf16_supported: bool
    - vram_gb: float
    - ram_gb: float
    - name: str
    - device: str (cuda, mps, xpu, or cpu)
    - backend: str (nvidia, apple, intel, generic)
    """
    specs = {
        "profile": "CPU_ONLY",
        "bf16_supported": False,
        "vram_gb": 0.0,
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
        best_vram_gb = 0.0
        best_name = "Unknown NVIDIA GPU"
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            if vram_gb > best_vram_gb:
                best_vram_gb = vram_gb
                best_name = props.name
                specs["compute_capability"] = (props.major, props.minor)

        specs["vram_gb"] = best_vram_gb
        specs["name"] = best_name
        
        try:
            specs["bf16_supported"] = torch.cuda.is_bf16_supported()
        except Exception:
            specs["bf16_supported"] = False

        if best_vram_gb >= 24.0:
            specs["profile"] = "HIGH_VRAM"
        elif best_vram_gb >= 6.0:
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
