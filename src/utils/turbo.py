from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass
class TurboSettings:
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    pad_to_multiple_of: int | None
    gradient_checkpointing: bool
    use_bf16: bool
    use_fp16: bool
    use_tf32: bool
    qlora_4bit: bool
    group_by_length: bool
    max_length: int


def apply_turbo_runtime(hw_specs: dict, logger=None):
    if hw_specs.get("device") != "cuda":
        return

    capability = hw_specs.get("compute_capability") or (0, 0)
    use_tf32 = capability[0] >= 8

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if logger:
        logger.info(
            "Turbo runtime CUDA: TF32=%s | capability=%s",
            use_tf32,
            capability,
        )


def _default_workers(hw_specs: dict) -> int:
    logical_cores = int(hw_specs.get("logical_cores", os.cpu_count() or 1))
    workers = max(0, min(8, logical_cores - 1))

    # Windows tends to hit diminishing returns earlier with dataloader workers.
    if os.name == "nt":
        workers = 0

    return workers


def build_turbo_settings(hw_specs: dict, workload: str, base_cfg: dict, max_length: int) -> TurboSettings:
    device = hw_specs["device"]
    vram_gb = float(hw_specs.get("vram_gb", 0.0))
    use_bf16 = bool(hw_specs.get("bf16_supported", False))
    use_fp16 = not use_bf16 and device in {"cuda", "mps"}
    qlora_from_profile = bool(base_cfg.get("qlora_4bit", False))
    workers = _default_workers(hw_specs)
    pin_memory = device == "cuda"
    pad_multiple = 16 if device == "cuda" else None
    use_tf32 = device == "cuda" and (hw_specs.get("compute_capability") or (0, 0))[0] >= 8

    if workload == "classifier":
        if device == "cuda":
            if vram_gb >= 20:
                train_bs, eval_bs, ga = 32, 64, 1
            elif vram_gb >= 10:
                train_bs, eval_bs, ga = 16, 32, 1
            elif vram_gb >= 6:
                train_bs, eval_bs, ga = 8, 16, 2
            else:
                train_bs, eval_bs, ga = 4, 8, 4
        elif device == "mps":
            train_bs, eval_bs, ga = 8, 8, 2
        else:
            train_bs, eval_bs, ga = 4, 4, 8

        return TurboSettings(
            train_batch_size=max(base_cfg.get("batch_size", 4), train_bs),
            eval_batch_size=max(eval_bs, train_bs),
            gradient_accumulation_steps=min(base_cfg.get("gradient_accumulation_steps", 1), ga),
            dataloader_num_workers=workers,
            dataloader_pin_memory=pin_memory,
            pad_to_multiple_of=pad_multiple,
            gradient_checkpointing=False,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            use_tf32=use_tf32,
            qlora_4bit=False,
            group_by_length=True,
            max_length=max_length,
        )

    qlora_4bit = device == "cuda" and (qlora_from_profile or vram_gb < 18)
    if device == "cuda":
        if vram_gb >= 38:           # A100 40GB reports ~39.6 GB real
            train_bs, eval_bs, ga = 8, 8, 1
            grad_ckpt = False
        elif vram_gb >= 24:
            train_bs, eval_bs, ga = 4, 4, 1
            grad_ckpt = True        # safety for 24-38 GB range with large models
        elif vram_gb >= 12:
            train_bs, eval_bs, ga = 2, 2, 2
            grad_ckpt = qlora_4bit
        elif vram_gb >= 8:
            train_bs, eval_bs, ga = 1, 1, 8
            grad_ckpt = True
        else:
            train_bs, eval_bs, ga = 1, 1, 16
            grad_ckpt = True
    elif device == "mps":
        train_bs, eval_bs, ga = 1, 1, 8
        grad_ckpt = True
    else:
        train_bs, eval_bs, ga = 1, 1, 16
        grad_ckpt = False

    return TurboSettings(
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        gradient_accumulation_steps=ga,
        dataloader_num_workers=workers,
        dataloader_pin_memory=pin_memory,
        pad_to_multiple_of=pad_multiple,
        gradient_checkpointing=grad_ckpt,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        use_tf32=use_tf32,
        qlora_4bit=qlora_4bit,
        group_by_length=True,
        max_length=max_length,
    )
