import inspect


def patch_accelerate_unwrap_model():
    """
    Backward-compatibility shim for environments where transformers expects
    `Accelerator.unwrap_model(..., keep_torch_compile=...)` but accelerate is
    older and does not accept that keyword yet.
    """
    try:
        from accelerate import Accelerator
    except Exception:
        return

    unwrap_model = getattr(Accelerator, "unwrap_model", None)
    if unwrap_model is None:
        return

    try:
        sig = inspect.signature(unwrap_model)
    except Exception:
        return

    if "keep_torch_compile" in sig.parameters:
        return

    original_unwrap_model = unwrap_model

    def wrapped_unwrap_model(self, model, keep_fp32_wrapper=True, keep_torch_compile=None):
        return original_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = wrapped_unwrap_model

