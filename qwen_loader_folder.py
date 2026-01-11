import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import gc
from folder_paths import models_dir

class QwenThinkingLoaderFolder:
    @classmethod
    def INPUT_TYPES(cls):
        qwen_dir = os.path.join(models_dir, "qwen")
        if not os.path.exists(qwen_dir):
            os.makedirs(qwen_dir, exist_ok=True)
        model_list = [d for d in os.listdir(qwen_dir) if os.path.isdir(os.path.join(qwen_dir, d))]
        if not model_list:
            model_list = ["No model folders found in models/qwen/"]

        return {
            "required": {
                "model_folder": (sorted(model_list),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16", "fp32", "auto"], {"default": "bf16"}),
                "compile_model": ("BOOLEAN", {"default": True}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("QWEN_THINKING_MODEL", "QWEN_TOKENIZER", "QWEN_THINKING_MODEL")
    RETURN_NAMES = ("model", "tokenizer", "model_out")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "load"

    def load(self, model_folder, device, dtype, compile_model, keep_loaded):
        torch_dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "auto": "auto"
        }
        torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

        model_path = os.path.join(models_dir, "qwen", model_folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found: {model_path}")

        device_map = "cpu" if device == "cpu" else "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            _attn_implementation="sdpa",  # ← important for speed
            use_safetensors=True
        )

        # SDPA tuning
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Compile
        if compile_model and device == "cuda" and torch.cuda.is_available():
            try:
                print("[Qwen Loader] Compiling...")
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                print("[Qwen Loader] Compiled ✓")
            except Exception as e:
                print(f"[Qwen Loader] Compile failed: {e}")

        model.eval()

        return_model = model
        model_out = model if keep_loaded else None

        if not keep_loaded:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return (return_model, tokenizer, model_out)
