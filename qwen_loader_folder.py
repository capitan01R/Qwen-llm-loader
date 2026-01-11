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
        
        # List only subdirectories (each = a model)
        model_list = [d for d in os.listdir(qwen_dir) if os.path.isdir(os.path.join(qwen_dir, d))]
        if not model_list:
            model_list = ["No model folders found in models/qwen/"]

        return {
            "required": {
                "model_folder": (sorted(model_list),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16", "fp32", "auto"], {"default": "bf16"}),  # modern naming + auto
                "compile_model": ("BOOLEAN", {"default": True}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("QWEN_THINKING_MODEL", "QWEN_TOKENIZER", "QWEN_THINKING_MODEL")
    RETURN_NAMES = ("model", "tokenizer", "model_out")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "load"

    def load(self, model_folder, device, dtype, compile_model, keep_loaded):
        # Modern dtype mapping
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

        print(f"[Qwen Folder Loader] Loading model from: {model_path}")
        print(f"[Qwen Folder Loader] dtype: {dtype} → {torch_dtype}")
        print(f"[Qwen Folder Loader] device: {device} | compile: {compile_model} | keep_loaded: {keep_loaded}")

        device_map = "cpu" if device == "cpu" else "auto"

        # ───────────────────────────────────────────────────────
        # Modern loading with SDPA (best native choice for Ampere)
        # ───────────────────────────────────────────────────────
        loading_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "_attn_implementation": "sdpa",              # ← explicit SDPA
            "use_safetensors": True,
        }

        print("[Qwen Folder Loader] Using torch SDPA (built-in scaled dot-product attention)")

        try:
            config = AutoConfig.from_pretrained(model_path, **loading_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path, **loading_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_path, **loading_kwargs)
        except Exception as e:
            print(f"[Qwen Folder Loader] Loading failed: {str(e)}")
            raise

        model.eval()

        # Enable best available SDPA optimizations (free gains)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Compile the model (biggest inference speedup after warmup)
        compiled = False
        if compile_model and device == "cuda" and torch.cuda.is_available():
            try:
                print("[Qwen Folder Loader] Compiling model (reduce-overhead + dynamic)...")
                print("   → First generation will be slow, then much faster")
                model = torch.compile(
                    model,
                    mode="reduce-overhead",     # best for RTX 30-series inference
                    dynamic=True,               # crucial for variable prompt lengths
                    fullgraph=False,
                )
                compiled = True
                print("[Qwen Folder Loader] Compilation successful ✓")
            except Exception as e:
                print(f"[Qwen Folder Loader] Compilation failed: {e}")
                print("   → Continuing in eager mode (still uses SDPA)")

        return_model = model
        model_out = model if keep_loaded else None

        # Clean up if not keeping loaded
        if not keep_loaded:
            print("[Qwen Folder Loader] keep_loaded=False → unloading model")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return (return_model, tokenizer, model_out)
