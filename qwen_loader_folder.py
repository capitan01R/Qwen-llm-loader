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
                "model_folder": (sorted(model_list), {"tooltip": "Select your Qwen model folder"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16", "fp32", "auto"], {"default": "bf16"}),
                "compile_model": ("BOOLEAN", {"default": True, "tooltip": "Enable torch.compile for faster inference after first run"}),
                "use_multi_gpu": ("BOOLEAN", {"default": True, "tooltip": "Use all visible GPUs (device_map='auto'). Turn off for single GPU only."}),
                "keep_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep model in memory for faster repeated use"}),
            }
        }

    RETURN_TYPES = ("QWEN_THINKING_MODEL", "QWEN_TOKENIZER", "QWEN_THINKING_MODEL")
    RETURN_NAMES = ("model", "tokenizer", "model_out")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "load"

    def load(self, model_folder, device, dtype, compile_model, use_multi_gpu, keep_loaded):
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
        print(f"[Qwen Folder Loader] device: {device} | compile: {compile_model} | multi_gpu: {use_multi_gpu} | keep_loaded: {keep_loaded}")

        # Device map logic (multi-GPU support)
        if device == "cpu":
            device_map = "cpu"
        elif use_multi_gpu:
            device_map = "auto"
            print("[Qwen Folder Loader] Multi-GPU mode → device_map='auto'")
        else:
            device_map = "cuda"  # or None → default GPU (usually cuda:0)
            print("[Qwen Folder Loader] Single-GPU mode → loading to default cuda device")

        loading_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "_attn_implementation": "sdpa",  # Built-in optimized attention
            "use_safetensors": True,
        }

        print("[Qwen Folder Loader] Using torch SDPA (scaled dot-product attention)")

        try:
            # Load tokenizer FIRST
            tokenizer = AutoTokenizer.from_pretrained(model_path, **loading_kwargs)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_path, **loading_kwargs)
        except Exception as e:
            print(f"[Qwen Folder Loader] Loading failed: {str(e)}")
            raise

        model.eval()

        # Enable SDPA memory/speed optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Optional compilation (big speedup after first run)
        if compile_model and device == "cuda" and torch.cuda.is_available():
            try:
                print("[Qwen Folder Loader] Compiling model (reduce-overhead + dynamic)...")
                print("   → First generation will be slow, then much faster")
                model = torch.compile(
                    model,
                    mode="reduce-overhead",   # Best for inference on RTX 30/40/50 series
                    dynamic=True,             # Important for variable prompt lengths
                    fullgraph=False,
                )
                print("[Qwen Folder Loader] Compilation successful ✓")
            except Exception as e:
                print(f"[Qwen Folder Loader] Compilation failed: {e}")
                print("   → Continuing in eager mode (still good)")

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
