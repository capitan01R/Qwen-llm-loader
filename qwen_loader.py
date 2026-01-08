import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import gc

class NewQwenThinkingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_folder": ("STRING", {"default": "models/qwen"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "bfloat16"}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("QWEN_THINKING_MODEL", "QWEN_TOKENIZER", "QWEN_THINKING_MODEL")
    RETURN_NAMES = ("model", "tokenizer", "model_out")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "load"

    def load(self, model_folder, device, dtype, keep_loaded):
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        # Resolve absolute path
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_folder_abs = os.path.join(comfy_root, model_folder)
        model_folder_abs = os.path.abspath(model_folder_abs)

        if not os.path.exists(model_folder_abs):
            raise FileNotFoundError(f"Model folder not found at {model_folder_abs}")

        device_map = "cpu" if device == "cpu" else "auto"

        # Load config, tokenizer, and model locally
        config = AutoConfig.from_pretrained(model_folder_abs, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_folder_abs, local_files_only=True, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_folder_abs,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

        model.eval()

        # Prepare outputs
        return_model = model  # Always return to the prompt node
        model_out = model if keep_loaded else None

        # Only unload AFTER we have safely returned the reference
        if not keep_loaded:
            print("[Qwen Thinking Loader] keep_loaded=False → fully unloading model from memory")
            del model  # Remove local reference
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()  # Free RAM immediately

        return (return_model, tokenizer, model_out)