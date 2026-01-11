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
        # List only directories (subfolders = models)
        model_list = [d for d in os.listdir(qwen_dir) if os.path.isdir(os.path.join(qwen_dir, d))]
        if not model_list:
            model_list = ["No model folders found in models/qwen/"]

        return {
            "required": {
                "model_folder": (sorted(model_list),),
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
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        model_path = os.path.join(models_dir, "qwen", model_folder)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found: {model_path}")

        print(f"[Qwen Folder Loader] Loading from folder: {model_path}")

        device_map = "cpu" if device == "cpu" else "auto"

        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_safetensors=True
        )

        model.eval()

        return_model = model
        model_out = model if keep_loaded else None

        if not keep_loaded:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return (return_model, tokenizer, model_out)
