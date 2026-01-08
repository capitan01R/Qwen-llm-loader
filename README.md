# Qwen-llm-loader
Local Qwen loader and prompt refiner for ComfyUI.
# ComfyUI-QwenThinkingLoader

Local Qwen3-4B-Thinking-2507 loader and prompt refiner for ComfyUI.

This custom node lets you load the Qwen3-4B-Thinking-2507 model completely offline and use it to refine text prompts with full control over the instructions. It supports visible thinking output, fixed seed for reproducible results, and optional full memory unloading to free VRAM after use.

## Features
- Fully local loading (no internet required after downloading the model)
- Flexible prompt template and instruction bodies
- Visible chain-of-thought ("thinking") output
- Fixed seed support for deterministic refinement
- Optional keep_loaded toggle to completely unload the model from GPU and RAM
- Works with any Stable Diffusion workflow

## Installation

1. Clone or download this repository into your `ComfyUI/custom_nodes/` folder:
2. (or download as ZIP and extract)

2. Download the model files from:  
https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

3. Place all files (`config.json`, `tokenizer.*`, `*.safetensors` or shards, and any `.py` files) into:  
`ComfyUI/models/qwen/`

4. Restart ComfyUI

No additional pip packages required.

## Usage

1. Add the **New Qwen Thinking Loader** node
- Set `model_folder` to `models/qwen`
- Choose device (cuda/cpu), dtype, and keep_loaded as needed

2. Connect it to the **Qwen Thinking Prompt** node
- Paste your raw prompt into `user_prompt`
- Customize `instruction_body` for different refinement styles
- Adjust seed, temperature, top_p, max_new_tokens as desired

3. Use the `refined_prompt` output in your text-to-image workflow

## Nodes

- **New Qwen Thinking Loader** – Loads the model locally
- **Qwen Thinking Prompt** – Refines prompts with customizable instructions and thinking visibility


## Credits

Based on Qwen/Qwen3-4B-Thinking-2507 from Hugging Face.
