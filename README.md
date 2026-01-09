# Qwen-llm-loader
Local Qwen loader and prompt refiner for ComfyUI.
# ComfyUI-llm-loader

Local Qwen3-4B-Thinking-2507/ Qwen3_4b loader/ general LLM models and prompt refiner for ComfyUI.

This custom node lets you load the Qwen3-4B-Thinking-2507 or any LLM model completely offline and use it to refine text prompts with full control over the instructions. It supports visible thinking output, and optional full memory unloading to free VRAM after use.

## Features
- Fully local loading (no internet required after downloading the model)
- Flexible prompt template and instruction bodies
- Visible chain-of-thought ("thinking") output
- Optional keep_loaded toggle to completely unload the model from GPU and RAM
- Works with any Stable Diffusion workflow

## Installation

1. Clone or download this repository into your `ComfyUI/custom_nodes/` folder:
2. (or download as ZIP and extract)

2. Download the model files, For this example I'm using Qwen3-4B-Thinking-2507 (download all repo files):  
https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

OR
   Download the merged model files from here (download all repo files):
https://huggingface.co/Capitan01R/qwen-thinking-merged/tree/main/qwen_thinking_merged

4. Place all files (`config.json`, `tokenizer.*`, `*.safetensors`, `model.safetensors.index.json` or shards, and any `.py` files) into:  
`ComfyUI/models/qwen/downloaded-model-folder`

5. Restart ComfyUI

No additional pip packages required.

## Usage

1. Add the **New Qwen Thinking Loader** node
- Choose device (cuda/cpu), dtype, and keep_loaded as needed, I personally keep it off-loaded

2. Connect it to the **Qwen Thinking Prompt** node
- Paste your raw prompt into `user_prompt`
- Customize `instruction_body` for different refinement styles
- Adjust temperature, top_p, max_new_tokens as desired

3. Use the `refined_prompt` output in your text-to-image workflow

## Nodes

- **New Qwen Thinking Loader** – Loads the model locally
- **Qwen Thinking Prompt** – Refines prompts with customizable instructions and thinking visibility


## Credits

Based on Qwen/Qwen3-4B-Thinking-2507 from Hugging Face.
