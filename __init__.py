from .qwen_loader_folder import QwenThinkingLoaderFolder
from .qwen_thinking_prompt import QwenThinkingPrompt

NODE_CLASS_MAPPINGS = {
    "QwenThinkingLoaderFolder": QwenThinkingLoaderFolder,
    "QwenThinkingPrompt": QwenThinkingPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenThinkingLoaderFolder": "Qwen Thinking Loader (Merged or Sharded)",
    "QwenThinkingPrompt": "Qwen Thinking Prompt (Before KSampler)",
}
