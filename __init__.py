from .qwen_loader import NewQwenThinkingLoader
from .qwen_thinking_prompt import QwenThinkingPrompt  # Keep your existing prompt node

NODE_CLASS_MAPPINGS = {
    "NewQwenThinkingLoader": NewQwenThinkingLoader,
    "QwenThinkingPrompt": QwenThinkingPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewQwenThinkingLoader": "New Qwen Thinking Loader",
    "QwenThinkingPrompt": "Qwen Thinking Prompt (Before KSampler)",
}