import torch
import re
import random

class QwenThinkingPrompt:
    _last_seed = None  # Tracks for increment/decrement across runs

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_THINKING_MODEL",),
                "tokenizer": ("QWEN_TOKENIZER",),
                "user_prompt": ("STRING", {"multiline": True, "dynamicPromptSupport": True}),
                "prompt_body": ("STRING", {
                    "multiline": True,
                    "default": "<think>\n{instructions}\n\nUser prompt:\n{user_prompt}\n\nFinal prompt:\n</think>\n\nFinal prompt:"
                }),
                "instruction_body": ("STRING", {
                    "multiline": True,
                    "default": """You are an expert Stable Diffusion prompt engineer.

Rules:
- Preserve the exact original meaning and intent.
- Do NOT add any new concepts, styles, subjects, or details not present or implied.
- Remove redundancy and make the prompt concise but descriptive.
- Structure as a single, clean, comma-separated line.
- Output ONLY the final prompt. No reasoning, no extra text."""
                }),
                "max_new_tokens": ("INT", {"default": 256, "min": 32, "max": 512}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.2}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),  # ← renamed from "seed"
                "control": (["randomize", "fixed", "increment", "decrement"], {"default": "randomize"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("refined_prompt", "thinking", "used_seed")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "run"

    def run(self, model, tokenizer, user_prompt, prompt_body, instruction_body,
            max_new_tokens, temperature, top_p, base_seed=42, control="randomize"):

        # Seed logic
        if control == "randomize":
            current_seed = random.randint(0, 2147483647)
        elif control == "fixed":
            current_seed = base_seed
        elif control == "increment":
            current_seed = (self._last_seed if self._last_seed is not None else base_seed) + 1
        elif control == "decrement":
            current_seed = (self._last_seed if self._last_seed is not None else base_seed) - 1
        else:
            current_seed = base_seed

        current_seed = current_seed % 2147483648  # keep in range
        QwenThinkingPrompt._last_seed = current_seed

        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        # Everything below is 100% identical to your original node
        full_prompt = prompt_body.format(
            instructions=instruction_body.strip(),
            user_prompt=user_prompt.strip()
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        final_marker = "Final prompt:"
        final_start = text.rfind(final_marker)
        if final_start != -1:
            final = text[final_start + len(final_marker):].strip()
        else:
            final = text.strip()

        final = re.sub(r"<.*?>", "", final)
        final = final.split("\n")[0].strip()

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else "No thinking captured."

        return (final, thinking, current_seed)
