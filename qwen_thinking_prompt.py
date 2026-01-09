import torch
import re

class QwenThinkingPrompt:
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("refined_prompt", "thinking")
    CATEGORY = "Qwen/Thinking"
    FUNCTION = "run"

    def run(self, model, tokenizer, user_prompt, prompt_body, instruction_body,
            max_new_tokens, temperature, top_p):

        # Insert the actual instructions and user prompt into the body template
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

        # Extract final prompt (everything after the last "Final prompt:")
        final_marker = "Final prompt:"
        final_start = text.rfind(final_marker)
        if final_start != -1:
            final = text[final_start + len(final_marker):].strip()
        else:
            final = text.strip()

        # Clean: single line, remove any tags
        final = re.sub(r"<.*?>", "", final)
        final = final.split("\n")[0].strip()

        # Extract thinking content
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else "No thinking captured."

        return (final, thinking)
