"""
Data Engine: Tokenization, Prompt Formatting, Token Statistics, and Loss Masking.
Supports Llama 3, ChatML, and Alpaca instruction templates.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# System prompt used across the SRE incident triage domain
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Principal Site Reliability & Systems Diagnostics Engineer (SRE). "
    "Analyze raw system telemetry, stack traces, and incident alerts. "
    "Output a strictly formatted JSON response containing root_cause, severity, blast_radius, "
    "mitigation_actions, and preventative_measure."
)

class PromptFormatter:
    """Formats raw dataset items into structured prompt strings for different LLM backbones."""

    @staticmethod
    def format_llama3(instruction: str, input_text: str, output_text: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Llama-3 special token format."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\n[INPUT LOGS]:\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if output_text is not None:
            prompt += f"{output_text}<|eot_id|>"
        return prompt

    @staticmethod
    def format_chatml(instruction: str, input_text: str, output_text: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """ChatML (Mistral / Qwen / generic) template."""
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n\n[INPUT LOGS]:\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if output_text is not None:
            prompt += f"{output_text}<|im_end|>"
        return prompt

    @staticmethod
    def format_alpaca(instruction: str, input_text: str, output_text: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Classic Alpaca instruction template."""
        prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
        if output_text is not None:
            prompt += f"{output_text}"
        return prompt

    @classmethod
    def format_sample(cls, item: Dict[str, Any], template_type: str = "llama3", include_output: bool = True) -> str:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "") if include_output else None

        if template_type == "llama3":
            return cls.format_llama3(instruction, input_text, output_text)
        elif template_type == "chatml":
            return cls.format_chatml(instruction, input_text, output_text)
        elif template_type == "alpaca":
            return cls.format_alpaca(instruction, input_text, output_text)
        else:
            raise ValueError(f"Unknown template_type: {template_type}")

class DatasetAnalyzer:
    """Analyzes token lengths, truncation risks, and vocabulary coverage."""

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records

    @classmethod
    def compute_token_statistics(cls, filepath: str, template_type: str = "llama3") -> Dict[str, Any]:
        """Calculates token distribution metrics (approximated via whitespace/BPE heuristic or tokenizer)."""
        records = cls.load_jsonl(filepath)
        if not records:
            return {"total_samples": 0, "mean_tokens": 0, "max_tokens": 0}

        prompt_lengths = []
        completion_lengths = []
        total_lengths = []

        for item in records:
            # Full formatted text
            full_text = PromptFormatter.format_sample(item, template_type=template_type, include_output=True)
            prompt_only = PromptFormatter.format_sample(item, template_type=template_type, include_output=False)
            output_text = item.get("output", "")

            # Heuristic token approximation: ~1.3 tokens per word for code/logs/JSON
            p_len = int(len(prompt_only.split()) * 1.3)
            c_len = int(len(output_text.split()) * 1.3)
            t_len = int(len(full_text.split()) * 1.3)

            prompt_lengths.append(p_len)
            completion_lengths.append(c_len)
            total_lengths.append(t_len)

        return {
            "total_samples": len(records),
            "mean_tokens": float(np.mean(total_lengths)),
            "median_tokens": float(np.median(total_lengths)),
            "p95_tokens": float(np.percentile(total_lengths, 95)),
            "max_tokens": int(np.max(total_lengths)),
            "min_tokens": int(np.min(total_lengths)),
            "mean_prompt_tokens": float(np.mean(prompt_lengths)),
            "mean_completion_tokens": float(np.mean(completion_lengths)),
            "token_bins": [
                {"range": "0-256", "count": int(sum(1 for l in total_lengths if l <= 256))},
                {"range": "257-512", "count": int(sum(1 for l in total_lengths if 256 < l <= 512))},
                {"range": "513-1024", "count": int(sum(1 for l in total_lengths if 512 < l <= 1024))},
                {"range": "1025-2048", "count": int(sum(1 for l in total_lengths if 1024 < l <= 2048))},
                {"range": ">2048", "count": int(sum(1 for l in total_lengths if l > 2048))},
            ]
        }

    @staticmethod
    def get_loss_mask_demo(sample: Dict[str, Any], template_type: str = "llama3") -> Dict[str, Any]:
        """
        Demonstrates how DataCollatorForCompletionOnlyLM masks prompt tokens with -100
        so that loss is only calculated on the assistant response.
        """
        full_text = PromptFormatter.format_sample(sample, template_type=template_type, include_output=True)
        prompt_only = PromptFormatter.format_sample(sample, template_type=template_type, include_output=False)
        output_text = sample.get("output", "")

        words = full_text.split()
        prompt_words_count = len(prompt_only.split())
        
        # Token mask visualization
        token_visual = []
        for i, word in enumerate(words):
            is_prompt = i < prompt_words_count
            token_visual.append({
                "token": word,
                "target_loss_id": -100 if is_prompt else 1, # -100 is PyTorch CrossEntropyLoss ignore_index
                "is_masked": is_prompt
            })

        return {
            "full_text": full_text,
            "prompt_tokens_masked": prompt_words_count,
            "completion_tokens_trained": len(words) - prompt_words_count,
            "token_visual": token_visual[:30] # Return preview
        }

if __name__ == "__main__":
    stats = DatasetAnalyzer.compute_token_statistics("./data/train.jsonl")
    print(f"[Data Engine] Dataset Token Statistics:\n{json.dumps(stats, indent=2)}")
