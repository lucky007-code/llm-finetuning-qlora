"""
Unit and Integration Tests for LLM Fine-Tuning Pipeline.
"""

import os
import json
import pytest
from src.data_engine import PromptFormatter, DatasetAnalyzer
from src.evaluate import LLMEvaluator
from src.export_gguf import export_lora_and_gguf
from src.train_qlora import run_simulated_training, load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_prompt_formatting_llama3():
    item = {
        "instruction": "Diagnose the incident.",
        "input": "Error: NullPointerException at Service.java:10",
        "output": '{"severity": "P2-HIGH"}'
    }
    formatted = PromptFormatter.format_sample(item, template_type="llama3", include_output=True)
    assert "<|begin_of_text|>" in formatted
    assert "<|start_header_id|>system<|end_header_id|>" in formatted
    assert "<|start_header_id|>user<|end_header_id|>" in formatted
    assert "<|start_header_id|>assistant<|end_header_id|>" in formatted
    assert '{"severity": "P2-HIGH"}' in formatted

def test_prompt_formatting_chatml():
    item = {
        "instruction": "Diagnose the incident.",
        "input": "Error: Timeout",
        "output": '{"severity": "P1-CRITICAL"}'
    }
    formatted = PromptFormatter.format_sample(item, template_type="chatml", include_output=True)
    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted

def test_dataset_statistics():
    data_path = os.path.join(BASE_DIR, "data", "train.jsonl")
    if os.path.exists(data_path):
        stats = DatasetAnalyzer.compute_token_statistics(data_path)
        assert stats["total_samples"] > 0
        assert stats["mean_tokens"] > 0
        assert len(stats["token_bins"]) == 5

def test_loss_masking_demo():
    sample = {
        "instruction": "Analyze log",
        "input": "Stack trace line 1",
        "output": '{"root_cause": "OOM"}'
    }
    demo = DatasetAnalyzer.get_loss_mask_demo(sample)
    assert demo["prompt_tokens_masked"] > 0
    assert demo["completion_tokens_trained"] > 0
    assert len(demo["token_visual"]) > 0

def test_schema_compliance_evaluator():
    valid_completions = [
        json.dumps({
            "incident_id": "INC-1",
            "severity": "P1-CRITICAL",
            "root_cause": "OOM",
            "blast_radius": ["checkout"],
            "mitigation_actions": ["kubectl restart"],
            "preventative_measure": "Tune RAM"
        }),
        json.dumps({
            "incident_id": "INC-2",
            "severity": "P2-HIGH",
            "root_cause": "Deadlock",
            "blast_radius": ["db"],
            "mitigation_actions": ["terminate pid"],
            "preventative_measure": "Add index"
        })
    ]
    invalid_completions = [
        "This is generic text without JSON.",
        '{"incident_id": "INC-3"}' # Missing required keys
    ]

    metrics = LLMEvaluator.calculate_schema_compliance(valid_completions + invalid_completions)
    assert metrics["valid_json_rate"] == 75.0
    assert metrics["schema_compliance_rate"] == 50.0
    assert metrics["total_tested"] == 4

def test_rouge_and_bleu_metrics():
    cand = "JVM Heap exhaustion inside checkout service"
    ref = "JVM Heap space exhaustion in checkout service"
    rouge = LLMEvaluator.simple_rouge_l(cand, ref)
    bleu = LLMEvaluator.simple_bleu(cand, ref)
    assert rouge > 50.0
    assert bleu > 40.0

def test_export_pipeline(tmp_path):
    outdir = str(tmp_path / "export_test")
    manifest = export_lora_and_gguf(output_dir=outdir, quantization_method="q4_k_m")
    assert os.path.exists(os.path.join(outdir, "Modelfile"))
    assert os.path.exists(os.path.join(outdir, "export_manifest.json"))
    assert manifest["quantization_format"] == "GGUF_Q4_K_M"

def test_training_simulation(tmp_path):
    outdir = str(tmp_path / "train_test")
    config = load_config(os.path.join(BASE_DIR, "configs", "qlora_config.yaml"))
    results = run_simulated_training(config, outdir)
    assert results["status"] == "COMPLETED"
    assert results["trainable_percent"] < 1.0
    assert len(results["loss_history"]) == 100
