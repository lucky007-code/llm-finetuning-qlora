"""
Standard test runner for LLM Fine-Tuning Pipeline.
"""

import os
import sys
import json

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_engine import PromptFormatter, DatasetAnalyzer
from src.evaluate import LLMEvaluator
from src.export_gguf import export_lora_and_gguf
from src.train_qlora import run_simulated_training, load_config

def run_all_tests():
    print("=" * 70)
    print(" Running LLM Fine-Tuning Pipeline Automated Test Suite")
    print("=" * 70)
    
    # 1. Prompt formatting
    print("[Test 1/7] Testing Prompt Formatting (Llama-3 & ChatML)...", end=" ")
    item = {
        "instruction": "Diagnose the incident.",
        "input": "Error: NullPointerException at Service.java:10",
        "output": '{"severity": "P2-HIGH"}'
    }
    fmt_llama = PromptFormatter.format_sample(item, template_type="llama3", include_output=True)
    fmt_chatml = PromptFormatter.format_sample(item, template_type="chatml", include_output=True)
    assert "<|begin_of_text|>" in fmt_llama
    assert "<|start_header_id|>assistant<|end_header_id|>" in fmt_llama
    assert "<|im_start|>system" in fmt_chatml
    print("[PASSED OK]")

    # 2. Dataset statistics
    print("[Test 2/7] Testing Dataset Statistics Engine...", end=" ")
    train_path = os.path.join(BASE_DIR, "data", "train.jsonl")
    stats = DatasetAnalyzer.compute_token_statistics(train_path)
    assert stats["total_samples"] == 240
    assert stats["mean_tokens"] > 50
    assert len(stats["token_bins"]) == 5
    print(f"[PASSED OK] ({stats['total_samples']} samples analyzed)")

    # 3. Loss Masking Demo
    print("[Test 3/7] Testing Completion Loss Masking (-100 label ignore)...", end=" ")
    sample = {
        "instruction": "Analyze log",
        "input": "Stack trace line 1",
        "output": '{"root_cause": "OOM"}'
    }
    mask_demo = DatasetAnalyzer.get_loss_mask_demo(sample)
    assert mask_demo["prompt_tokens_masked"] > 0
    assert mask_demo["completion_tokens_trained"] > 0
    print(f"[PASSED OK] (Prompt Masked: {mask_demo['prompt_tokens_masked']} tokens, Trained: {mask_demo['completion_tokens_trained']} tokens)")

    # 4. JSON Schema Compliance & Evaluator
    print("[Test 4/7] Testing JSON Schema Compliance Evaluator...", end=" ")
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
    invalid_completions = ["Unstructured generic advice.", '{"incident_id": "INC-3"}']
    metrics = LLMEvaluator.calculate_schema_compliance(valid_completions + invalid_completions)
    assert metrics["valid_json_rate"] == 75.0
    assert metrics["schema_compliance_rate"] == 50.0
    print(f"[PASSED OK] (Valid JSON: {metrics['valid_json_rate']}%, Schema: {metrics['schema_compliance_rate']}%)")

    # 5. ROUGE & BLEU Calculation
    print("[Test 5/7] Testing ROUGE-L & BLEU Scorers...", end=" ")
    cand = "JVM Heap exhaustion inside checkout service"
    ref = "JVM Heap space exhaustion in checkout service"
    rouge = LLMEvaluator.simple_rouge_l(cand, ref)
    bleu = LLMEvaluator.simple_bleu(cand, ref)
    assert rouge > 50.0
    assert bleu > 40.0
    print(f"[PASSED OK] (ROUGE-L: {rouge:.1f}, BLEU: {bleu:.1f})")

    # 6. GGUF Export & Ollama Modelfile
    print("[Test 6/7] Testing GGUF & Ollama Modelfile Packaging...", end=" ")
    test_export_dir = os.path.join(BASE_DIR, "outputs", "test_export")
    manifest = export_lora_and_gguf(output_dir=test_export_dir, quantization_method="q4_k_m")
    assert os.path.exists(os.path.join(test_export_dir, "Modelfile"))
    assert os.path.exists(os.path.join(test_export_dir, "export_manifest.json"))
    assert manifest["quantization_format"] == "GGUF_Q4_K_M"
    print("[PASSED OK]")

    # 7. QLoRA Training Simulation Loop
    print("[Test 7/7] Testing QLoRA Training Pipeline & Checkpointing...", end=" ")
    test_train_dir = os.path.join(BASE_DIR, "outputs", "test_train")
    cfg = load_config(os.path.join(BASE_DIR, "configs", "qlora_config.yaml"))
    train_res = run_simulated_training(cfg, test_train_dir)
    assert train_res["status"] == "COMPLETED"
    assert train_res["trainable_percent"] < 1.0
    assert len(train_res["loss_history"]) == 100
    assert os.path.exists(os.path.join(test_train_dir, "adapter_config.json"))
    print("[PASSED OK]")

    print("-" * 70)
    print("ALL 7/7 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("-" * 70)

if __name__ == "__main__":
    run_all_tests()
