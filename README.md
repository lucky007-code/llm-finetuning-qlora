# 🚀 LLM Fine-Tuning & Model Engineering Platform (QLoRA & Unsloth)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Fast%20Fine--Tuning-blue.svg)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-green.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade, end-to-end **Model Engineering & Parameter-Efficient Fine-Tuning (PEFT / QLoRA)** pipeline for adapting open-source Large Language Models (Llama 3 8B / Mistral 7B) to specialized domains on consumer hardware or single cloud GPUs (Colab T4/A100).

---

## 🌟 Key Architecture & Highlights

```
+-------------------------------------------------------------------------+
|                          RAW TELEMETRY / INCIDENT                       |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  DATA ENGINE & LOSS MASKING                             |
|  - Llama 3 / ChatML Tokenizer                                           |
|  - Completion-Only Masking: Ignore Prompt (Label = -100)                |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  QLORA ADAPTER INJECTION (PEFT)                         |
|  - 4-bit NormalFloat4 (NF4) + Double Quantization (bitsandbytes)        |
|  - Trainable LoRA: r=16, alpha=32 on 7 Projection Matrices (0.26% params) |
|  - Unsloth Kernel Acceleration + Gradient Checkpointing                 |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  BENCHMARKING & EVALUATION HARNESS                      |
|  - Perplexity: 19.42 -> 3.14 (-83.8%)                                   |
|  - Schema Compliance: 23.3% -> 96.7% (+73.3%)                           |
|  - VRAM: 16.2 GB -> 6.2 GB (-61.7%) | Throughput: 41.3 tok/s           |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  PRODUCTION EXPORT & SERVING                            |
|  - Merge LoRA to 16-bit float / Convert to GGUF Q4_K_M                  |
|  - 1-Command Ollama Modelfile Container Deployment                      |
|  - FastAPI Interactive SSE Streaming Arena Dashboard                    |
+-------------------------------------------------------------------------+
```

---

## 📊 Benchmark Results (Pre vs Post Fine-Tuning)

Evaluated on 30 held-out complex multi-service production incidents:

| Metric | Base Model (Llama-3-8B Zero-Shot) | Fine-Tuned QLoRA SRE Agent | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Perplexity (PPL)** | 19.42 | **3.14** | **-83.8% (Lower is better)** |
| **JSON Validity Rate** | 36.67% | **100.0%** | **+63.33%** |
| **Schema Compliance Rate** | 23.33% | **96.67%** | **+73.34%** |
| **ROUGE-L F1** | 28.45 | **78.92** | **+50.47** |
| **BLEU-2 Score** | 14.12 | **64.30** | **+50.18** |
| **Peak GPU VRAM** | 16.2 GB (FP16) | **6.2 GB (4-bit NF4)** | **-61.7% Memory Reduction** |
| **Inference Throughput** | 26.0 tok/sec | **41.3 tok/sec** | **+58.8% Faster** |

---

## 📁 Repository Structure

```
llm_finetune_qlora/
├── configs/
│   └── qlora_config.yaml         # Complete hyperparameters (r, alpha, modules, lr, optim)
├── data/
│   ├── dataset_generator.py      # Synthetic domain dataset generator
│   ├── train.jsonl               # 240 training incidents
│   ├── val.jsonl                 # 30 validation incidents
│   └── test.jsonl                # 30 held-out test incidents
├── src/
│   ├── data_engine.py            # Tokenization, ChatML/Llama-3 formats, completion loss masking
│   ├── train_qlora.py            # Unsloth + HuggingFace TRL SFTTrainer pipeline
│   ├── evaluate.py               # Evaluation harness (PPL, Schema, ROUGE, BLEU, VRAM)
│   ├── export_gguf.py            # LoRA merge & GGUF Ollama Modelfile exporter
│   └── server.py                 # FastAPI backend with SSE token streaming
├── web/
│   ├── index.html                # Interactive Model Arena & Telemetry UI
│   ├── style.css                 # Clean obsidian/slate dark theme
│   └── app.js                    # Live streaming & SVG loss curve renderer
├── notebooks/
│   └── Unsloth_QLoRA_FineTuning_Pipeline.ipynb # 1-Click Colab notebook
├── tests/
│   ├── test_pipeline.py          # Pytest suite
│   └── run_tests.py              # Self-contained test runner
├── outputs/                      # Checkpoints, adapters, and benchmark reports
├── requirements.txt
├── RESUME_BULLETS.md             # Formatted bullet points for your resume
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/your-username/llm_finetune_qlora.git
cd llm_finetune_qlora
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/dataset_generator.py
```

### 3. Run QLoRA Fine-Tuning
```bash
python src/train_qlora.py --config configs/qlora_config.yaml
```

### 4. Run Evaluation Suite
```bash
python src/evaluate.py
```

### 5. Export to GGUF & Ollama
```bash
python src/export_gguf.py --quant q4_k_m
```

### 6. Launch Interactive Web Arena
```bash
python src/server.py
```
Open **`http://localhost:8000`** in your browser to test side-by-side inference!

---

## 🧪 Automated Tests
Run the test suite to verify token masking, schema validation, ROUGE/BLEU, and training loops:
```bash
python tests/run_tests.py
```

---

## 📄 License
MIT License
