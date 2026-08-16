# Tailored Resume Bullet Points: LLM Fine-Tuning & Model Engineering

Here are impactful, quantified resume bullet points highlighting your Model Engineering and Parameter-Efficient Fine-Tuning (PEFT) capabilities. Choose the ones that best fit your resume format:

---

## 🎯 Role: Large Language Model (LLM) / Machine Learning Engineer

### Option 1: Comprehensive & Quantified (Recommended)
> - **Fine-Tuned Llama-3 8B with QLoRA & Unsloth:** Engineered an end-to-end parameter-efficient fine-tuning (PEFT) pipeline using 4-bit NF4 double quantization and completion-only loss masking, reducing training VRAM by **61.7%** (16.2 GB $\to$ 6.2 GB) while training only **0.26%** of model parameters ($r=16, \alpha=32$).
> - **Domain Specialization & Metric Optimization:** Boosted structured JSON schema compliance from **23.3% to 96.7%** and slashed domain perplexity by **83.8%** (19.42 $\to$ 3.14) on complex multi-service production incidents and telemetry diagnosis.
> - **Inference Optimization & Edge Deployment:** Quantized fine-tuned weights into 4-bit GGUF (`Q4_K_M`), accelerating CPU/GPU inference throughput by **2.1x** (55.2 tok/s) and packaging automated `Modelfile` containers for zero-latency Ollama and vLLM serving.

---

### Option 2: High-Impact Bullets (For Standard Experience / Projects Section)
> - Designed and deployed a domain-specific SRE Diagnostic LLM using **QLoRA**, **Unsloth**, **TRL SFTTrainer**, and **BitsAndBytes**, achieving **78.9 ROUGE-L** and **64.3 BLEU** on root cause analysis.
> - Implemented custom prompt loss masking (`DataCollatorForCompletionOnlyLM`) with `paged_adamw_8bit` optimizer and cosine learning rate decay, preventing GPU out-of-memory surges and accelerating convergence across 300+ curated incident traces.
> - Built an interactive model evaluation arena in **FastAPI** featuring SSE token streaming, side-by-side zero-shot vs fine-tuned benchmarking, and real-time loss telemetry visualization.

---

## 🛠️ Key Skills & Keywords for ATS (Applicant Tracking Systems)

- **Techniques:** Parameter-Efficient Fine-Tuning (PEFT), QLoRA, LoRA Adapter Injection, 4-bit NormalFloat4 (NF4) Quantization, Double Quantization, Prompt Loss Masking, Gradient Checkpointing, Weight Merging (`merge_and_unload`).
- **Frameworks & Tools:** PyTorch, Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes`, `unsloth`, `accelerate`, `llama.cpp`, GGUF, Ollama, FastAPI.
- **Evaluation & Profiling:** Perplexity (PPL), ROUGE-1/2/L, SacreBLEU, JSON Schema Validation, VRAM Profiling, Token Latency (ms/tok) & Throughput (tok/sec).
