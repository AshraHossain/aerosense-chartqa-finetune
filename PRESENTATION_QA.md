# AeroSense ChartQA — Presentation Q&A

Thought process, design decisions, and step-by-step walkthrough for presenting this project.

---

## Part 1 — The Big Picture

**Q: What is this project in one sentence?**

A fine-tuning pipeline that teaches a small open-source LLM (Qwen2.5-3B) to accurately answer
questions about aeronautical charts and FAA regulations — fully automated from data generation
through training, evaluation, and local deployment, running on a MacBook Air.

---

**Q: Why did you choose the aviation domain?**

Three reasons that made it a stronger portfolio choice than a generic fine-tuning demo:

1. **Correctness actually matters.** Aviation has authoritative source documents (FAA AIM, FARs,
   approach plate specs). A wrong answer is measurably wrong, not just subjectively bad. This lets
   you evaluate quality rigorously.

2. **Domain specificity is real.** A general-purpose LLM handles simple aviation questions
   adequately but struggles with precise procedural details — the exact VFR minimums by airspace
   class, the difference between LNAV and LPV minimums, the specific meaning of each chart symbol.
   Fine-tuning has a clear job to do.

3. **Safety-critical framing adds depth.** Including a `safety_critical` category and measuring
   safety refusal rate as a first-class metric demonstrates production-grade thinking — not just
   "did the model answer correctly" but "did it refuse when it should."

---

**Q: Why Qwen2.5-3B-Instruct as the base model?**

- **Size is practical.** 3B parameters fits in 16GB unified memory with room to spare. You can
  train, run inference, and evaluate on the same machine without cloud costs.
- **Instruction-tuned base.** Starting from an already instruction-following model means the fine-
  tuning budget focuses on domain knowledge rather than teaching basic response formatting.
- **Strong baseline.** Qwen2.5 benchmarks well for its size — it understands nuanced questions
  even before fine-tuning, so we're measuring real domain uplift, not just basic comprehension.
- **HuggingFace-native.** Loads with `AutoModelForCausalLM`, works with standard PEFT and TRL
  without custom code.

---

**Q: Why generate synthetic data instead of scraping real data?**

Real aeronautical Q&A data that is cleanly formatted, diverse, and paired (question + correct
answer) essentially does not exist publicly. Options were:

1. **Manual curation** — extremely slow, requires aviation expertise to verify accuracy.
2. **Scrape forums / StackExchange Aviation** — noisy, inconsistent format, no ground truth labels.
3. **Synthetic generation via Claude** — structured prompts produce FAA-traceable Q&A pairs at
   scale, with consistent Alpaca format, difficulty tagging, and source references.

The key guardrail: every generated answer must include a `source_reference` field pointing to
a specific FAA AIM chapter or FAR part. This makes the data auditable and the model's outputs
traceable — which directly mirrors how certified avionics software is documented.

---

**Q: Why Claude for data generation and evaluation?**

For generation: Claude's system prompt adherence and JSON output reliability made it the right
tool for structured batch generation. It reliably returns valid JSON arrays without markdown
fences or hallucinated schema fields.

For evaluation (LLM-as-Judge): Using Claude as the judge creates an independent evaluator that
can reason about domain accuracy in free-text answers — something rule-based metrics (BLEU,
exact match) cannot do. It scores each response on three axes: domain accuracy (0-10),
hallucination (did it fabricate a regulation?), and safety (did it refuse when it should have?).

The judge is a different model evaluating a fine-tuned model's outputs — this avoids the model
grading its own work.

---

## Part 2 — Technical Decisions

**Q: What is LoRA and why use it instead of full fine-tuning?**

LoRA (Low-Rank Adaptation) freezes the base model's weights and inserts small trainable matrices
into the attention and MLP projection layers. Instead of updating 3.1 billion parameters, you
update ~30 million (0.96%).

Why this matters here:
- **Memory:** Full fine-tuning of Qwen2.5-3B in bfloat16 requires ~24GB just for weights +
  optimizer states. LoRA fits in 16GB with headroom.
- **Speed:** Fewer parameters to update = faster training steps.
- **The base model stays intact:** The frozen weights represent a lot of pre-training compute.
  LoRA adds domain knowledge without overwriting general reasoning ability.
- **Adapter portability:** The adapter is ~120MB vs ~6GB for a full model. Easier to version,
  distribute, and swap between base models.

---

**Q: What is QLoRA and how does it differ from LoRA?**

QLoRA (Quantized LoRA) applies the same adapter approach but loads the base model in 4-bit
NF4 (Normal Float 4) quantization instead of 16-bit. The adapter itself still trains in full
precision via a technique called double quantization.

Trade-offs:
- **Pro:** Roughly halves memory usage for the base model (~3.5GB vs ~7GB for the weights).
- **Con:** Quantization introduces noise that slightly degrades final model quality. We
  compensate by using a higher LoRA rank (r=64 vs r=16) to give the adapter more expressive
  capacity.
- **The comparison matters:** Running both LoRA and QLoRA on the same data and evaluating
  both gives a concrete memory/quality trade-off table — a realistic production decision point.

---

**Q: Why target all 7 projection layers (q/k/v/o/gate/up/down)?**

The original LoRA paper only targeted q_proj and v_proj. Later research (especially QLoRA paper
and subsequent work) found that including all linear layers — key, value, query, output
projection in attention, plus gate/up/down in the MLP feed-forward — consistently improves
performance. The additional parameters are still small relative to the base model.

---

**Q: Why PEFT + TRL instead of Unsloth on Apple Silicon?**

The original design used Unsloth, which provides optimized training kernels. However, Unsloth
on Apple Silicon loads models via MLX (Apple's ML framework) rather than PyTorch. MLX models
are not compatible with TRL's SFTTrainer, which is a pure PyTorch training loop.

The fix was to use the standard HuggingFace stack directly:
- `AutoModelForCausalLM` loads a standard PyTorch model
- `peft.LoraConfig + get_peft_model` applies LoRA adapters
- `trl.SFTTrainer` with `SFTConfig` runs training on MPS (Metal Performance Shaders)

This is actually the more portable path — it works identically on MPS, CPU, and CUDA without
backend-specific branches.

---

**Q: Why Alpaca format for the training data?**

Alpaca format (instruction / input / output) is the most widely adopted instruction-tuning
format. It maps cleanly onto how Qwen2.5-Instruct was itself fine-tuned, so the model already
understands how to respond to this structure. It also separates the question (`instruction`)
from optional context (`input`) and the answer (`output`), making evaluation straightforward.

---

**Q: Why bfloat16 and not float16?**

The M4 chip natively supports bfloat16. Compared to float16:
- Same memory footprint
- Larger dynamic range (same exponent bits as float32) → fewer NaN/overflow issues during
  training
- No loss scaling required
Float16 can silently produce NaN gradients when learning rates are slightly too high. bfloat16
is more numerically stable for training.

---

## Part 3 — File by File

**Q: What does `scripts/run_pipeline.py` do?**

It is the single entry point for the entire project. Running `--step lora` calls `step_lora()`,
which imports and runs `LoRATrainer`. Each step is a self-contained function. This design means:
- You can run any single step in isolation without understanding the rest
- The `--step all` path runs every function in order
- Logging separators make it clear where each step starts and ends in the output

---

**Q: What does `src/dataset/generator.py` do?**

Calls the Claude API in batches to generate aviation Q&A pairs. Key design choices:
- 6 categories × 10 topics × 4 difficulty levels = coverage matrix ensuring variety
- 5 examples per API call (batch_size=5) balances token efficiency vs retry cost
- Schema validation via `_is_valid()` — rejects any item missing required fields or with
  too-short answers, so bad API responses don't corrupt the training data
- Deduplication by exact instruction match before the train/eval split
- `source_reference` required in every example — makes the dataset auditable

---

**Q: What does `src/training/lora_trainer.py` do?**

Orchestrates the full LoRA fine-tuning loop:
1. Loads `Qwen/Qwen2.5-3B-Instruct` via `AutoModelForCausalLM` in bfloat16
2. Wraps it with `peft.LoraConfig` → `get_peft_model` to inject LoRA adapters
3. Formats the dataset as Alpaca prompts and tokenizes via SFTConfig
4. Runs `SFTTrainer.train()` with evaluation after each epoch
5. Saves the adapter weights (not the full model — much smaller)
6. `merge_and_export()` merges the adapter back into the base weights for deployment
7. `export_gguf()` converts the merged model to GGUF format for Ollama

---

**Q: What does `src/evaluation/judge.py` do?**

Takes a list of (question, model_response, reference_answer) triples and sends each to Claude
with a structured scoring prompt. Claude returns JSON with three scores per response. The
`compare_models()` method aggregates results across all three models into a comparison table.

This avoids human annotation for evaluation while still getting nuanced quality judgments that
BLEU score cannot capture — e.g., whether a model's answer about Class B airspace is technically
correct but missing the transponder requirement.

---

**Q: What does `src/inference/ollama_client.py` do?**

Wraps Ollama's REST API into a simple `generate(prompt) -> str` function that matches the
signature expected by `LLMJudge.evaluate_model`. This allows the evaluator to treat the base
Ollama model, the LoRA-loaded model, and the QLoRA-loaded model identically — same function
signature, different model names.

---

**Q: What does `scripts/push_to_hub.py` do?**

Reads the output directories from each config file, uploads adapter folders and GGUF files to
Hugging Face Hub using `HfApi.upload_folder`. Handles both variants (lora / qlora) in a loop
and gracefully warns rather than crashing if an artifact is missing (e.g., if QLoRA was skipped).

---

**Q: What are the config YAML files for?**

Separating hyperparameters from code means you can run ablations (change r from 16 to 32,
try a different learning rate) without touching Python. The config is also logged directly to
MLflow as a flat parameter dict, so every experiment run is reproducible from the YAML alone.

---

## Part 4 — Step by Step Execution

**Q: Walk me through exactly what happened to build this, in order.**

**Day 1 — Windows machine (project setup + data + baseline)**

1. Designed the 6-step pipeline architecture and created the full project scaffold:
   `scripts/`, `src/`, `configs/`, `tests/`, `.env.example`, `requirements.txt`, CI workflow.

2. Wrote `src/dataset/generator.py` with the 6 categories, batch generation loop, schema
   validation, and deduplication. Chose Alpaca format, added `source_reference` validation
   to enforce data traceability.

3. Ran `--step dataset` → generated 395 train + 50 eval examples (500 target, ~100 deduplicated
   out). Took ~15 minutes of Claude API calls.

4. Ran `--step baseline` → scored Qwen2.5-3B (base, no fine-tuning) via Ollama on the 50
   eval examples as the before-fine-tuning reference point.

**Between machines — data transfer**

5. The dataset files (`train.jsonl`, `eval.jsonl`) are gitignored (large, regenerable). To
   move them to the Mac without regenerating: `git add -f data/synthetic/*.jsonl` on Windows,
   pushed, pulled on Mac, then `git rm --cached` to remove them from tracking again.

**Mac (M4 MacBook Air) — training**

6. Set up SSH key for GitHub (`ssh-keygen -t ed25519`), added public key to GitHub account,
   pushed the project from Windows, cloned on Mac.

7. Created `.venv`, installed `requirements.txt`. Discovered Unsloth on Mac loads via MLX
   which is incompatible with TRL's PyTorch SFTTrainer. Rewrote `lora_trainer.py` to use
   standard `AutoModelForCausalLM + PEFT + SFTConfig` instead.

8. Fixed a cascade of TRL 1.4 API changes encountered during smoke test iterations:
   - `tokenizer` → `processing_class` in SFTTrainer
   - `TrainingArguments` → `SFTConfig`
   - `dataset_text_field`, `max_seq_length`, `packing` moved into `SFTConfig`
   - `max_seq_length` → `max_length` inside `SFTConfig`
   - `evaluation_strategy` → `eval_strategy`
   - `optim`: `adamw_8bit` → `adamw_torch` (bitsandbytes CUDA not available on MPS)
   - `model.parameters()` returning strings in MLX → fixed by removing param count code

9. Smoke test passed: 1 epoch, train loss 1.996, eval loss 1.769, 58.5% token accuracy.

10. Started full LoRA run (3 epochs, 75 steps, ~74s/step on M4 MPS) → running overnight.

**Still to come**

11. QLoRA run — same approach, 4-bit NF4. Will need to verify bitsandbytes MPS compatibility
    or adapt to run in cpu+bfloat16 mode.

12. Eval — `judge.py` sends all three models' responses to Claude, produces comparison table.

13. Deploy — `push_to_hub.py` uploads adapters to HF Hub, `export_gguf.py` converts to GGUF
    for Ollama, FastAPI endpoint and Streamlit demo ready to start.

---

## Part 5 — Anticipated Questions

**Q: Why not just use GPT-4 or Claude for this instead of fine-tuning a small model?**

Three reasons:
1. **Latency and cost at inference time.** A local 3B model answers in <1 second with zero API
   cost. An API call adds ~500ms and billing per query — unacceptable for a cockpit-assist tool.
2. **Data sovereignty.** Aviation operators cannot send flight plans and chart queries to third-
   party API endpoints. A local model runs in a secure environment.
3. **Fine-tuning demonstrates the engineering.** Using GPT-4 is a product decision. Fine-tuning
   a small model is a systems engineering exercise — it shows you understand the full stack from
   data quality through training stability to evaluation methodology.

---

**Q: How do you know the generated data is actually correct?**

Every example includes a `source_reference` field pointing to a specific FAA AIM section or
FAR part. The generation prompt instructs Claude to only produce answers traceable to public
FAA/ICAO documents and flags `safety_critical` examples for extra scrutiny. The evaluation
harness then cross-validates model responses against reference answers using Claude as judge —
so systematic factual errors in the training data would show up as low domain accuracy scores
in eval, not just get buried.

---

**Q: What would you do differently with more time?**

1. **Real FAA document RAG.** Chunk the actual FAA AIM, TERPS, and chart manuals into a vector
   store and use retrieval-augmented generation to ground both data synthesis and evaluation in
   authoritative source text rather than relying solely on Claude's knowledge of those documents.

2. **Human-in-the-loop validation.** Have a licensed pilot or instrument-rated CFI review a
   sample of the generated Q&A pairs before training. Even 50 validated examples would anchor
   the eval quality.

3. **Larger dataset.** 395 training examples is a minimal demonstration. 2,000–5,000 examples
   with balanced category coverage and adversarial edge cases would produce a more robust model.

4. **DPO/RLHF stage.** After SFT, add a preference-tuning stage where the model learns to
   prefer safer, more precisely cited answers over fluent but vague ones.

---

**Q: Is this model safe to use in a real aircraft cockpit?**

No — and that's explicitly stated. This is a research and portfolio demonstration. A
safety-certified avionics system requires DO-178C Level A software assurance, verified training
data with full traceability to approved source documents, and formal verification of outputs.
This project demonstrates the engineering methodology of applying those disciplines to LLM
fine-tuning — it is not a certified product.

---

**Q: What does the evaluation output actually look like?**

The judge returns a structured comparison table, for example:

```
model       domain_accuracy  hallucination_rate  safety_refusal_rate
base               4.2/10           0.31                 0.55
lora               7.6/10           0.12                 0.89
qlora              7.1/10           0.15                 0.86
```

And per-example JSON with the judge's reasoning for each score, stored in
`outputs/evaluation/` for audit review.
