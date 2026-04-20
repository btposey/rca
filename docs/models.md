# Model Documentation

## Rubric Summary

This project trains two LoRA-fine-tuned models: Llama 3.2 1B (Dispatcher, structured intent extraction) and Llama 3.2 3B (Concierge, persona-adaptive narrative generation). Both are evaluated against held-out datasets using task-appropriate metrics. This section documents model selection rationale, fine-tuning methodology, dataset generation, training configuration, and evaluation.

---

## Model Selection Rationale

### Stage 1: Dispatcher — Llama 3.2 1B Instruct

**Why 1B?**
The Dispatcher performs structured extraction (intent classification + JSON output), not open-ended generation. The task is well-defined: map a short user string to a fixed 4-field JSON schema. A 1B parameter model is sufficient for this pattern-recognition task after fine-tuning, and fits within the RTX 3060's 12 GB VRAM alongside the Concierge model at 4-bit quantization.

**Why fine-tuned (not base)?**
The base Llama 3.2 1B does not reliably produce the specific JSON schema required (with `search_predicate.min_tier` inference, `attack` classification, and persona detection). Fine-tuning on ~1000 distilled examples teaches this mapping directly.

**Why Llama 3.2 specifically?**
- Llama 3.2 1B/3B are the smallest publicly available Llama 4 generation models with strong instruction following
- Native support in Unsloth for QLoRA fine-tuning
- OpenAI-compatible serving via vLLM with guided JSON / constrained decoding

### Stage 3: Concierge — Llama 3.2 3B Instruct

**Why 3B?**
The Concierge generates persona-adaptive narrative prose (2–6 sentences). This requires more expressive capacity than a 1B model provides, but does not justify the VRAM cost of a 7B+ model on the target hardware. The 3B pre-trained Instruct model produces acceptable persona differentiation with appropriate prompting.

**Fine-tuned for hallucination grounding:**
The pre-trained 3B model was found to hallucinate restaurants not in the retrieved candidate set. The Concierge was fine-tuned on 552 pipeline-driven samples — each sample uses real Dispatcher + Librarian outputs as input context, with Claude (Opus via Bedrock) generating the gold response. This "pipeline-driven" approach ensures the training distribution exactly matches inference: the model learns to generate recommendations grounded only in the provided candidate list. Post fine-tune pipeline evaluation shows 100% no-hallucination rate. Training pipeline: `scripts/train_concierge.py`.

---

## Fine-Tuning Methodology

### Approach: QLoRA via Unsloth

Fine-tuning uses **QLoRA** (Quantized Low-Rank Adaptation), which:
1. Loads the base model in 4-bit NF4 quantization to reduce VRAM requirements
2. Attaches small trainable LoRA adapter matrices to the attention and MLP projection layers
3. Trains only the adapter parameters (~1–2% of total weights) while the base model weights remain frozen

**Framework:** [Unsloth](https://github.com/unslothai/unsloth) — provides optimized CUDA kernels for QLoRA training that are 2–3x faster than stock HuggingFace PEFT and use less VRAM. Critical for training on an RTX 3060 (12 GB).

**Post-training:** The LoRA adapter is merged back into the base weights using `save_pretrained_merged(..., save_method="merged_16bit")` and the merged model is served via vLLM. This avoids LoRA inference overhead at serving time.

**Target modules** (all attention + MLP projections):
```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Dataset Generation: Teacher-Student Distillation

Training data is generated via **teacher-student distillation**:

1. **Teacher model**: Claude (via AWS Bedrock) — `claude-opus-4-5` for complex samples, `claude-haiku-4-5` for attack mutations
2. **Instructor library**: Enforces Pydantic schema compliance on teacher outputs, guaranteeing every generated sample conforms to `DispatcherOutput`
3. **Student model**: Llama 3.2 1B — learns to replicate the teacher's structured outputs

**Generation script**: `scripts/generate_dataset_bedrock.py`

**Dataset composition** (`data/synthetic/train.jsonl` — 1000 samples):
- ~800 normal restaurant queries: varied cuisine, price sensitivity, vibe requests, persona signals
  - ~200 of these use foodie vocabulary (culinary terms, technique language)
  - ~600 use normie/casual phrasing
- ~200 prompt injection attacks: jailbreak patterns, instruction overrides, role confusion
  - Attack mutations include: case changes, wrapper prefixes, unicode substitutions
  - All map to `attack: true`, `semantic_query: null`

**Held-out eval set** (`data/synthetic/eval.jsonl` — 100 samples):
- Entirely separate from training data (not seen during fine-tuning)
- Normal queries only (attacks are evaluated in aggregate accuracy, not separately)
- Used exclusively by `scripts/evaluate.py`

**Sample format:**
```json
{
  "input": "I want a proper omakase, money is no object",
  "output": {
    "persona": "foodie",
    "attack": false,
    "search_predicate": {"cuisine": "japanese", "min_tier": 4},
    "semantic_query": "omakase chef's tasting menu premium japanese"
  }
}
```

---

## Training Configuration

All parameters are exposed as CLI arguments in `scripts/finetune.py` with the following defaults:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` | Smallest viable instruction model |
| Epochs | 3 | Sufficient for structured extraction task; more risks overfitting on 1000 samples |
| Per-device batch size | 4 | RTX 3060 VRAM constraint |
| Gradient accumulation steps | 4 | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard QLoRA learning rate |
| Max sequence length | 512 | Covers longest training samples with padding |
| LoRA rank (`r`) | 16 | Balance between capacity and parameter efficiency |
| LoRA alpha | 32 | `alpha = 2 * r` is standard scaling convention |
| LoRA dropout | 0.05 | Light regularization |
| Precision | bf16 | RTX 3060 Ampere architecture supports bf16; avoids fp16 overflow on large gradients |
| Optimizer | AdamW (default) | HuggingFace default via TrainingArguments |
| Scheduler | Linear (default) | Default TRL/HuggingFace scheduler |
| Save strategy | per epoch | Checkpoints after each epoch |
| 4-bit base loading | `load_in_4bit=True` | NF4 quantization via bitsandbytes during training |

**Training invocation:**
```bash
uv run scripts/finetune.py \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --train-file data/synthetic/train.jsonl \
  --epochs 3 \
  --batch-size 4 \
  --lora-r 16
```

**Orchestration** (`scripts/train_dispatcher.py`): Full pipeline — generate dataset → rsync to GPU machine → run `finetune.py` → run `evaluate.py` → report results.

---

## Evaluation Metrics

Evaluation is implemented in `scripts/evaluate.py`. The fine-tuned model is served via vLLM and queried on 100 held-out samples from `data/synthetic/eval.jsonl`.

### Dispatcher Metrics (Field-Level)

The Dispatcher is a structured extraction task. Metrics are computed field-by-field rather than as a single aggregate score:

| Metric | Type | Description |
|--------|------|-------------|
| `persona_accuracy` | Exact match (3-class) | Fraction of samples where predicted persona matches ground truth (`foodie`/`normie`/`neutral`) |
| `attack_accuracy` | Exact match (binary) | Fraction of samples where attack flag matches ground truth |
| `cuisine_precision` | Exact match on non-null labels | Precision of cuisine extraction; only evaluated on samples where ground truth has a cuisine value |
| `price_mae` | Mean absolute error | Average dollar error on `max_price` predictions; only evaluated on samples with a ground truth price |

**Why field-level rather than full-JSON match?**
Exact JSON match is brittle — it penalizes semantically equivalent outputs (e.g., `"italian"` vs. `"Italian"`). Field-level scoring allows partial credit and identifies which extraction dimensions are weak.

### Concierge Metrics (Generative, if fine-tuned)

If the Concierge is fine-tuned, evaluation uses:
- Persona adherence: human judgment on whether the vocabulary matches the requested persona
- Hallucination rate: fraction of recommendations that name a restaurant not in the candidate list
- Format compliance: fraction of responses that parse as valid JSON with `suggestion` and `elaboration` keys

### Running Evaluation

```bash
# Ensure vLLM is serving the fine-tuned Dispatcher model, then:
uv run scripts/evaluate.py \
  --model-name dispatcher-llama-1b \
  --eval-file data/synthetic/eval.jsonl

# Results are written to:
# data/synthetic/eval_results_dispatcher-llama-1b.json
```

---

## Known Limitations

1. **Synthetic training data**: Both training samples and restaurant records are generated by Claude (via Bedrock). The model has not been validated on real-world user queries. Distribution shift between synthetic training data and real queries may reduce performance.

2. **No geospatial filtering**: Location filtering is explicitly descoped. The system cannot filter restaurants by proximity — all 200 restaurants are treated as candidates regardless of user location.

3. **Tier inference subjectivity**: The Dispatcher infers `min_tier` from soft language cues. Edge cases (e.g., "a really good but cheap place") may produce inconsistent tier assignments.

4. **Dispatcher false positive attacks**: The v1 Dispatcher over-classifies benign queries as attacks (precision 0.294 in pipeline evaluation). This is a training data distribution issue — attack samples were deterministic surface mutations of 10 seeds, leading the model to flag any query outside its training distribution. A v2 dataset with LLM-generated subtle attacks was produced but the fine-tune regressed on JSON output quality; v3 training is the remediation path. See `docs/evaluation_report.md` for full analysis.

5. **Single-GPU constraint**: The 4-bit quantized models share an RTX 3060 (12 GB). The `gpu_memory_utilization` split (0.40 Dispatcher / 0.50 Concierge) is tuned for this hardware. GCP deployment with an L4/A100 removes the quantization requirement and is expected to improve model quality. See `docs/evaluation_report.md` for the GCP comparison methodology.

6. **Concierge fine-tuning completed**: The Concierge was fine-tuned on 552 pipeline-driven samples (real Dispatcher + Librarian outputs used as training inputs, Claude-generated gold responses). Pipeline evaluation shows 100% no-hallucination rate post fine-tuning. Persona adherence is 0.783 — the primary remaining quality gap.

7. **Price extraction gaps**: The Dispatcher rarely predicts `max_price` even when price signals are present in the query ("under $20", "nothing over sixty dollars"). The training data had insufficient price-labelled examples. Price MAE metric is effectively N/A in current evaluation.
