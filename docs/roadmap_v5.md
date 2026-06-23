# 🚀 v5.0 Roadmap — Qwen2.5-1.5B Instruction Fine-Tuning

> **Status: IN PROGRESS** — Replacing failed flat-tensor control-token approach.
>
> v4.2.3 shipped: 31.5M custom GPT, 92% all-5-rules on Lục Bát, deployed on doitho.net.
> v5 Phase 1 (flat-tensor QLoRA): **FAILED** — val loss 17 vs train loss 0.6.
> **This roadmap** describes the replacement strategy.

---

## 🔴 Why v5 Flat-Tensor Approach Failed

Two training attempts on Qwen2.5-1.5B base model, both catastrophically overfit:

| Attempt | Approach | Train Loss | Val Loss | Root Cause |
|---------|----------|-----------|----------|------------|
| v5.0 | Flat tensor + 8.8K syllable tokens | 0.60 | 17.3 | `modules_to_save` made 490M params trainable |
| v5.1 | Removed syllable tokens, kept 214 control tokens | 0.81 | 16.6 | Flat tensor poisoning: 98% of 128-token windows cross poem boundaries |

### Root Cause Analysis

Three fatal design flaws, each sufficient to cause failure:

| # | Flaw | Why |
|---|------|-----|
| **1** | **Flat-tensor training** | 128-token windows span across poem boundaries. Model learns `đời <|end|> <|start|> [LUC_BAT]...` as valid transitions. During inference, these boundaries don't exist → distribution mismatch. |
| **2** | **Loss on all tokens** | Control tags (`[RHYME:ong]`, `[TONE:BBTBBT]`, etc.) get same loss weight as poetry content. Model wastes capacity learning to predict tokens it never generates. |
| **3** | **Alien format vs Qwen pretraining** | `<|start|> [LUC_BAT] [RHYME:X] [TONE:XX] [TRAMBONG:X] ...` means nothing to a model pretrained on natural text and chat. Qwen fights the format instead of leveraging its 1.5B parameters of prior knowledge. |
| **4** | **Vocab mismatch** | Adding 214 control tokens requires `modules_to_save` on embedding layers. Even this small addition creates embedding mismatch between frozen 4-bit Qwen embeddings and trainable new token embeddings. |

### Why v4.2.3's 31.5M Model Worked

The custom GPT was trained FROM SCRATCH on exactly this format. Every weight, every embedding was learned on poetry data. There was no "prior knowledge" to fight. The control-token approach is correct for a model that grows up with it — it's wrong for a pretrained model being fine-tuned.

---

## 🎯 v5.1: Instruction Fine-Tuning Strategy

**Core insight**: Qwen2.5-1.5B-Instruct already knows how to follow instructions. Don't fight it — instruct it.

### Old (broken) vs New (instruction)

```
OLD: <|start|> [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] [TRAMBONG:NH]
      thân em như chẽn lúa đòng <|reply|> phất phơ dưới ngọn nắng hồng ban mai <|end|>

NEW: <|im_start|>system
     Bạn là nhà thơ Lục Bát. Cho dòng Lục (6 chữ), viết dòng Bát (8 chữ) 
     tuân thủ: chữ thứ 6 Bát vần với chữ thứ 6 Lục (vần: ong); 
     thanh điệu dòng Bát: B-T-B-B; chữ thứ 6 và 8 khác dấu (Ngang≠Huyền, NH).
     Chỉ trả lời dòng Bát, không giải thích.<|im_end|>
     <|im_start|>user
     thân em như chẽn lúa đòng<|im_end|>
     <|im_start|>assistant
     phất phơ dưới ngọn nắng hồng ban mai<|im_end|>
```

### What Changes

| | Old (Flat-Tensor) | New (Instruction SFT) |
|---|---|---|
| **Model** | Qwen2.5-1.5B (base) | Qwen2.5-1.5B-Instruct |
| **Format** | Custom control tokens | Native chat template (`<|im_start|>`) |
| **Prompt** | `<\|start\|> [LUC_BAT] [RHYME:X]...` | System instruction + user query |
| **Response** | `<\|reply\|> poem <\|end\|>` | `assistant` turn with poem only |
| **Loss masking** | On all tokens | **Only on `assistant` tokens** |
| **Batching** | Flat tensor, cross-boundary | Example-aligned, one poem per row |
| **Vocab changes** | +214 control tokens | **Zero** new tokens |
| **Trainable params** | ~25M (with embedding mismatch) | ~24M (pure LoRA, no mismatch) |
| **Tokenization** | Custom BPE fragmentation | Qwen native, post-process BPE fragments |

### Why Instruction Fine-Tuning Solves All Four Flaws

| Flaw | How Instruction SFT Fixes It |
|------|------------------------------|
| Flat tensor | Each training example is a complete chat conversation. Padding separates examples. No cross-boundary noise. |
| Loss on all tokens | `labels` mask: `-100` for system and user turns. Loss only computed on `assistant` response. Zero wasted gradient. |
| Alien format | Chat template is Qwen's **native** format. The model was pretrained to follow this structure. Instructions leverage its 1.5B parameters of language understanding. |
| Vocab mismatch | No new tokens. All words come from Qwen's 151,936 vocabulary. Pure LoRA on attention layers. |

---

## 🏗️ Architecture

```
v4.2.3 (Current — deployed)          v5.1 (Instruction SFT)
──────────────────────────            ──────────────────────────
PoetryDuelGPT (31.5M)                 Qwen2.5-1.5B-Instruct
  ├── Custom BPE tokenizer             ├── Native Qwen tokenizer
  ├── Trained from scratch             ├── QLoRA fine-tuned
  ├── Control token format             ├── Chat template format
  └── CPU inference (~0.3s)            └── GPU inference (4-bit)

Data pipeline:
  poems_dataset_clean.csv              poems_dataset_clean.csv
    → preprocess_doi_tho.py              → preprocess_instruct.py
    → doi_tho_corpus.txt (flat lines)    → instruct_train.jsonl (chat format)
                                           → instruct_val.jsonl

Training:
  src/train.py (custom GPT)            src/train_instruct.py
    ├── PoetryDuelGPT                    ├── Qwen2.5-1.5B-Instruct
    ├── from scratch                     ├── QLoRA (r=16)
    ├── custom loss                      ├── SFT loss (masked)
    └── block_size=256                   └── max_length=256

Generation:
  src/generation.py                    src/generation_instruct.py
    ├── build_prompt()                   ├── build_chat_prompt()
    ├── control tokens                   ├── system + user messages
    └── decode_doi_tho()                 └── extract + clean response

Evaluation:
  evaluate/eval_rules.py               evaluate/eval_instruct.py
    ├── 5-rule Lục Bát                   ├── 5-rule Lục Bát (same logic)
    ├── quality metrics                  ├── quality metrics
    └── 116 couplet prompts              └── 116 couplet prompts
```

---

## 📋 Implementation Plan

### Phase 1 — Data Pipeline (1 hour)

**Goal**: Convert 618K Lục Bát couplets to chat-format JSONL.

**New file: `src/finetune/preprocess_instruct.py`**

```
Input:  corpus_luc_bat.txt (618K lines)
        <|start|> [LUC_BAT] [RHYME:ong] [TONE:TBTTBB] [TRAMBONG:HN]
          khóc than kể hết niềm tây <|reply|> chàng ôi! biết nỗi nước này cho chưa <|end|>

Output: instruct_train.jsonl (556K conversations) + instruct_val.jsonl (62K)
        {
          "messages": [
            {"role": "system", "content": "Bạn là nhà thơ Lục Bát..."},
            {"role": "user", "content": "khóc than kể hết niềm tây"},
            {"role": "assistant", "content": "chàng ôi! biết nỗi nước này cho chưa"}
          ]
        }
```

**Key design decisions**:
- **System prompt**: Contains rules (rhyme group, tone pattern, Trầm-Bổng). Extracted from the control tags in the original corpus.
- **User message**: Just the Lục line. Clean, minimal.
- **Assistant message**: Just the Bát line. Clean, minimal.
- **No control tokens in generation**: Model outputs pure Vietnamese poetry text.

**Train/val split**: 90/10 random split (shuffled), saved as separate JSONL files.

- [ ] Parse corpus_luc_bat.txt, extract fields from control tokens
- [ ] Build system prompts with varying instruction phrasing (5-10 templates)
- [ ] Generate train.jsonl + val.jsonl
- [ ] Spot-check 50 examples for correctness

### Phase 2 — Training (2-3 hours)

**Goal**: QLoRA fine-tune Qwen2.5-1.5B-Instruct on instruction-formatted poetry data.

**New file: `src/finetune/train_instruct.py`**

```python
# Key differences from failed flat-tensor approach:

# 1. Chat template tokenization
def tokenize_conversation(messages, tokenizer, max_length=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(text, truncation=True, max_length=max_length)
    # Create labels: -100 for system/user, keep for assistant
    input_ids = tokens["input_ids"]
    labels = input_ids.clone()
    # Mask everything before the assistant turn
    assistant_start = find_assistant_start(input_ids, tokenizer)
    labels[:assistant_start] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": tokens["attention_mask"]}

# 2. No new tokens — zero vocab changes
# (No tokenizer.add_tokens(), no resize_token_embeddings)

# 3. Example-aligned batching
# Each row = one complete conversation, padded to max_length
# HuggingFace DataCollatorForSeq2Seq handles padding + label masking

# 4. Standard SFT training
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=256,
    packing=False,  # No packing — each example is a row
)
```

**Training config**:
| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Method | QLoRA (4-bit NF4) |
| LoRA r/alpha | 16/32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| modules_to_save | **None** (zero embedding mismatch) |
| Batch size | 4 × 4 = 16 effective |
| Max length | 256 (chat template + poem fits easily) |
| Steps | 5,000 (~1.3 epochs over 556K examples) |
| LR | 2e-4, cosine to 1e-6, warmup 100 |
| Optimizer | AdamW, weight_decay=0.01 |
| GPU | RTX 3090 (~3h, ~$0.27) |

- [ ] Implement `train_instruct.py` with SFTTrainer
- [ ] Add train/val loss tracking
- [ ] Add checkpoint saving (best val loss)
- [ ] Test with 100-step smoke test on Colab/local
- [ ] Run full 5,000-step training on Salad

### Phase 3 — Evaluation (1-2 hours)

**Goal**: Generate 116 couplet completions and evaluate against v4.2.3 baseline.

**New file: `evaluate/eval_instruct.py`**

```python
def generate_with_chat(model, tokenizer, luc_line, rhyme, tone, trambong):
    system = build_system_prompt(rhyme, tone, trambong)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": luc_line},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        temperature=0.75,
        top_p=0.92,
        top_k=50,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Extract only the generated assistant response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], 
                                 skip_special_tokens=True)
    return response.strip()
```

**Evaluation protocol** (same 5 rules as v4.2.3):
| Rule | Description | Method |
|------|-------------|--------|
| R1 | Vần lưng | Rhyme group match (pos6 Lục vs pos6 Bát) |
| R2 | Bằng-Trắc | Tone pattern BTBB on output Bát |
| R3 | Syllable | Exact 8 syllables in output Bát |
| R4 | Trầm-Bổng | Dấu Ngang ≠ Huyền at pos6 vs pos8 |
| R5 | Nhịp điệu | Exact line length |

- [ ] Implement generation with chat template
- [ ] Add BPE fragment post-processing (`join_bpe_fragments`)
- [ ] Add optional soft rhyme constraint at inference
- [ ] Run evaluation on 116 couplet prompts
- [ ] Compare all metrics against v4.2.3 baseline

### Phase 4 — Integration & Deploy (1-2 hours)

- [ ] Integrate with `src/generation.py` (or create `src/generation_instruct.py`)
- [ ] Update `client/server.py` for chat-format generation
- [ ] Deploy to doitho.net or separate GPU endpoint
- [ ] Update HuggingFace model card
- [ ] Update README + CHANGELOG

---

## 📊 v5.1 Targets

### Phase 1 Target: Lục Bát Quality

| Metric | v4.2.3 (31.5M) | v5.1 Phase 1 Target | Notes |
|--------|---------------|---------------------|-------|
| R1 Rhyme (vần lưng) | 92% | ≥ 85% | May be lower without hard rhyme constraint |
| R2 Tone (BTBB) | 100% | ≥ 90% | Instruction should handle this naturally |
| R3 Syllable (exact 8) | 100% | ≥ 90% | Model must learn to count syllables |
| R4 Trầm-Bổng | 99% | ≥ 85% | Explicit in system prompt |
| R5 Nhịp điệu | 100% | ≥ 90% | Same as R3 |
| **All 5 pass** | 91% | ≥ 70% | Lower bar initially; v4.2.3 took 5 iterations |
| Lexical diversity | 0.96 | ≥ 0.85 | Qwen has richer vocab → should be good |
| BPE artifacts | 0% | < 5% | `join_bpe_fragments` post-processing handles |
| Human quality | 3.5/5 | ≥ 3.0/5 | Qwen's language quality should compensate |
| **Inference speed** | 0.3s (CPU) | 1-2s (GPU 4-bit) | Trade-off for quality |

### Phase 1 SUCCESS GATE

```
✅ PASS if:
  - Train/val loss gap < 1.0 (no overfitting)
  - All-5-pass ≥ 70% on 116 couplet prompts
  - Human quality ≥ 3.0/5 on 20 blind samples
  - Zero BPE artifacts visible in human review (< 1/20)
  - Generation produces valid Vietnamese (no gibberish)

❌ FAIL if:
  - Val loss diverges from train loss (> 3.0 gap)
  - BPE artifacts visible in > 5/20 samples
  - Empty/short responses > 10%
  - Human quality < 2.5/5
```

### Phase 2: Multi-Genre (future)

Once Lục Bát quality is solid:
- Add Thất Ngôn (7/7) via genre-specific system prompts
- Add poet style conditioning (Nguyễn Bính, Hồ Xuân Hương, etc.)
- Combined training: 748K pairs, same instruction format

### Phase 3: Quality Polish (future)

- Soft rhyme constraint at inference (logit bias for matching rhyme group)
- Generate-and-rerank (N=3 candidates, score by rule compliance)
- Beam search for rhyme quality
- Scheduled sampling (if needed)

---

## 📁 Files Changed / Created

### New Files
| File | Purpose |
|------|---------|
| `src/finetune/preprocess_instruct.py` | Convert corpus_luc_bat.txt → chat-format JSONL |
| `src/finetune/train_instruct.py` | SFT with SFTTrainer, proper loss masking |
| `evaluate/eval_instruct.py` | Generate + evaluate using chat template |
| `data/instruct_train.jsonl` | Training data (556K examples) |
| `data/instruct_val.jsonl` | Validation data (62K examples) |

### Deprecated Files (from failed v5.0 approach)
| File | Reason |
|------|--------|
| `src/finetune/train.py` | Flat-tensor approach — replaced by train_instruct.py |
| `src/finetune/preprocess_syllables.py` | Syllable tokens — not needed for instruct approach |
| `evaluate/eval_qwen.py` | Control-token generation — replaced by eval_instruct.py |
| `src/finetune/corpus/syllable_vocab.json` | Unused |
| `checkpoints/qwen_stage1_*` | Failed training runs |

### Preserved Files (v4.2.3 deployment)
| File | Purpose |
|------|---------|
| `src/model.py` | v4.2.3 model — kept for doitho.net deployment |
| `src/train.py` | v4.2.3 training — kept for reproducibility |
| `src/generation.py` | v4.2.3 generation — kept for deployment |
| `checkpoints/doi_tho_best.pt` | v4.2.3 deployed checkpoint |
| `evaluate/eval_rules.py` | v4.2.3 evaluation suite |

---

## 💵 Cost Estimate

| Phase | GPU | Time | Cost |
|-------|-----|------|------|
| Smoke test (100 steps) | Colab T4 / local | ~15 min | Free |
| Full train (5,000 steps) | Salad RTX 3090 | ~3h | **~$0.27** |
| Eval (116 prompts) | Colab T4 | ~10 min | Free |
| Retry (if needed) | Salad RTX 3090 | ~3h | $0.27 |
| **Total** | | ~6h GPU | **~$0.54** |

---

## ⚠️ Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Qwen-Instruct refuses poetry task | Low | Instruct variant follows instructions well. If it refuses, fall back to base model with chat template. |
| Syllable counting is hard for Qwen | Medium | v4.2.3 model also struggled initially. Multiple templates in system prompt help. Can add syllable-count post-correction. |
| Rhyme quality lower than v4.2.3 | Medium | v4.2.3 had beam rhyme masking at inference. Can add as post-generation filter (Phase 3). |
| GPU inference too slow for real-time | Medium | 4-bit Qwen on T4 ~1-2s/request. Acceptable for MVP. Batch with vLLM for production. |
| System prompt too long, wastes context | Low | ~100 tokens. Max length 256 leaves 156 tokens for poem (~11 syllables). Generous. |

---

## 🎓 Design Principles

1. **Leverage pretraining, don't fight it.** Qwen knows how to chat — use chat.
2. **Loss only on what matters.** Mask everything except the assistant response.
3. **Example-aligned batching.** One poem per row, padded. No cross-boundary noise.
4. **No new tokens.** Pure LoRA on attention. Zero embedding mismatch.
5. **Metrics match human experience.** 5-rule + quality + blind review.
6. **Gate every phase.** Don't proceed until the current phase passes.
7. **Keep v4.2.3 deployed.** It works. v5.1 is additive, not replacement.
8. **Soft constraints at inference.** Bias, don't force. Fall back when uncertain.

---

## 📚 References

- Qwen2.5-1.5B-Instruct: [huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Chat template: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
- SFTTrainer: [huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)
- QLoRA paper: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- v4.2.3 roadmap: [roadmap_v4.2.md](roadmap_v4.2.md)
- 5 Lục Bát rules: [rule_evaluation.md](rule_evaluation.md)
