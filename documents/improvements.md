# 🚀 Remaining Work

> Everything below is what's left. Done items have been removed to keep this focused.

---

## 🐛 CRITICAL BUG: Tokenizer Corruption from Rhyme/Tone Tokens

> **Date:** 2026-05-20 | **Diagnosed and fixed**

### What happened

Rhyme/tone control tokens (`[RHYME:ong]`, `[TONE:BBBTTB]`, `[LINK2:B]`) were included as **plain text** in the training corpus. When BPE tokenizer was retrained on this corpus, the tokens got split into 5-6 subword pieces:

```
[LUC_BAT]    → 1 token (id=4)  ✅ special token
[RHYME:ong]  → 5 tokens         ❌ BPE-split: [, RHYME, :, ong, ]
[TONE:BBBTTB] → 5 tokens        ❌ BPE-split
```

### Diagnosis: model IS working, sample.py missing auto_tag

Tests confirmed the model generates coherent Vietnamese poetry when given properly tagged prompts:

```
Prompt:  [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng
Output:  "để anh ngơ ngẩn đứng trông ngóng nhìn em"
```

The broken output came from plain-text prompts without genre tags. `sample.py` batch mode doesn't call `auto_tag()`, unlike interactive mode which does. Fixed: batch mode now auto-tags.

### Remaining concern: fragmented control tokens

Even though the model works, `[RHYME:ong]` being 5 BPE tokens is suboptimal. The model has to assemble rhyme meaning from 5 positions. For stronger rhyme control, these should be special tokens (single IDs). Left as future improvement.

### Also fixed: sample.py batch mode auto_tag

```python
# Before: raw prompt passed to generate
_, ids = generate(model, tok, args.prompt, ...)

# After: auto-detect and tag
prompt = args.prompt
if not prompt.startswith('['): prompt = auto_tag(prompt)
if '[LUC_BAT]' in prompt and '[RHYME:' not in prompt:
    prompt = auto_tag(prompt.replace('[LUC_BAT] ', ''))
_, ids = generate(model, tok, prompt, ...)

---

## 🔴 Next: Two-Stage Training

> 📖 Full guide: **[two_stage_training.md](two_stage_training.md)**

**Goal:** Train on all genres first (136K poems), then fine-tune on Lục Bát only.

**What you need to implement:**
- `--resume` flag in `train.py` (load checkpoint + optimizer state, continue from saved step)
- Filter Lục Bát-only corpus from the existing 942K pairs
- Run Stage 1 (15K steps, all genres) → then Stage 2 (5K steps, Lục Bát only, lower LR)

**Why this first:** Establishes your 30M model's quality ceiling. No new data formats, no new tokens — just training strategy.

---

## 🟡 Then: Rhyme Conditioning ✅

> 📖 Full guide: **[rhyme_conditioning.md](rhyme_conditioning.md)**

**Implemented:** `src/tones.py`, corpus regenerated with `[RHYME:en] [TONE:BBBTTB]`, tokenizer retrained.

**Remaining:** Train model, evaluate rhyme/tone improvement after training.

---

## 🔵 Future: Qwen2.5-1.5B Fine-Tune

**Goal:** Swap your 30M GPT for a 1.5B pretrained model with LoRA fine-tuning.

**Why this last:** Your 30M model will hit a quality wall (correct form, simple vocabulary). Qwen brings rich Vietnamese understanding from its pretraining. The same rhyme/tone control tokens transfer directly.

**Requirements:** HuggingFace `transformers` + `peft` + `bitsandbytes`. Colab L4 can handle it with 4-bit QLoRA.

---

## 📊 Done (for reference)

| What | Status |
|------|--------|
| Pad token loop fix | ✅ |
| True couplets (step=2) | ✅ |
| Top-p nucleus sampling | ✅ |
| 30M model (n_embd=512, n_head=8, n_layer=8) | ✅ |
| Comma-free prompt format | ✅ |
| Data cleaning pipeline | ✅ |
| Multi-genre ([LUC_BAT] + [THAT_NGON]) | ✅ |
| Dual-genre corpus (942K pairs) | ✅ |
| Retrained tokenizer (10,785 vocab) | ✅ |
| Chat UI (FastAPI + React) | ✅ |
| Auto-detect genre (6-syl→LUC_BAT, 7-syl→THAT_NGON) | ✅ |
| Save best.pt on val improvement | ✅ |

---

## 🧪 Evaluation Checklist

After each remaining step, generate 5 samples:

```bash
python src/sample.py --num_samples 5 --temperature 0.75 --top_p 0.92
```

| Metric | Current | After two-stage | After rhyme |
|--------|---------|-----------------|-------------|
| 6→8 syllable accuracy | ~85% | 95%+ | 95%+ |
| B-T-B-B tone correctness | ~60% | 80%+ | 90%+ |
| Rhyme (6th-syl match) | ~30% | ~30% | 60%+ |
| Semantic coherence | Medium | Medium-High | High |
| No empty output | ✅ | ✅ | ✅ |
