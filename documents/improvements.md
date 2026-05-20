# 🚀 Remaining Work

> Everything below is what's left. Done items have been removed to keep this focused.

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

## 🟡 Then: Rhyme Conditioning

> 📖 Full guide: **[rhyme_conditioning.md](rhyme_conditioning.md)**

**Goal:** Inject `[RHYME:en]` and `[TONE:BTB]` control tokens so the model learns to produce rhyming, tone-correct poetry.

**What you need to implement:**
- `src/tones.py` — tone classification + rhyme extraction utilities
- Update `preprocess.py` — inject control tokens into training pairs
- Update `server.py` + `sample.py` — auto-inject during generation
- Retrain tokenizer + retrain model

**Why this second:** Builds on the two-stage model. The control token format is model-agnostic — works for your 30M GPT and Qwen later.

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
