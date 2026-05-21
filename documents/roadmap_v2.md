# 🚀 v2.0 Roadmap

> v1.0 shipped: 31M params, 4 rules (rhyme 58%, tone 88%, syllable 78%, đối âm 69%), two-stage training, chat UI, Colab pipeline.
> Everything below is the path to v2.0.

---

## 🔮 v2.0: Qwen2.5-1.5B QLoRA

**Goal:** Swap the 31M custom GPT for a 1.5B pretrained Qwen model with LoRA fine-tuning. Same control tokens transfer directly — no format changes needed.

**Why:** A 31M model trained from scratch on 135K poems hits a quality ceiling. Qwen2.5-1.5B was pretrained on terabytes of text including Vietnamese. It already knows grammar, vocabulary, idioms, and cultural context. Fine-tuning on poetry data with control tokens just teaches it the rules — the foundation is already there.

**What Qwen brings:**
- Rich Vietnamese vocabulary (not limited to 11K BPE tokens)
- Grammatical correctness (no garbled subword fragments)
- Cultural knowledge (folklore, idioms, historical references)
- Coherent multi-sentence generation

**Implementation:**
```
Same training format → same preprocess.py → same control tokens
Just swap PoetryDuelGPT(31M) → Qwen2.5(1.5B) + LoRA adapters
```

**⚡ Base, not Instruct.** The base model (`Qwen/Qwen2.5-1.5B`) does pure next-token prediction — no chat template, no instruction format baked in. The Instruct variant (`Qwen/Qwen2.5-1.5B-Instruct`) is optimized for `<|im_start|>user`/`<|im_start|>assistant` chat format, which actively fights our `<|start|> [LUC_BAT] ... <|reply|>` format. Base model accepts any format; Instruct has been trained to *refuse* non-chat formats. Start with base, experiment with Instruct later.

**Requirements:** `transformers` + `peft` + `bitsandbytes`. Colab T4/L4 handles 4-bit QLoRA.

**Expected quality:** Semantic coherence from "medium" → "high". Rule accuracy similar or better. Poetry that reads like poetry, not just rule-compliant text.

---

## 📜 Multi-Couplet Generation

**Goal:** Generate full 4-line (Lục Bát) or 8-line (Thất Ngôn bát cú) poems, not just single couplets.

**Why:** Vietnamese poetry is structured in stanzas. A Lục Bát poem is 4+ lines (6-8-6-8...), and Thất Ngôn bát cú is 8 lines (7-7-7-7-7-7-7-7). Single couplet generation shows the model understands the form, but full-poem generation unlocks:
- **End rhyme** between couplets (currently skipped because we only generate one couplet)
- **Thematic coherence** across stanzas
- **Real poetic structure** that reads like actual Vietnamese poetry

**Implementation approach:**
1. Train on multi-couplet sequences (2-4 couplets per example)
2. Add `[COUPLET1]`, `[COUPLET2]` markers
3. Generate autoregressively until `[END_POEM]` token
4. This automatically enables end-rhyme evaluation between couplets

**Challenge:** Longer sequences (4 couplets ≈ 56 syllables ≈ ~60 tokens) fit in block_size=256, so no architecture change needed. Just data format change.

---

## 📚 Better Training Data

**Goal:** Expand beyond the current 135K-poem single-source corpus with poems from 8 canonical Vietnamese poets.

**Target authors** (scraped from isach.info via `data_service/scraper.py`):

| # | Author | Period | Poems | Source |
|---|--------|--------|-------|--------|
| 1 | **Nguyễn Du** | Trung đại | Truyện Kiều (3,254 lines) | isach.info |
| 2 | **Hồ Xuân Hương** | Trung đại | ~50 poems | isach.info |
| 3 | **Nguyễn Khuyến** | Trung đại | ~100 poems | isach.info |
| 4 | **Hàn Mặc Tử** | Hiện đại | ~80 poems | isach.info |
| 5 | **Tố Hữu** | Hiện đại | ~100 poems | isach.info |
| 6 | **Xuân Diệu** | Hiện đại | **160 poems** | isach.info ✅ |
| 7 | **Huy Cận** | Hiện đại | ~80 poems | isach.info |
| 8 | **Nguyễn Bính** | Hiện đại | ~80 poems | isach.info |

**Usage:**
```bash
pip install playwright && playwright install chromium
python data_service/scraper.py --all          # download all 8 authors
python data_service/scraper.py --author xuan_dieu  # single author
python data_service/scraper.py --dry-run      # preview without downloading
```

---

## 📊 v1.0 Baseline (for comparison)

| Metric | Stage 2 (v1.0) |
|--------|----------------|
| Rhyme (R1) | 58.4% |
| Tone (R2) | 87.5% |
| Syllable (R3) | 78.0% |
| Đối Âm (R4, Stage 1) | 69.4% |
| All 3 pass | 50.9% |
| Semantic quality | Medium |

Full evaluation: [rule_evaluation.md](rule_evaluation.md)

---

## 🎯 Priority

| # | Item | Impact | Effort | Blocks |
|---|------|--------|--------|--------|
| 1 | Qwen2.5-1.5B QLoRA | 🚀🚀🚀 | 1 day | Nothing — same data, same tokens |
| 2 | Multi-couplet generation | 🚀🚀 | 2-3 days | Enables end-rhyme rule |
| 3 | Better training data | 🚀 | Ongoing | Requires data collection |
| 4 | Scheduled sampling | 🚀🚀 | 1 hour | Reduces train/inference gap |
| 5 | Curriculum learning | 🚀 | 30 min | Train on short→long poems |

---

## 🎓 Scheduled Sampling

**Problem:** Teacher forcing gap — during training the model always sees perfect context, but at inference it sees its own (potentially wrong) previous tokens. This creates a gap: training loss looks good, but real generation quality is worse.

**Fix:** During training, gradually replace some ground-truth tokens with the model's own predictions. Early steps use 100% teacher forcing; later steps mix in model-generated tokens. The model learns to recover from its own mistakes.

**Implementation (~10 lines in train.py):**
```python
use_teacher_prob = max(0.5, 1.0 - step / total_steps * 0.5)  # decay 1.0→0.5
mask = torch.rand(x.shape, device=x.device) < use_teacher_prob
x_input = torch.where(mask, x_teacher, x_model_generated)
```

**Expected:** More robust generation, fewer empty/malformed outputs, higher rhyme/tone accuracy at inference.

---

## 📏 Curriculum Learning

**Problem:** Model trains on random poem lengths. Short poems (2 lines) and long poems (20+ lines) are mixed together. The model never learns progressive complexity.

**Fix:** Sort training data by poem length. Start with short poems (2-4 lines), gradually add longer ones. The model learns basic form first, then extends to longer sequences.

**Implementation:** Sort `poetry_corpus.txt` by token length, or group by number of couplets. Modify `PoetryDataset` to sample from progressively larger subsets.

**Expected:** Better form adherence on long sequences, smoother training curves.
