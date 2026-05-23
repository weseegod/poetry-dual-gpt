# 🚀 v4.0 Roadmap — Break the Quality Ceiling

> v3 shipped: 100% stress test pass, zero BPE collapse, 80% rhyme, 97% tone.
> v4 targets the remaining gaps: rhyme quality, content diversity, and architectural ceiling.

---

## 📊 v3 Baseline (step 8800)

| Metric | v3 |
|--------|-----|
| Stress test (valid output) | 100% (14/14) |
| BPE collapse rate | 0% |
| Syllable count (6+8) | 93% |
| Rhyme accuracy (pos6) | 80% |
| Tone pattern (avg 4 pos) | 97% |
| Linebreak emission | 100% |
| Training data size | 541K pairs |
| Train/infer alignment | 100% (window-1, example-aligned) |

### What's still weak

| Issue | Current | Target |
|-------|---------|--------|
| Rhyme at pos 6 | 80% | 90%+ |
| Content quality | Generic, rural imagery dominant | Diverse, literary |
| Vocabulary | 12K BPE, limited | Richer vocabulary |
| Multi-couplet coherence | Independent duels only | Thematic flow across couplets |
| Single-line inputs | Sometimes 3 lines output | Always exactly 1 couplet |

---

## 🎯 v4 Priorities

### 🔴 P1: Beam Search for Rhyme Quality
**Impact: 80% → 90%+ rhyme | Effort: ~30 lines**

**Problem:** Current sampling (top-k + temperature + top-p) picks randomly among candidates. At position 6 of the 8-syl line, the model sometimes picks a word that doesn't match the target rhyme group.

**Fix:** At position 6, constrain candidates to only tokens whose rhyme group matches `[RHYME:X]`:

```python
# After top-k/top-p filtering, before softmax:
if position == 5:  # pos 6 (0-indexed) of 8-syl line
    for tid in top_candidates:
        word = tok.decode([tid])
        if get_rhyme_group(word.strip()) != target_rhyme:
            logits[:, tid] = float('-inf')
```

**Tradeoff:** Reduces diversity at one position. But rhyme is mandatory in Lục Bát — not optional.
**Risk:** None. Already at 80% naturally, beam just forces the last 10-20%.

---

### 🔴 P2: Expand Training Data with Canonical Poets
**Impact: Vocabulary + content quality | Effort: Scrape + preprocess (1 day)**

**Problem:** All 541K training pairs come from `poems_dataset_clean.csv` — mostly Truyện Kiều. The model's vocabulary and stylistic range is narrow. Output tends toward generic rural/romantic imagery.

**Fix:** Add 8 canonical poets:

| Poet | Style | Est. poems | Contribution |
|------|-------|-----------|-------------|
| Nguyễn Du | Truyện Kiều (full 3254 lines) | 1 | Deeper Kiều coverage |
| Hồ Xuân Hương | Humorous, double-entendre, feminist | ~50 | Wit, wordplay |
| Hàn Mặc Tử | Symbolist, surreal, religious | ~80 | Abstract imagery |
| Xuân Diệu | Romantic, passionate, modern | ~160 | Love, urgency, modern vocab |
| Huy Cận | Philosophical, cosmic | ~80 | Nature, existence |
| Nguyễn Bính | Folk, rural, nostalgic | ~80 | Rural life, simple beauty |
| Tố Hữu | Revolutionary, patriotic | ~100 | Historical, political vocab |
| Nguyễn Khuyến | Classical, nature, autumn | ~100 | Seasonal imagery, formality |

**Usage:**
```bash
pip install playwright && playwright install chromium
python data_service/scraper.py --all
python src/preprocess_doi_tho.py --csv data/poems_dataset_expanded.csv --window 1
```

**Expected:** 2-3× vocabulary diversity, fewer repetitive patterns, richer imagery.

---

### 🟡 P3: Fix Single-Line Overgeneration
**Impact: Eliminate last stress test failures | Effort: Post-processing**

**Problem:** Single-line input (6-syl only) sometimes generates 3 lines instead of 2. The model has no 8-syl line to condition on for the rhyme tag, so it's confused.

**Fix:** If output has > 2 lines, detect couplet boundaries and take first complete couplet:
```python
if len(lines) > 2:
    # Find first valid (6,8) pair
    for i in range(len(lines)-1):
        if len(lines[i].split()) == 6 and len(lines[i+1].split()) == 8:
            lines = [lines[i], lines[i+1]]
            break
```
Or: duplicate the single line as both 6-syl and 8-syl input (gives rhyme tag a value).

---

### 🟡 P4: True Multi-Couplet Training
**Impact: Thematic coherence across multiple couplets | Effort: New training format**

**Problem:** Current training is always `1 couplet → 1 couplet`. At inference, multi-couplet input runs independent duels (C1→C3, C2→C4). The model has no concept that C3 and C4 should form a coherent poem together.

**Fix:** Train on sequences of 2+ couplets with a `[CHAIN]` token:
```
<|start|> [DOI_THO] [RHYME:X] [TONE:XXXXXX] C1_6 <|linebreak|> C1_8 <|reply|>
  C2_6 <|linebreak|> C2_8 <|chain|>
  C3_6 <|linebreak|> C3_8 <|end|>
```
This teaches the model to generate 2 consecutive couplets that rhyme and flow together.

**Tradeoff:** Longer sequences = slower training. May need block_size increase.

---

### 🟢 P5: Scheduled Sampling
**Impact: Better inference quality | Effort: ~10 lines in train.py**

**Problem:** Teacher forcing gap — during training the model always sees perfect context. At inference it sees its own (potentially wrong) previous tokens.

**Fix:** During training, gradually mix in model-generated tokens:
```python
use_teacher_prob = max(0.5, 1.0 - step / total_steps * 0.5)
mask = torch.rand(x.shape) < use_teacher_prob
x_input = torch.where(mask, x_teacher, x_model_generated)
```

---

### 🟢 P6: Evaluation Dashboard
**Impact: Measure progress | Effort: 1 day**

Unified evaluation script with:
- Chain rhyme between couplets
- Per-position tone accuracy
- Lexical diversity (unique n-gram ratio)
- Semantic coherence (successive couplet similarity)
- Content quality rubric (human + automated)

---

### ⚪ P7: Qwen2.5-1.5B QLoRA — Deferred
**Impact: Ceiling-breaker | Effort: 1 day**

The 31M model has a hard quality ceiling. Qwen2.5-1.5B pretrained on terabytes of text would bring:
- Rich Vietnamese vocabulary (150K+ tokens)
- Grammatical correctness
- Cultural knowledge (folklore, idioms, history)
- Coherent multi-sentence generation

**When to do this:** After maximizing the 31M model (P1-P5 done, rhyme 90%+, content diverse). The architecture is ready — same training format, same control tokens, just swap the model.

---

## 📋 v4 Priority Order

| # | Item | Rhyme | Content | Diversity | Effort |
|---|------|-------|---------|-----------|--------|
| P1 | Beam search rhyme | +10% | — | — | 30 min |
| P2 | Expand data | — | +quality | +variety | 1 day |
| P3 | Fix single-line | — | +reliability | — | 15 min |
| P4 | Multi-couplet training | — | +coherence | — | 2-3 days |
| P5 | Scheduled sampling | +2% | +quality | — | 1 hour |
| P6 | Eval dashboard | — | Measurable | Measurable | 1 day |
| P7 | Qwen QLoRA | +ceiling | Major | Major | 1 day |

**Recommended sprint:** P1 → P3 → P2 (quick wins first, then data expansion)
