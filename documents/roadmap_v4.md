# 🚀 v4.0 Roadmap — Zero Retrain Wins

> v3 shipped: 100% stress test, 80% rhyme, 97% tone, 93% syllable.
> v4 delivers three inference-only improvements. No retrain. 1 hour total.

---

## 📊 v3 Baseline (step 8800)

| Metric | v3 |
|--------|-----|
| Stress test | 100% (14/14) |
| BPE collapse | 0% |
| Rhyme (pos6) | 80% |
| Tone (avg) | 97% |
| Syllable (6+8) | 93% |
| Linebreak | 100% |
| Thất Ngôn support | ❌ None (regression from v1) |

### Remaining issues

| Issue | Fix needed |
|-------|-----------|
| Rhyme at pos 6 | 80% → 90% with beam constraint |
| Single-line input | Sometimes 3 lines → always 1 couplet |
| Content diversity | Repetitive rural imagery |
| **Thất Ngôn (7-7)** | Training data dropped it (see below) |

---

## 🔴 Bug: Why Thất Ngôn Disappeared

**Root cause — `src/preprocess_doi_tho.py` line 156:**

```python
# Focus on Lục Bát for v2.0 (Thất Ngôn đối thơ is v3.0)
df = df[df["genre"] == "lục bát"]   # ← drops ALL bảy chữ poems
```

| Stage | Status |
|-------|--------|
| v1 `preprocess.py` | Generated `[LUC_BAT]` + `[THAT_NGON]` pairs from all genres |
| v2 `preprocess_doi_tho.py` | Lục Bát only. Comment: "Thất Ngôn is v3.0" |
| v3 | Thất Ngôn was never added. Training data = 541K `[DOI_THO]` lines, zero 7-syl |
| **Result** | Model has never seen a 7-syl line. Can't generate Thất Ngôn. |

`poems_dataset_clean.csv` still contains bảy chữ poems. The old `preprocess.py` still works. The model just never trained on them.

**Fix** (v5): Regenerate corpus with both genres, retrain. Two approaches:
- **A)** Extend `[DOI_THO]` for Thất Ngôn — add `[RHYME:Y]` (pos 7), `[TONE:YYYYYYY]` (7 tones), 7-7 couplet extraction
- **B)** Run old `preprocess.py` + `preprocess_doi_tho.py`, concatenate both corpora

**Effort:** Retrain (~3 hours Colab). → **v5 with data expansion.**

---

## ✅ v4 — Implement Now (Zero Retrain)

### P1: Beam Search for Rhyme (80% → 90%)
**Effort: 30 min | Impact: +10% rhyme | File: `src/sample.py`**

At position 6 of the response 8-syl line, constrain token candidates to only words matching `[RHYME:X]`:

```python
if current_position == 5:  # pos 6 (0-indexed) of 8-syl response line
    for token_id in valid_candidates:
        word = tokenizer.decode([token_id]).strip()
        if get_rhyme_group(word) != target_rhyme:
            logits[0, token_id] = float('-inf')
```

Only applies to 1 token per response. No diversity loss elsewhere.

---

### P2: Fix Single-Line Overgeneration
**Effort: 15 min | Impact: Eliminate last stress test failure | File: `src/sample.py`**

Single-line input (6 syl only, no 8-syl line) occasionally generates 3+ output lines instead of exactly 1 couplet. Add post-generation cleanup:

```python
if len(lines) > 2:
    # Find first valid (6,8) pair
    for i in range(len(lines) - 1):
        if len(lines[i].split()) == 6 and len(lines[i+1].split()) == 8:
            lines = lines[i:i+2]
            break
    else:
        # Fallback: take first 2 lines
        lines = lines[:2]
```

---

### P3: Scheduled Sampling (Teacher → Inference Gap)
**Effort: 1 hour | Impact: Better inference quality | File: `src/train.py`**

During training, gradually replace teacher-forced tokens with model-generated tokens:

```python
teacher_prob = max(0.5, 1.0 - step / total_steps * 0.5)
mask = torch.rand(x.shape) < teacher_prob
x_input = torch.where(mask, x_teacher, x_sampled)
```

Reduces the gap between training (perfect context) and inference (noisy context from previous generation).

---

### P4: Evaluation Dashboard
**Effort: 1 day | Impact: Metrics-driven development | File: `evaluate/`**

Unified script measuring across 50+ diverse prompts:
- Chain rhyme between couplets
- Per-position tone accuracy (B-T-B-B)
- Lexical diversity (unique n-gram ratio, type-token ratio)
- Semantic coherence (cosine similarity between successive couplets)
- Content quality rubric (human + automated heuristics)
- Per-genre breakdown (Lục Bát vs Thất Ngôn)

---

## 📊 v4 Summary

| # | Item | Time | Retrain? |
|---|------|------|----------|
| P1 | Beam rhyme | 30 min | No |
| P2 | Fix overgeneration | 15 min | No |
| P3 | Scheduled sampling | 1 hour | No |
| P4 | Eval dashboard | 1 day | No |

**Expected v4 results:** Rhyme 90%+, no overgeneration, measurable quality.

---

## 🔮 v5 — Retrain Bundle (Do These Together)

All v5 items require regenerating the training corpus and retraining. Batch them into one Colab run.

| # | Item | Impact | Effort |
|---|------|--------|--------|
| T1 | **Thất Ngôn support** — preprocess bảy chữ → `[DOI_THO]` pairs | Critical feature regression fix | Preprocess + retrain |
| T2 | **Data expansion** — 8 canonical poets (Hồ Xuân Hương, Hàn Mặc Tử, Xuân Diệu, Huy Cận, Nguyễn Bính, Tố Hữu, Nguyễn Khuyến, full Truyện Kiều) | Vocabulary + style diversity | Scrape + retrain |
| T3 | **Multi-couplet training** — `[CHAIN]` token for 2+ consecutive couplets | Thematic coherence | New format + retrain |

**v5 expected:** ~1.5-2M pairs (Lục Bát + Thất Ngôn, diverse sources), 85%+ rhyme, multi-couplet coherence.

---

## 🚫 Not Planned (Lower Priority)

| Item | Reason |
|------|--------|
| Qwen2.5-1.5B QLoRA | Deferred indefinitely — 31M model is sufficient after improvements |
| Curriculum learning | Marginal gains vs implementation cost |
| Tone contrast (đối âm) | 97% tone already, diminishing returns |
