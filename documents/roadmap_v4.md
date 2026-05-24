# 🚀 v4.0 Roadmap — Maximum Existing Data

> v3 shipped: 100% stress test, 80% rhyme, 97% tone, Lục Bát only.
> v4 adds Thất Ngôn + Multi-couplet from EXISTING data (125K poems in CSV).
> One Colab retrain (~3-4h). v5 = new data scraping only.

---

## 📊 v3 Baseline

| Metric | v3 |
|--------|-----|
| Stress test | 100% |
| Rhyme (pos6) | 80% |
| Tone (avg) | 97% |
| Syllable (6+8) | 93% |
| Thất Ngôn (7→7) | ❌ Dropped by line 156 |
| Multi-couplet (2+ couplets) | ❌ Window=1 only |
| Genres used | 1/2 in CSV |

---

## 🗂️ What's Already in the CSV (No Scraping Needed)

```
poems_dataset_clean.csv — 125,892 poems
├── Lục Bát:         84,538  poems  ── 541K pairs generated (window=1)
│   ├── 3+ couplets: 74,430  poems  ── ~148K multi-couplet pairs (window=2)
│   └── 4+ couplets: 70,514  poems  ── ~200K+ multi-couplet pairs (window=2+3)
│
└── Bảy Chữ:         41,354  poems  ── currently FILTERED OUT
    ├── 2+ couplets: 40,699  poems  ── ~80K window=1 pairs
    ├── 3+ couplets: 34,063  poems  ── ~68K window=2 pairs
    ├── Thất ngôn bát cú:  1,554 poems (8 lines = 4 couplets)
    ├── Thất ngôn tứ tuyệt:  712 poems (4 lines = 2 couplets)
    └── Song thất lục bát:  371 poems (7-7-6-8 → split into TN + LB)
```

**Total v4 corpus potential:** ~541K (existing LB w=1) + ~80K (TN w=1) + ~216K (multi-couplet LB+TN w=2) ≈ **837K pairs** — all from the CSV we already have.

---

## 🔴 Bug: Thất Ngôn Dropped

**`src/preprocess_doi_tho.py` line 156:**

```python
# Focus on Lục Bát for v2.0 (Thất Ngôn đối thơ is v3.0)
df = df[df["genre"] == "lục bát"]   # drops 41,354 bảy chữ poems
```

The comment promised v3.0. v3 shipped without it. The data was there the whole time.

---

## ✅ v4 Plan — One Retrain, All Gains

### T1: Thất Ngôn Support (7→7)
**Effort: ~50 lines in preprocess_doi_tho.py | Retrain: yes**

Add bảy chữ handling alongside lục bát:

```python
# Currently: df = df[df["genre"] == "lục bát"]
# Fix: handle both genres

if genre == "lục bát":
    syl_target = (6, 8)
    rhyme_pos = 7     # pos 8 (0-indexed)
    tone_len = 6
elif genre == "bảy chữ":
    syl_target = (7, 7)
    rhyme_pos = 6     # pos 7 (0-indexed)
    tone_len = 7
```

**Thất Ngôn rules:**
| Rule | Lục Bát (6→8) | Thất Ngôn (7→7) |
|------|---------------|-----------------|
| Syllables | 6 + 8 | 7 + 7 |
| Rhyme source | Pos 8 of 8-syl line | Pos 7 of input 7-syl line |
| Tone pattern | `[TONE:BBBBBB]` (6 chars) | `[TONE:BBBBBBB]` (7 chars) |
| Format tag | `[DOI_THO]` | `[DOI_THO:TN]` or `[DOI_THO]` |

**Genre tag:** Add `[DOI_THO:TN]` token so model knows it's generating 7-syl lines. Without genre awareness, the model can't distinguish Lục Bát from Thất Ngôn responses since both use the same `[RHYME:X] [TONE:...]` prefix. The tone length difference (6 vs 7) provides a weak signal, but an explicit tag is safer.

Alternatively, use `[DOI_THO]` for both and let the input line count (6 vs 7 tokens between `<|linebreak|>`) be the implicit signal. Simpler but risks mode confusion.

**Recommendation:** `[DOI_THO:TN]` — explicit, one new special token, model learns clean separation.

---

### T2: Multi-Couplet Training (2+ Couplets)
**Effort: ~30 lines in preprocess_doi_tho.py | Retrain: yes**

**Problem:** Current training is window=1 only. Model has no concept of generating consecutive couplets that form a poem together. Inference does independent duels (C1→C3, C2→C4) with no coupling between responses.

**Fix:** Add window=2 pairs: couplet_k + couplet_{k+1} → couplet_{k+2}

```
<|start|> [DOI_THO] [RHYME:ong] [TONE:BBBBBB]
  C1_6 <|linebreak|> C1_8 <|linebreak|> C2_6 <|linebreak|> C2_8
  <|reply|>
  C3_6 <|linebreak|> C3_8
  <|end|>
```

This teaches the model that:
- 2 input couplets → 1 output couplet (just like 1→1, but with more context)
- Chain rhyme tag still comes from LAST input couplet (C2)
- The response is still ONE couplet, not two — keeps format consistent

**Data available:** 74K Lục Bát + 34K Thất Ngôn poems with 3+ couplets ≈ **108K window=2 pairs**. This interleaves naturally with window=1 data — same format, different input depth.

**Inference behavior:** If user sends 2 couplets, model generates 1 response couplet (just like now). The difference is the model has learned to use BOTH input couplets for context, not just the last one.

**Optional: window=3:** 70K Lục Bát poems with 4+ couplets could generate ~140K pairs. But window=3 means 3 input couplets → 1 output. Rarely used at inference. Defer to v5.

---

### P1: Beam Rhyme Constraint
**Effort: ~20 lines | Retrain: no | Rhyme: 80% → 90%+**

At position 6 of the response 8-syl line (or pos 7 for Thất Ngôn), force tokens to match `[RHYME:X]`:

```python
if current_pos == rhyme_target_position:
    for tid in top_k_candidates:
        word = tok.decode([tid]).strip()
        if get_rhyme_group(word) != target_rhyme:
            logits[:, tid] = float('-inf')
```

One token per response. Zero diversity impact elsewhere.

---

### P2: Fix Single-Line Overgeneration
**Effort: ~10 lines | Retrain: no**

Post-generation cleanup for single-line input (no 8-syl companion):

```python
if len(lines) > 2:
    for i in range(len(lines) - 1):
        if len(lines[i].split()) == s6 and len(lines[i+1].split()) == s8:
            lines = lines[i:i+2]
            break
    else:
        lines = lines[:2]
```

---

### P3: Scheduled Sampling
**Effort: ~15 lines in train.py | Retrain: yes (bundled with T1+T2)**

Reduce teacher-forcing gap during training:

```python
teacher_prob = max(0.5, 1.0 - step / total_steps * 0.5)
```

---

### P4: Unified Eval Dashboard
**Effort: 1 day | Retrain: no**

Single script covering 50+ prompts across both genres:
- Per-genre breakdown (Lục Bát vs Thất Ngôn)
- Chain rhyme, tone per position, syllable accuracy
- Lexical diversity (type-token ratio, unique n-grams)
- Multi-couplet coherence (cosine similarity between successive responses)

---

## 📊 v4 Summary

| # | Item | Lines | Retrain |
|---|------|-------|---------|
| P1 | Beam rhyme constraint | ~20 | No |
| P2 | Fix overgeneration | ~10 | No |
| P3 | Scheduled sampling | ~15 | Yes |
| P4 | Eval dashboard | ~150 | No |
| T1 | Thất Ngôn support | ~50 | **Yes** |
| T2 | Multi-couplet (window=2) | ~30 | **Yes** |

**Retrain bundle:** T1 + T2 + P3 in one Colab session (~3-4h, ~837K pairs).

**Expected results:** 2 genres, 90%+ rhyme, multi-couplet context awareness, measurable quality.

---

## 🔮 v5 — New Data (Scraping Required)

| # | Item | Effort |
|---|------|--------|
| E1 | 8 canonical poets (Xuân Diệu, Hồ Xuân Hương, etc.) | Scrape + retrain |
| E2 | Window=3 multi-couplet | Preprocess tweak |
| E3 | Qwen2.5-1.5B QLoRA | Architecture swap |
