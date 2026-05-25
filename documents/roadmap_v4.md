# 🚀 v4.0 Roadmap — Thất Ngôn + Inference Wins

> v3 shipped: 100% stress test, 80% rhyme, 97% tone, Lục Bát only.
> v4 adds Thất Ngôn (no new data) + beam rhyme + overgeneration fix.
> One retrain. v5 = multi-couplet coherence + data expansion.

---

## 📊 v3 Baseline

| Metric | v3 |
|--------|-----|
| Stress test | 100% |
| Rhyme (pos6) | 80% |
| Tone (avg) | 97% |
| Syllable (6+8) | 93% |
| Thất Ngôn (7→7) | ❌ Dropped by line 156 |
| Multi-couplet coherence | ❌ Independent duels only |

---

## 🔴 Bug: Thất Ngôn Dropped

**`src/preprocess_doi_tho.py` line 156:**

```python
# Focus on Lục Bát for v2.0 (Thất Ngôn đối thơ is v3.0)
df = df[df["genre"] == "lục bát"]   # drops 41,354 bảy chữ poems
```

v3 shipped without Thất Ngôn. The data (41K poems) was there the whole time.

---

## ✅ v4 Items

### P1: Beam Rhyme Constraint (80% → 90%+)
**Effort: 30 min | Retrain: no | Risk: none | Status: ✅ DONE**

At the rhyme position (pos 6 of 8-syl line for LB, pos 7 for TN), force candidates to match `[RHYME:X]`. Inference-only. Togglable flag. Uses `get_rhyme_group()` to check candidate tokens. Only masks non-matching tokens if at least one matching candidate exists (avoids all-masked edge case).

---

### P2: Fix Single-Line Overgeneration
**Effort: 15 min | Retrain: no | Risk: none | Status: ✅ DONE**

Post-generation cleanup when model outputs >2 lines. Detects genre from first line syllable count, finds first valid couplet matching that genre's syllable pattern.

---

### T1: Thất Ngôn Support (7→7)
**Effort: ~50 lines in preprocess_doi_tho.py | Retrain: yes (~3h) | Risk: low | Status: ✅ CODE DONE, ⚠️ NEEDS RETRAIN**

**Data:** 41K bảy chữ poems already in CSV. 748,807 total training pairs (540K LB + 208K TN).

---

## 🔧 T1-FIX: Genre Token (step 8800 evaluation → fix)

### Problem

At step 8800 (val=3.04), Lục Bát held at 80% but Thất Ngôn first lines stuck at 6 syllables:

| Metric | v4 LB | v4 TN | Root cause |
|--------|-------|-------|------------|
| Syllable | 80% | 0% | Model defaults to 6-syl first line |
| Rhyme | 80% | 60% | Rhyme works even with wrong syl count |
| Tone | 60% | 0% | Dual-genre confused tone patterns |

**Root cause:** 540K LB vs 208K TN (2.6:1 ratio). Model learns "first line = 6 syllables" as dominant pattern. The 6 vs 7 char `[TONE:...]` tag length is too subtle a signal.

### Fix: Explicit Genre Token

Change format from:
```
[DOI_THO] [RHYME:X] [TONE:BBBBBB] ...    (model must guess genre)
```
To:
```
[DOI_THO] [LUC_BAT] [RHYME:X] [TONE:BBBBBB] ...
[DOI_THO] [THAT_NGON] [RHYME:X] [TONE:BBBBBBB] ...
```

- `[LUC_BAT]` (token 4) and `[THAT_NGON]` (token 7) already exist as single-token special tokens
- No BPE retrain needed
- `GENRE_CONFIG` now includes `genre_token` field
- `make_doi_tho_pairs_multi()` accepts `genre_token` parameter
- Inference code (`auto_tag_doi_tho`, `_build_doi_tho_prompt`) detects genre and injects token
- Rhyme constraint detects genre from `[THAT_NGON]` presence instead of tone tag length

### Files changed for T1-FIX

| File | Change |
|------|--------|
| `src/preprocess_doi_tho.py` | `genre_token` in `GENRE_CONFIG` + `make_doi_tho_pairs_multi` |
| `src/sample.py` | `auto_tag_doi_tho` injects `[LUC_BAT]`/`[THAT_NGON]`; rhyme constraint uses tag |
| `client/server.py` | `_build_doi_tho_prompt` + `_auto_tag_doi_tho` inject genre token |

### Retrain needed

Regenerate corpus → upload `data.zip` to Drive → run `colab/colab_train.ipynb`. The explicit genre signal should fix the 6-vs-7 syllable distinction since the model now has an unambiguous token for each genre.

---

## 📋 v4 Status

| # | Item | Status |
|---|------|--------|
| P1 | Beam rhyme constraint | ✅ Done |
| P2 | Overgeneration fix | ✅ Done |
| T1 | Thất Ngôn preprocessing | ✅ Done |
| T1-FIX | Genre token | ✅ Code done, ⚠️ retrain |
| — | Corpus (748K pairs) | ✅ Regenerated |
| — | data.zip | ✅ Updated |
| — | Colab notebook | ✅ Updated for v4 |
| — | Tests (90/90) | ✅ Passing |

---

## 📊 v4.1 Target (after retrain with genre token)

| Metric | Target |
|--------|--------|
| Lục Bát syllable (6+8) | 90%+ |
| Thất Ngôn syllable (7+7) | 80%+ |
| Rhyme (combined) | 85%+ |
| Tone (combined) | 85%+ |
| Stress test | 100% |

---

## 🗂️ Data Inventory

| Source | Content | Status |
|--------|---------|--------|
| `poems_dataset_clean.csv` | 125K poems (lục bát + bảy chữ) | ✅ v4 uses all of it |
| 8 canonical poets | Hồ Xuân Hương, Hàn Mặc Tử, etc. | 🔮 v5 scraping |

No new data needed for v4. All from existing CSV.

> **v5:** Multi-couplet coherence + data expansion → [roadmap_v5.md](roadmap_v5.md)
