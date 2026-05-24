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

## ✅ v4 Plan

### P1: Beam Rhyme Constraint (80% → 90%+)
**Effort: 30 min | Retrain: no | Risk: none**

At the rhyme position (pos 6 of 8-syl line), force candidates to match `[RHYME:X]`:

```python
if position == rhyme_pos:
    for tid in candidates:
        if get_rhyme_group(tok.decode([tid])) != target_rhyme:
            logits[:, tid] = float('-inf')
```

Inference-only. One token per response. Togglable flag.

---

### P2: Fix Single-Line Overgeneration
**Effort: 15 min | Retrain: no | Risk: none**

Post-generation cleanup when model outputs >2 lines:

```python
if len(lines) > 2:
    for i in range(len(lines) - 1):
        if len(lines[i].split()) == 6 and len(lines[i+1].split()) == 8:
            lines = lines[i:i+2]; break
    else:
        lines = lines[:2]
```

---

### T1: Thất Ngôn Support (7→7)
**Effort: ~50 lines in preprocess_doi_tho.py | Retrain: yes (3h) | Risk: low**

**Data:** 41K bảy chữ poems already in CSV. No scraping needed.

**Format:** Shared `[DOI_THO]` tag for both genres. Model distinguishes via tone sequence (6 vs 7 characters) and syllable count. No new special token needed — avoids BPE retraining.

| | Lục Bát | Thất Ngôn |
|---|---|---|
| Input | 6 + 8 syllables | 7 + 7 syllables |
| Tone tag | `[TONE:BBBBBB]` (6) | `[TONE:BBBBBBB]` (7) |
| Rhyme | Pos 8 of 8-syl line | Pos 7 of input 7-syl line |
| Data | 541K pairs | ~80K pairs (13% of total) |

**Risk mitigation:** Thất Ngôn is only 13% of training data — Lục Bát dominates. If Lục Bát quality drops, the shared `[DOI_THO]` format can be split into `[DOI_THO:LB]` / `[DOI_THO:TN]`.

**Corpus size after T1:** ~620K pairs.

---

## 📋 v4 Summary

| # | Item | Time | Retrain |
|---|------|------|---------|
| P1 | Beam rhyme | 30 min | No |
| P2 | Fix overgeneration | 15 min | No |
| T1 | Thất Ngôn | 1h + 3h Colab | Yes |

**Retrain:** T1 only. ~620K pairs. 3 hours on Colab T4.

---

## 🗂️ Data Inventory

| Source | Content | Status |
|--------|---------|--------|
| `poems_dataset_clean.csv` | 125K poems (lục bát + bảy chữ) | ✅ v4 uses all of it |
| 8 canonical poets | Hồ Xuân Hương, Hàn Mặc Tử, etc. | 🔮 v5 scraping |

No new data needed for v4. All from existing CSV.

> **v5:** Multi-couplet coherence + data expansion → [roadmap_v5.md](roadmap_v5.md)
