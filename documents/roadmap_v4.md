# 🚀 v4.0 Roadmap — Thất Ngôn + Inference Wins

> v3 shipped: 100% stress test, 80% rhyme, 97% tone, Lục Bát only.
> v4 adds Thất Ngôn (no new data) + beam rhyme + overgeneration fix.
> v5 = multi-couplet coherence + data expansion.

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

## ✅ v4 Items (done)

### P1: Beam Rhyme Constraint
**Effort: 30 min | Retrain: no | Status: ✅ DONE**

At rhyme position in 2nd output line, mask tokens whose rhyme group doesn't match `[RHYME:X]`. Only masks if at least one matching candidate exists (avoids all-masked edge case). Genre detected from `[THAT_NGON]` tag → pos7 for TN, pos6 for LB.

### P2: Fix Single-Line Overgeneration
**Effort: 15 min | Retrain: no | Status: ✅ DONE**

Post-generation cleanup trims >2 line outputs to first valid couplet. Genre-aware (6+8 for LB, 7+7 for TN).

### T1: Thất Ngôn Data Pipeline
**Effort: ~80 lines | Retrain: yes | Status: ✅ DONE**

41K bảy chữ poems from existing CSV. 748,807 training pairs (540K LB + 208K TN). Genre token `[LUC_BAT]` / `[THAT_NGON]` injected into format. Corpus regenerated, colab updated.

### Format

```
LB: <|start|> [DOI_THO] [LUC_BAT] [RHYME:X] [TONE:BBBBBB]
    6-syl <|linebreak|> 8-syl <|reply|> 6-syl <|linebreak|> 8-syl <|end|>

TN: <|start|> [DOI_THO] [THAT_NGON] [RHYME:X] [TONE:BBBBBBB]
    7-syl <|linebreak|> 7-syl <|reply|> 7-syl <|linebreak|> 7-syl <|end|>
```

---

## 🔴 Training Analysis

### Step 8400 results (val=2.90, with genre token)

| Metric | LB | TN |
|--------|-----|-----|
| Syllable | 60% | 0% |
| Rhyme | 80% | 60% |
| Tone | 60% | 0% |

### Root cause: data ratio, not format

Step-by-step token trace shows the model generates exactly 6 syllables then emits `<|linebreak|>` — regardless of `[THAT_NGON]` tag:

```
Step 5: "bát"  [6th syllable of output line 1]
Step 6: " "    
Step 7: <|linebreak|>   ← model decides line 1 done at 6 syllables
Step 8: "anh"  [starts line 2]
```

**Why:** At the position "just generated 6 syllables, what's next?", the model has seen:

| Next token | LB examples (540K) | TN examples (208K) |
|------------|-------------------|-------------------|
| 7th syllable | ❌ never | ✅ always |
| `<\|linebreak\|>` | ✅ always | ❌ never |

`<|linebreak|>` is correct **72% of the time**. The `[THAT_NGON]` genre tag (1 token) can't overcome this positional weight from 540K training examples.

---

## 🔧 T2: Two-Tier Fix

### T2a: Post-process fix (no retrain, ship now)

In `decode_doi_tho`, when genre is `[THAT_NGON]` and first output line < 7 syllables, reject the premature `<|linebreak|>`. Continue generating until 7 syllables reached, then insert linebreak manually.

**Cost:** ~20 lines in `sample.py`. Zero retrain. Fixes TN today.

### T2b: Weighted loss (retrain for v4.1)

TN examples get 2.6× loss weight during training. No data duplication — each real poem is used once, but the loss says "this matters more":

```python
# In training loop, after computing per-token loss (shape: B, T)
for i in range(batch_size):
    if is_tn[i]:
        loss[i] *= 2.6
```

**Why it works:** Stronger gradient for every aspect of TN — syllable count, rhyme, tone, word choice. The `<|linebreak|>` decision gets weighted through the full sequence. Same effect as oversampling but no duplicate data.

**Cost:** ~10 lines in `train.py`. One retrain.

### T2c: Drop `[DOI_THO]` (during retrain, optional)

Model is single-task (always đối thơ). `[DOI_THO]` costs 1 token per example with zero information. Removal simplifies format to:

```
<|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] ...
<|start|> [THAT_NGON] [RHYME:X] [TONE:BBBBBBB] ...
```

Only worth doing if retraining anyway. Saves ~20 lines across preprocess + inference code.

---

## 📋 v4 Status

| # | Item | Status |
|---|------|--------|
| P1 | Beam rhyme constraint | ✅ Done |
| P2 | Overgeneration fix | ✅ Done |
| T1 | Thất Ngôn pipeline (preprocess, corpus, format, colab) | ✅ Done |
| T1-FIX | Genre token `[LUC_BAT]`/`[THAT_NGON]` | ✅ Done |
| T2a | Post-process fix (no retrain) | ⬜ TODO |
| T2b | Weighted TN loss (retrain) | ⬜ TODO |
| T2c | Drop `[DOI_THO]` (retrain, optional) | ⬜ TODO |

---

## 📊 v4.1 Targets (after T2a + T2b)

| Metric | Target |
|--------|--------|
| Lục Bát syllable (6+8) | 90%+ |
| Thất Ngôn syllable (7+7) | 80%+ |
| Rhyme (combined) | 85%+ |
| Tone (combined) | 80%+ |
| Stress test | 100% |

---

## 🗂️ Data Inventory

| Source | Content | Status |
|--------|---------|--------|
| `poems_dataset_clean.csv` | 125K poems (lục bát + bảy chữ) | ✅ v4 uses all |
| 8 canonical poets | Hồ Xuân Hương, Hàn Mặc Tử, etc. | 🔮 v5 scraping |

> **v5:** Multi-couplet coherence + data expansion → [roadmap_v5.md](roadmap_v5.md)
