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

---

## 🔴 Bug: Thất Ngôn Dropped

**`src/preprocess_doi_tho.py` line 156:**

```python
# Focus on Lục Bát for v2.0 (Thất Ngôn đối thơ is v3.0)
df = df[df["genre"] == "lục bát"]   # drops 41,354 bảy chữ poems
```

The comment promised v3.0. v3 shipped without it. The data was there the whole time.

---

## 🛡️ RISK ASSESSMENT — What Could Break v3 Quality

### P1: Beam Rhyme Constraint ✅ SAFE
**Risk: None.** Inference-only. Constrains 1 token per response. If rhyme mapping is wrong, model falls through to normal sampling. Reversible with one flag.

### P2: Fix Overgeneration ✅ SAFE
**Risk: None.** Post-processing only. Takes first valid (6,8) pair from output. Doesn't touch model.

### P3: Scheduled Sampling ⚠️ RISK
**Risk: Medium.** Changes training dynamics. Model has 31.5M params — small enough that teacher forcing is beneficial. Scheduled sampling can:
- Cause distribution drift (model's own noise accumulates)
- Destabilize early training if teacher_prob drops too fast
- Hurt more than help for narrow-domain models

**Recommendation:** Defer. The 31.5M-parameter ceiling + narrow domain (poetry) means teacher forcing works well. Scheduled sampling helps more on large models with diverse outputs.

### T1: Thất Ngôn Support ⚠️ RISK (Mitigatable)
**Risks:**
1. **Genre confusion:** 31.5M params splitting capacity between 2 genres → possible 5-10% Lục Bát quality drop
2. **Special token issue:** `[DOI_THO:TN]` is 8 BPE subwords, not 1 token. Must be added to tokenizer. Retrain BPE.
3. **Syllable mismatch:** Model might mix 6-syl and 7-syl outputs

**Mitigations:**
1. Use `[DOI_THO]` for both (no new token). Model learns from tone sequence length (6 vs 7) + input line length.
2. Keep Lục Bát at 87% of data (541K/621K). Thất Ngôn is 13% — small enough not to dominate.
3. If Lục Bát quality drops, train separate models.

**Recommendation:** Proceed with shared `[DOI_THO]` format (no new token needed). Risk is low — 13% data dilution with implicit genre signal from tone sequence length.

### T2: Multi-Couplet Window=2 ❌ DANGER — Do Not Add
**Risk: High.** Critical conflict with inference:

| | Training | Inference |
|---|----------|-----------|
| Window=1 | C1 → C2 | C1 → C2 ✅ match |
| Window=2 | (C1, C2) → C3 | C1 → C3, C2 → C4 ❌ mismatch |

Adding window=2 teaches the model a format that NEVER appears at inference. The model learns to expect 2 input couplets when it only gets 1. This degrades single-couplet quality — the exact opposite of what we want.

**Alternative — inference-only multi-couplet chaining:**
```
User sends:  C1
Model →     C3
Chain:      use C3 as next input
Model →     C5  
Chain:      use C5 as next input
Model →     C7
```

No training changes. No new data. Chain rhyme propagates naturally because each generation's rhyme tag feeds the next. This is how human đối thơ works — each response becomes the next prompt.

**Recommendation:** Skip T2. Implement iterative chaining at inference instead.

---

## ✅ v4 Plan — Risk-Adjusted

| # | Item | Effort | Retrain | Risk |
|---|------|--------|---------|------|
| P1 | Beam rhyme constraint | 30 min | No | None |
| P2 | Fix overgeneration | 15 min | No | None |
| T1 | Thất Ngôn support | ~50 lines | Yes (3h) | Low |
| — | Multi-couplet via chaining | ~20 lines | No | None |

**Dropped from v4:**
- P3 (Scheduled sampling) — medium risk, low reward for small model
- T2 (Window=2 training) — conflicts with independent- duel inference

**Retrain:** Only T1 triggers retrain (adds 80K Thất Ngôn pairs). Everything else is code-only.

---

## 🔮 v5 — New Data

| # | Item |
|---|------|
| E1 | 8 canonical poets (Xuân Diệu, Hồ Xuân Hương, etc.) |
| E2 | Window=2 training if inference changes |
| E3 | Scheduled sampling after data expansion |
