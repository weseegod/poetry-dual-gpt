# 📜 Single-Stage Đối Thơ Training — Plan (v2.1)

> **Status:** Finalized  
> **Decision:** One corpus, one stage. Simpler, faster, focused on the product.

---

## 1. The Decision

**Train on `data/doi_tho_corpus.txt` only. One stage. No combined corpus. No Stage 2.**

The product is đối thơ — couplet-to-couplet dueling. Training on what users actually do is better than mixing formats and diluting model attention.

---

## 2. What Rules Are Preserved

### `doi_tho_corpus.txt` contains: 998,002 pairs, all with `[RHYME:X]` and `[TONE:XXXXXX]`

| Rule | How It's Encoded | Coverage |
|------|-----------------|----------|
| **Syllable count** (6→8 per couplet) | Implicit in data. Every output has 6+8. | 100% (100/100 verified) |
| **Tone pattern** (B-T-B / B-T-B-B) | Implicit in data + `[TONE:XXXXXX]` tag from 6-syl line | 98% (98/100 B-T-B verified) |
| **Chain rhyme** (pos 8 of input → pos 6 of first output line) | `[RHYME:X]` tag extracted from pos 8 of last input 8-syl line | 100% (tag on every pair) |
| **Internal rhyme** (pos 6 of output 6-syl ↔ pos 6 of output 8-syl) | Implicit in data. Every output couplet in the corpus has correct internal rhyme. | 69% exact, rest thông vần |

### What's Lost

| Capability | Status | Impact |
|-----------|--------|--------|
| `[LUC_BAT]` single-couplet mode | ❌ Gone | User typing 1 line won't work. But đối thơ always takes couplets. |
| `[THAT_NGON]` 7-syllable mode | ❌ Gone | Not part of the product. |
| Multi-genre flexibility | ❌ Gone | Only Lục Bát đối thơ. Add later if needed. |

### Verdict: **Nothing essential is lost.** All rules needed for Lục Bát đối thơ are present in the data.

---

## 3. The Simplified Training Flow

```
Locally (one time):
  python src/preprocess_doi_tho.py
  → data/doi_tho_corpus.txt (998K pairs)
  → zip data/ → Drive

Colab:
  Cell 2:  Download data.zip, unzip
           python src/train_bpe.py --corpus data/doi_tho_corpus.txt
  
  Cell 3:  python src/train.py --corpus data/doi_tho_corpus.txt --name stage1_
           → 10K–15K steps, one stage, done

  Cell 4:  Verify + generate
```

**Time:** ~3 hours on T4 (vs 4+ hours for two-stage).

---

## 4. Changes Needed

### Code changes: minimal

| File | Change |
|------|--------|
| `colab/colab_train.ipynb` | Remove Stage 2. Single corpus `data/doi_tho_corpus.txt`. Remove verify/comparison cells. |
| `src/train.py` | No changes needed — already supports `--corpus` override |
| `src/preprocess.py` | No changes. Still useful for single-couplet if ever needed. |
| `src/preprocess_doi_tho.py` | No changes. Already correct. |

### Data: already done

`data/doi_tho_corpus.txt` is already generated and verified:
- 998,002 pairs
- 540,728 window=1, 457,274 window=2
- 100% have `[RHYME:X]` and `[TONE:XXXXXX]`
- Syllable accuracy: 100% (from 100-sample check)
- Tone accuracy: 98%
- Re-zip data/ with just the necessary files

---

## 5. Training Config

```python
# Single stage, focused
python src/train.py \
    --mode train \
    --name doi_tho_ \
    --corpus data/doi_tho_corpus.txt \
    --steps 10000
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_steps` | 10,000 | Same as old Stage 1. 998K pairs at batch 128. |
| `batch_size` | 128 | T4-safe. |
| `learning_rate` | 3e-4 | From scratch (no fine-tune needed). |
| `corpus` | `data/doi_tho_corpus.txt` | Single format, 100% focused. |

---

## 6. Expected Quality

Based on v1 experience (942K single-couplet pairs → 50% rhyme, 88% tone):

| Metric | v1 (single-couplet, 2-stage) | v2.1 (đối thơ, 1-stage) |
|--------|------------------------------|-------------------------|
| Syllable count | 71% | 70-80% expected |
| Tone pattern | 88% | 85-90% expected |
| Chain rhyme | N/A | 40-60% expected |
| Internal rhyme | 50% | 45-55% expected |
| Output is valid couplet | N/A | 80%+ expected |

---

## 7. Rollback Plan

If single-stage results are poor:
1. Fall back to combined corpus (`poetry_corpus_combined.txt`)
2. Add Stage 2 fine-tune
3. The data is already generated, just use different config
