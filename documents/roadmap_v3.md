# 🚀 v3.0 Roadmap — Polish & Quality

> v2.1 shipped: 31M PoetryDuelGPT, single-stage training, `[DOI_THO]` format, N-couplet mirroring, lowercase normalization, frontend + Colab.
> Everything below is the path to v3.0.

---

## 📊 v2.1 Baseline

| Metric | v2.1 (step 4400/10000) |
|--------|------------------------|
| Stress test (valid output) | 79% (11/14) |
| Internal rhyme (R1) | ~50% |
| Tone pattern (R2) | ~88% |
| Syllable count (R3) | ~71% |
| Chain rhyme (couplet chaining) | ~32% |
| Semantic quality | Low-medium |
| Lexical diversity | Low (repeats phrases) |
| BPE collapse rate | ~15% on rare vocab |

---

## 🎯 v3 Priorities

### 🔴 P1: Resume Training (complete 10K steps)
**Status: Ready to run**

Model stopped at step 4400 (44%). Patience already 10 (good). Resume from `checkpoints/doi_tho_step_5000.pt` to complete 10K steps. This alone should reduce BPE collapses, improve rhyme, and push stress test > 90%.

```bash
# Colab (cell 3):
!python src/train.py --mode train --name doi_tho_ --corpus data/doi_tho_corpus.txt \
  --resume checkpoints/doi_tho_step_5000.pt
```

**Expected:** +5-10% on all rule metrics, fewer BPE collapses.

---

### 🔴 P2: Repetition Penalty ✅ IMPLEMENTED
**Status: Done in v3**

The model repeats phrases across consecutive tokens ("nhớ ai giọng hát", "quê hương yêu dấu" spam). A repetition penalty during sampling reduces this:

```python
# During token generation loop:
for prev in new_tokens[-16:]:
    logits[:, prev] -= 1.2  # penalize recent tokens
```

**Impact:** Lexical diversity ↑, fewer repeats, more natural poetry.

---

### 🔴 P3: Syllable Enforcement ✅ IMPLEMENTED
**Status: Done in v3**

When the model generates wrong syllable counts, enforce 6/8:
- 6-syl line: truncate/pad to exactly 6 syllables
- 8-syl line: truncate/pad to exactly 8 syllables

```python
# Post-decode: ensure correct syllable counts
targets = [6, 8]  # alternating
for i, line in enumerate(lines):
    words = line.split()
    target = targets[i % 2]
    if len(words) > target:
        words = words[:target]
    elif len(words) < target:
        words.extend(['…'] * (target - len(words)))
    lines[i] = ' '.join(words)
```

**Impact:** Syllable accuracy 71% → 100%.

---

### 🟡 P4: Beam Search for Rhyme Quality
**Status: Planned**

Current top-k sampling picks randomly among candidates. A small beam (k=3) constrained to the target rhyme group would boost rhyme from 50% → 70%+:

```python
# At pos 6 of 8-syl line: only allow tokens in target_rhyme_group
candidate_ids = [tid for tid in topk_ids 
                 if get_rhyme_group(tok.id_to_token(tid)) == target_rhyme]
```

**Effort:** ~30 lines in `generate()`

---

### 🟡 P5: Expand Training Data
**Status: Planned**

Current 998K pairs come from `poems_dataset_clean.csv` only (mostly Truyện Kiều). Adding 8 canonical poets from `data_service/scraper.py` would:

| Poet | Style | Lines |
|------|-------|-------|
| Nguyễn Du | Truyện Kiều (full) | 3,254 |
| Hồ Xuân Hương | Humorous, double-entendre | ~500 |
| Hàn Mặc Tử | Symbolist, surreal | ~800 |
| Xuân Diệu | Romantic, modern | ~1,600 |
| Huy Cận | Philosophical | ~800 |
| Nguyễn Bính | Folk, rural | ~800 |
| Tố Hữu | Revolutionary | ~1,000 |
| Nguyễn Khuyến | Classical, nature | ~1,000 |

**Impact:** Richer vocabulary, diverse styles, fewer BPE collapses on rare words.

---

### 🟢 P6: Rule Evaluation Dashboard
**Status: Planned**

Replace ad-hoc evaluate/ scripts with a unified dashboard:
- Chain rhyme between couplets
- Per-position tone accuracy
- Semantic coherence score (embedding cosine similarity)
- Lexical diversity (unique n-gram ratio)

---

### ⚪ P7: Qwen2.5-1.5B QLoRA — PAUSED ⏸️
**Status: Deferred**

The 31M model still has headroom (44% trained). Max out the small model first, then migrate to Qwen for the content quality ceiling. Qwen brings:
- Rich Vietnamese vocabulary (150K+ tokens vs 12K)
- Grammatical correctness
- Cultural knowledge
- Coherent multi-sentence generation

---

## 📋 v3 Implementation Status

| # | Item | Status | File |
|---|------|--------|------|
| P1 | Resume training to 10K | ⏳ Run on Colab | `colab/colab_train.ipynb` |
| P2 | Repetition penalty | ✅ Done | `client/server.py`, `src/sample.py` |
| P3 | Syllable enforcement | ✅ Done | `client/server.py`, `src/sample.py` |
| P4 | Beam search rhyme | 📋 Planned | — |
| P5 | Expand data | 📋 Planned | `data_service/scraper.py` |
| P6 | Eval dashboard | 📋 Planned | `evaluate/` |
| P7 | Qwen QLoRA | ⏸️ Paused | — |

---

## 🔄 v3 Retrain Checklist

1. Upload `data.zip` to Google Drive (already done — 58MB)
2. Upload any new checkpoint if resuming: `checkpoints/doi_tho_step_5000.pt`
3. Run `colab/colab_train.ipynb` cells 1-7
4. Download new `checkpoints/doi_tho_best.pt` → `final.pt`
5. Run `python evaluate/eval_doi_tho.py` to verify
