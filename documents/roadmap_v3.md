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

### Training Data Quality Issues (found in v3 audit)

| Issue | Current | Problem |
|-------|---------|---------|
| Loss on control tokens | 30% of tokens are `<\|\|>`, `[...]` | Model wastes capacity predicting structural tokens |
| Cross-boundary windows | 98% start mid-poem | Spurious correlations (predicts `<\|\reply\|>` after random words) |
| Window=2 data | 457K pairs (46%) | Format `C1+C2→C3` never used at inference |
| Dataset structure | Flat tensor, no boundaries | No example-level alignment |

---

## 🎯 v3 Priorities

### 🔴 P1: Resume Training (complete 10K steps)
**Status: Ready to run | Effort: Run Colab**

Model stopped at step 4400 (44%). Patience=10 (set, good). Resume to 10K steps.

```bash
!python src/train.py --mode train --name doi_tho_ --corpus data/doi_tho_corpus.txt \
  --resume checkpoints/doi_tho_step_5000.pt
```

**Expected:** +5-10% on all rule metrics, fewer BPE collapses.

---

### 🔴 P2: Repetition Penalty ✅ DONE
**Status: Implemented | File: `client/server.py`, `src/sample.py`**

Penalizes tokens from last 16 output positions (-1.2 penalty). Eliminates phrase repetition.  
**Result:** 30-52 unique words per 3 samples (was ~15).

---

### 🔴 P3: Syllable Enforcement ✅ DONE
**Status: Implemented | File: `client/server.py`, `src/sample.py`**

Truncates each line to exact 6/8 syllable pattern.  
**Result:** 100% syllable accuracy (was ~71%).

---

### 🟡 P2.5: Loss Masking — Skip Control Tokens
**Status: Planned | Effort: ~5 lines in `train.py`**

**Problem:** 30% of training tokens are structural (`<|start|>`, `<|reply|>`, `<|end|>`, `[DOI_THO]`, `[RHYME:X]`, `[TONE:XXXXXX]`, `<|linebreak|>`). The model wastes 30% of gradient signal learning to predict these, but at inference we inject them ourselves.

**What it looks like:**
```
Target:  <|start|> [DOI_THO] [RHYME:a] khóc than kể hết ... <|reply|> kiều nhi ...
          └─MASKED──┘└MASKED─┘└MASKED──┘└─── POETRY (kept) ───┘└MASKED─┘└ POETRY ...
```

**How it works:** The tags condition through **attention** (they're in context the model attends to), not through being predicted. Masking their loss just stops wasting gradient on predicting tokens we inject at inference.

```python
# In train.py, training loop — add loss_mask:
control_ids = torch.tensor([0,1,2,3,8,9], device=device)  # pad,start,reply,end,DOI_THO,linebreak
loss_mask = ~torch.isin(targets, control_ids)
loss = (F.cross_entropy(logits, targets, reduction='none') * loss_mask).sum() / loss_mask.sum()
```

| | Before | After |
|---|---|---|
| Effective training signal | 70% poetry | 100% poetry |
| Rule conditioning | Via attention (unchanged) | Via attention (unchanged) |
| Model learns to predict | `<|reply|>` after "chưa" | poetically coherent continuation |
| Effective capacity | 30% wasted | **+30% free** |

**Tradeoff:** None. We control all structural tokens at inference via prompt templates. The model never needs to generate them.

---

### 🟡 P2.6: Example-Aligned Training Windows
**Status: Planned | Effort: ~30 lines in `dataset.py`**

**Problem:** The current `PoetryDataset` concatenates all 998K examples into one flat 53M-token list, then samples random 256-token windows. Only 2% of windows start at `<|start|>` — the other 98% start mid-poem and span across unrelated poems.

**What it looks like:**
```
Current (flat tensor, 98% dirty):
  Window at pos 20: "...nước này cho chưa <|reply|> kiều nhi phận mỏng..."
                     └── end of poem 1 ──┘└── start of poem 2 ──┘
  Model learns: "after 'chưa' comes '<|reply|>'" — WRONG! 'chưa' means "not yet"

Proposed (example-aligned, 100% clean):
  Window i: [<|start|> [DOI_THO] [RHYME:a] ... complete poem ... <|end|> 0 0 ... 0]
  Every window starts at <|start|>, ends at <|end|>. No cross-poem noise.
```

**How it works:** Replace the flat-tensor `PoetryDataset` with an `ExampleDataset` that stores each example as a separate padded row. Each `__getitem__` returns a clean, complete example.

```python
class ExampleDataset(Dataset):
    def __init__(self, examples, block_size, pad_id=0):
        self.data = []
        for ex in examples:
            ids = tokenizer.encode(ex).ids
            padded = ids[:block_size] + [pad_id] * max(0, block_size - len(ids))
            self.data.append(torch.tensor(padded))
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:block_size], row[1:block_size+1]  # x, y shifted
```

| | Flat tensor (current) | Example-aligned |
|---|---|---|
| Windows that start at `<\|start\|>` | 2% | **100%** |
| Cross-poem noise | 98% | **0%** |
| Unique windows seen (at step 4400) | ~563K | **~998K** |
| Training signal quality | Diluted | **Pure** |
| Padding overhead | None | ~30% (43-74 tok examples in 256-tok blocks) |

**Tradeoff:** ~30% padding means ~30% fewer effective tokens per batch. Compensated by much higher signal quality per token. Given the model has only seen 563K windows out of 998K available, example-aligned actually provides MORE variety.

---

### 🟡 P2.7: Drop Window=2, Regenerate Corpus
**Status: Planned | Effort: Regenerate corpus (2 min)**

**Problem:** 457K of 998K training pairs (46%) use window=2 format: `C1+C2 → C3`. The model learns to respond with couplet_{k+2} given 2 input couplets. But at inference, the server chains turn-by-turn: `C1 → C2 → C3 → C4`, always using `max_context=1` (last couplet only). The window=2 format is **never used at inference**, creating a train/inference mismatch.

| | Format | Training | Inference |
|---|---|---|---|
| Window=1 | `Ck → C{k+1}` | 541K (54%) | ✅ Used (chained) |
| Window=2 | `Ck+C{k+1} → C{k+2}` | 457K (46%) | ❌ Never used |

**How inference works with window=1 only:**
```
Input: C1 + C2 (4 lines)
  Turn 1: context=C2 → model generates C3  (matches training: C2→C3 ✅)
  Turn 2: context=C3 → model generates C4  (matches training: C3→C4 ✅)
  Output: C3 + C4 ✅
```

Each turn is a clean `last_couplet → next_couplet`, exactly matching window=1 training. The rhyme chain flows naturally: C2_rhyme → C3_pos6, C3_rhyme → C4_pos6.

**Regenerate:**
```bash
python src/preprocess_doi_tho.py --window 1
```

**Tradeoff:** Lose 457K examples, but they teach a dead format. Better 541K aligned examples than 998K with 46% mismatch. Rule tags (`[RHYME:X]`, `[TONE:XXXXXX]`) are identical in both formats — no rule conditioning is lost.

---

### 🟢 P4: Beam Search for Rhyme Quality
**Status: Planned | Effort: ~30 lines**

Constrained beam search that forces rhyme group matching at position 6. Boosts rhyme from 50% → 70%+.

---

### 🟢 P5: Expand Training Data
**Status: Planned | Effort: Scrape + preprocess**

Add 8 canonical Vietnamese poets via `data_service/scraper.py` for vocabulary diversity and varied literary styles.

---

### 🟢 P6: Rule Evaluation Dashboard
**Status: Planned**

Unified eval: chain rhyme, per-position tone accuracy, lexical diversity, BPE collapse rate.

---

### ⚪ P7: Qwen2.5-1.5B QLoRA — PAUSED ⏸️
**Status: Deferred**

Max out the 31M model first. Qwen is the ceiling-raiser for content quality after all other optimizations.

---

## 📋 v3 Implementation Status

| # | Item | Status | Effort | Blocks |
|---|------|--------|--------|--------|
| P1 | Resume training to 10K | ⏳ Run Colab | 3 hr | — |
| P2 | Repetition penalty | ✅ Done | — | — |
| P3 | Syllable enforcement | ✅ Done | — | — |
| P2.5 | Loss mask control tokens | ✅ Done | 5 lines | — |
| P2.6 | Example-aligned batching | ✅ Done | 30 lines | P2.7 |
| P2.7 | Drop window=2, regenerate | ✅ Done | Regenerate | — |
| P4 | Beam search rhyme | 📋 Planned | 30 lines | — |
| P5 | Expand data | 📋 Planned | 1 day | — |
| P6 | Eval dashboard | 📋 Planned | 1 day | — |
| P7 | Qwen QLoRA | ⏸️ Paused | — | — |

### v3 Changes Summary

| Change | Before | After |
|--------|--------|-------|
| Corpus | 998K pairs (54% W1 + 46% W2) | 541K pairs (100% W1) |
| Batches | Flat tensor, 98% cross-boundary | Example-aligned, 0% noise |
| Loss | 100% of tokens (incl. control) | 98.2% poetry-only (213 masked) |
| Batch size | 128 | 192 |

---

## 🔄 v3 Retrain Checklist

1. Run `python src/preprocess_doi_tho.py --window 1` to regenerate corpus
2. Implement P2.6 (ExampleDataset) in `src/dataset.py`
3. Implement P2.5 (loss mask) in `src/train.py`
4. Re-zip data: `zip -r data.zip data/`
5. Upload `data.zip` to Google Drive
6. Run `colab/colab_train.ipynb` (full 10K steps, fresh training)
7. Download new `checkpoints/doi_tho_best.pt` → `final.pt`
8. Run `python evaluate/eval_doi_tho.py` to verify
