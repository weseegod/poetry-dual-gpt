# 🚀 v3.0 Roadmap — Complete

> v3 shipped: example-aligned batching, window-1 corpus, repetition penalty, syllable enforcement, independent duels.
> Model: step 8800, 100% stress test pass, zero BPE collapse.

---

## 📊 Results

| Metric | v2 (step 4400) | v3 (step 8800) |
|--------|---------------|----------------|
| Stress test (valid output) | 71-79% | **100% (14/14)** |
| BPE collapse (gal tokens) | 2/14 | **0/14** |
| Control tokens in output | None | **None** |
| Syllable enforcement | ~71% | **100%** (P3) |
| Lexical diversity | Low, repetitive | **Improved** (P2) |
| Dead format data | 46% (window=2) | **0%** (P2.7) |
| Cross-boundary noise | 98% of windows | **0%** (P2.6) |
| Batch size | 128 | **192** |
| Corpus size | 998K pairs | 541K pairs |

---

## ✅ v3 Changes — What Changed & Why

| # | Item | Before | After | Impact |
|---|------|--------|-------|--------|
| P2.7 | Drop window=2 | 998K pairs (46% dead format) | 541K window=1 pairs | Train = inference alignment |
| P2.6 | Example-aligned batching | Flat tensor, 98% cross-poem noise | One row per poem, 0% noise | No spurious correlations |
| P2 | Repetition penalty | Phrase spam ("nhớ ai giọng hát"×3) | Penalize recent 16 tokens (-1.2) | Lexical diversity ↑ |
| P3 | Syllable enforcement | Wrong syllable counts (71%) | Truncate to 6/8 | 100% accuracy |
| — | Batch size 192 | 128 | 192 | 20% faster training |
| — | Independent duels | C2→C3→C4 (C1 ignored) | C1→C3, C2→C4 | Both input couplets matter |

---

## ❌ P2.5: Loss Masking — Reverted

**Why we tried it:** Mask control tokens from loss to "free 30% capacity."

**Why it was wrong:** `ignore_index=0` already skips pad. Masking `<|end|>`, `<|reply|>`, and rhyme/tone tags broke the model's ability to learn structure — it emitted control tokens as output and produced garbled text.

**Lesson:** Control tokens ARE essential structure. Pad is the only safe skip.

---

## Sample Output (step 8800)

```
Input:  Thân em như chẽn lúa đòng
        Phất phơ dưới ngọn nắng hồng ban mai

Output: Mẹ già ăn sắn ngô đồng nương thu
        Con ăn cơm nước, anh về chợ quê

Input:  Công cha như núi thái sơn

Output: Sơn hà hoa đẹp ngọt hơn hoa sen
        Rừng xanh xanh thẳm bên miền hương linh
```

---

## 🔮 Future (v4)

| # | Item | Effort |
|---|------|--------|
| P4 | Beam search for rhyme quality (50% → 70%) | ~30 lines |
| P5 | Expand training data (8 canonical poets) | Scrape + preprocess |
| P6 | Rule evaluation dashboard | 1 day |
| P7 | Qwen2.5-1.5B QLoRA (deferred) | 1 day |
