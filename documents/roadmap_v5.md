# 🔮 v5.0 Roadmap — Multi-Couplet + Data Expansion

> v4 shipped: Thất Ngôn support, beam rhyme, overgeneration fix.
> v5 tackles the two remaining gaps: multi-couplet coherence and data diversity.

---

## 📊 v4 Baseline (projected)

| Metric | v4 |
|--------|-----|
| Genres | Lục Bát + Thất Ngôn |
| Rhyme | 90%+ (with beam constraint) |
| Tone | 97% |
| Stress test | 100% |
| Multi-couplet coherence | ❌ Independent duels only |
| Data sources | 1 (vietnamese-poetry-corpus) |

---

## 🔴 P1: Multi-Couplet Coherence

### The Problem

Current design: **independent duels** — each user couplet gets its own response.

```
User sends:  C1, C2
Model does:  C1 → C3    (independent)
             C2 → C4    (independent)

Problem: C3 and C4 are generated in isolation.
They don't rhyme with each other. They don't form a flowing 4-line poem.

Expected behavior:
  C1: trèo lên cây khế nửa ngày / ai làm cho khế rụng đầy vườn ai
  C3: trái kia chín mọng trên cây / lòng em thương nhớ từ ngày xa anh
  C2: gió đưa cành trúc la đà / tiếng chuông Trấn Vũ canh gà Thọ Xương
  C4: chuông ngân vọng tiếng yêu thương / nhớ ai da diết đoạn trường nhớ ai
         ↑ should rhyme with C3's last word ("anh")

But C4 doesn't know C3 exists → no rhyme between C3 and C4.
```

### Why Window=2 Was Dropped from v4

Adding `(C1, C2) → C3` training pairs (window=2) to the existing window=1 corpus creates a format mismatch:

| | Training (mixed) | Inference (current) |
|---|---|---|
| Window=1 | 1 couplet → 1 couplet | 1 couplet → 1 couplet ✅ |
| Window=2 | 2 couplets → 1 couplet | **Never used at inference** ❌ |

Model would learn to expect 2 input couplets 33% of the time, but inference always sends 1. This degrades single-couplet quality — the model's primary use case.

### Approaches to Explore

| Approach | How It Works | Tradeoff |
|----------|-------------|----------|
| **A) Separate model/pass** | Train window=2 as a second training stage or separate model | Two checkpoints, more complexity |
| **B) Inference-side chaining** | Generate C3, then use C3's rhyme tag as additional context for C4 generation | Heuristic, may feel forced or mechanical |
| **C) `[MULTI]` format token** | Train on both `[DOI_THO]` (single) and `[DOI_THO:MULTI]` (multi) formats. Model learns to switch modes. | Requires both formats in training data |
| **D) 4-couplet output** | Input 2 couplets → output 2 couplets (C3+C4) in one generation. `[RHYME:X]` from C2, chain rhyme internally. | Longer sequences, may need block_size increase |

No decision yet. Research in v5.

---

## 🟡 P2: Data Expansion — 8 Canonical Poets

**Impact: Vocabulary diversity + stylistic range | Effort: Scrape + preprocess + retrain**

Current model is trained on a single dataset (vietnamese-poetry-corpus) — heavily dominated by Truyện Kiều. Output tends toward generic rural/romantic imagery.

| Poet | Style | Est. poems | Value |
|------|-------|-----------|-------|
| **Hồ Xuân Hương** | Wit, double-entendre, feminist | ~50 | Humor, wordplay |
| **Hàn Mặc Tử** | Symbolist, surreal, religious | ~80 | Abstract imagery |
| **Xuân Diệu** | Romantic, passionate, modern | ~160 | Love, urgency, modern vocab |
| **Huy Cận** | Philosophical, cosmic | ~80 | Nature, existence |
| **Nguyễn Bính** | Folk, rural, nostalgic | ~80 | Rural life, simple beauty |
| **Tố Hữu** | Revolutionary, patriotic | ~100 | Historical, political vocab |
| **Nguyễn Khuyến** | Classical, nature, autumn | ~100 | Seasonal imagery, formality |
| **Full Truyện Kiều** | Narrative epic | 1 poem, 3,254 lines | Deeper coverage |

**Expected:** 2-3× vocabulary diversity, fewer repetitive patterns, richer imagery.

---

## 📋 v5 Summary

| # | Item | Effort |
|---|------|--------|
| P1 | Multi-couplet coherence | Research + implement |
| P2 | 8 canonical poets | Scrape + preprocess + retrain |
