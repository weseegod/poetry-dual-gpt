# 🔮 v5.0 Roadmap — Thất Ngôn Quality + Data Expansion

> v4.1 shipped: LB 100%/100%/80% (syl/tone/rhyme), TN 60%/0%/40%.
> v5 fixes TN quality (needs more data) + adds canonical poets for style diversity.

---

## 📊 v4.1 Baseline

| Metric | LB | TN |
|--------|-----|-----|
| Stress test | 100% | 100% |
| Syllable | **100%** (6+8) | 60% (7+7) |
| Rhyme | 80% (pos6) | 40% (pos7) |
| Tone | **100%** (BTB/BTBB) | 0% (BTB/BTB) |
| Text quality | Fluent poetry | Garbled words, BPE artifacts |

**Root cause for TN:** Only 41K poems → 208K training pairs at 28% data ratio. The model learns the shape (syllable count via post-process hack) but not the soul (fluent 7-syllable poetry). 41K poems spread across 272 authors — most have <200 poems, no single author provides enough data for the model to learn any particular TN style.

---

## 🎯 v5 Priorities

### P0: Thất Ngôn Data Expansion
**Impact: Fix TN quality | Effort: Scrape + preprocess | Retrain: yes**

Current TN data is too thin. Need 2-3× more poems.

| Source | Est. poems | Genre | Quality |
|--------|-----------|-------|---------|
| **Thivien.net scraping** | 5,000-10,000 | Thất ngôn | High — classical poets |
| **Thơ Đường Luật collections** | 2,000-5,000 | Thất ngôn bát cú | High — structured form |
| **Existing CSV unused** | ? | Check if any TN poems were filtered out | Quick win |

**Target:** 60K-80K TN poems total (vs 41K now). At that scale, TN should hit 80%+ fluency.

### P1: Tone Fix for Thất Ngôn
**Impact: Fix TN tone (0% → 80%+) | Effort: Analysis + training tweak | Retrain: yes**

TN tone at 0% because:
1. Post-process re-split scrambles word order → tone pattern destroyed
2. Model never learned TN tone patterns (only 28% data)

Fix: once P0 adds enough TN data, remove post-process hack. Model should learn 7+7 + BTB natively. If tone still weak, add tone-specific loss weight for TN examples.

### P2: Canonical Poets (Style Diversity)
**Impact: Richer vocabulary, less repetitive output | Effort: Scrape + preprocess | Retrain: yes**

Current model's vocabulary is biased toward one dataset — output tends toward generic rural/romantic imagery. Adding distinctive poetic voices expands range.

| Poet | Poems | Style value | Priority |
|------|-------|-------------|----------|
| **Nguyễn Bính** | ~80 | Folk, rural, simple beauty — aligns with current data | S1 |
| **Hồ Xuân Hương** | ~50 | Wit, double-entendre, feminist — unique vocabulary | S1 |
| **Hàn Mặc Tử** | ~80 | Symbolist, surreal, religious — abstract imagery | S2 |
| **Xuân Diệu** | ~160 | Romantic, passionate, modern vocab | S2 |
| **Nguyễn Khuyến** | ~100 | Classical, nature, autumn — formal register | S3 |
| Tố Hữu, Huy Cận, Tản Đà | ~280 | Mix of styles | S3 |

### P3: Multi-Couplet Coherence (Research)
**Impact: C3 rhymes with C4 | Effort: Experiment | Retrain: maybe**

Current independent-duel approach works for single-couplet input. Multi-couplet is a niche feature. Investigate inference-side chaining first — no retrain needed.

---

## 📋 v5 Plan

| # | Item | Effort | Retrain |
|---|------|--------|---------|
| P0 | TN data expansion (20K-40K more poems) | Scrape + 1h code | Yes |
| P1 | Remove post-process hack, native TN training | 0 code (P0 enables this) | Yes |
| P2 | 3-5 canonical poets | Scrape + 1h code | Yes |
| P3 | Multi-couplet research | Experiment | Maybe |

**One retrain.** All data changes combined into single corpus.

---

## 📊 v5 Targets

| Metric | LB | TN |
|--------|-----|-----|
| Syllable | 100% | 85%+ |
| Rhyme | 85%+ | 70%+ |
| Tone | 100% | 80%+ |
| Text quality | Fluent | Fluent |

---

## 🗂️ Data Inventory

| Source | Content | Status |
|--------|---------|--------|
| `poems_dataset_clean.csv` | 125K poems (84K LB + 41K TN) | ✅ v4 |
| Thivien.net scraping | ~10K TN poems | 🔮 v5 |
| Canonical poets | Nguyễn Bính, Hồ Xuân Hương, Hàn Mặc Tử, Xuân Diệu | 🔮 v5 |
| Full Truyện Kiều | 3,254 lines | 🔮 v5 (optional) |
