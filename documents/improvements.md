# 🚀 v2.0 Roadmap

> v1.0 shipped: 31M params, 4 rules (rhyme 58%, tone 88%, syllable 78%, đối âm 69%), two-stage training, chat UI, Colab pipeline.
> Everything below is the path to v2.0.

---

## 🔮 v2.0: Qwen2.5-1.5B QLoRA

**Goal:** Swap the 31M custom GPT for a 1.5B pretrained Qwen model with LoRA fine-tuning. Same control tokens transfer directly — no format changes needed.

**Why:** A 31M model trained from scratch on 135K poems hits a quality ceiling. Qwen2.5-1.5B was pretrained on terabytes of text including Vietnamese. It already knows grammar, vocabulary, idioms, and cultural context. Fine-tuning on poetry data with control tokens just teaches it the rules — the foundation is already there.

**What Qwen brings:**
- Rich Vietnamese vocabulary (not limited to 11K BPE tokens)
- Grammatical correctness (no garbled subword fragments)
- Cultural knowledge (folklore, idioms, historical references)
- Coherent multi-sentence generation

**Implementation:**
```
Same training format → same preprocess.py → same control tokens
Just swap PoetryDuelGPT(31M) → Qwen2.5(1.5B) + LoRA adapters
```

**Requirements:** `transformers` + `peft` + `bitsandbytes`. Colab T4/L4 handles 4-bit QLoRA.

**Expected quality:** Semantic coherence from "medium" → "high". Rule accuracy similar or better. Poetry that reads like poetry, not just rule-compliant text.

---

## 📜 Multi-Couplet Generation

**Goal:** Generate full 4-line (Lục Bát) or 8-line (Thất Ngôn bát cú) poems, not just single couplets.

**Why:** Vietnamese poetry is structured in stanzas. A Lục Bát poem is 4+ lines (6-8-6-8...), and Thất Ngôn bát cú is 8 lines (7-7-7-7-7-7-7-7). Single couplet generation shows the model understands the form, but full-poem generation unlocks:
- **End rhyme** between couplets (currently skipped because we only generate one couplet)
- **Thematic coherence** across stanzas
- **Real poetic structure** that reads like actual Vietnamese poetry

**Implementation approach:**
1. Train on multi-couplet sequences (2-4 couplets per example)
2. Add `[COUPLET1]`, `[COUPLET2]` markers
3. Generate autoregressively until `[END_POEM]` token
4. This automatically enables end-rhyme evaluation between couplets

**Challenge:** Longer sequences (4 couplets ≈ 56 syllables ≈ ~60 tokens) fit in block_size=256, so no architecture change needed. Just data format change.

---

## 📚 Better Training Data

**Goal:** Expand beyond the current 135K-poem single-source corpus.

**Current data:** 135,863 poems from one collection → 942K pairs. Good for form/rules, but limited vocabulary and themes (mostly classical/romantic poetry).

**Suggested additions:**

| Source | Poems | Style | Value |
|--------|-------|-------|-------|
| Ca dao / folk poetry | 5,000+ | Rural life, proverbs, love | Everyday Vietnamese, idioms |
| Nguyễn Du (Truyện Kiều) | 3,254 lines | Epic, classical | Rich vocabulary, literary canon |
| Tố Hữu | ~500 poems | Revolutionary, modern | 20th century Vietnamese |
| Xuân Diệu, Huy Cận | ~1,000 poems | Romantic, modern | Diverse themes, modern language |
| Lục Bát online collections | 10,000+ | Various | Volume for fine-tuning |

**Impact:** Broader vocabulary, more diverse themes, better generalization to novel prompts. The current model only knows the vocabulary of its 135K-poem corpus — adding folk poetry and modern works would dramatically expand its range.

---

## 📊 v1.0 Baseline (for comparison)

| Metric | Stage 2 (v1.0) |
|--------|----------------|
| Rhyme (R1) | 58.4% |
| Tone (R2) | 87.5% |
| Syllable (R3) | 78.0% |
| Đối Âm (R4, Stage 1) | 69.4% |
| All 3 pass | 50.9% |
| Semantic quality | Medium |

Full evaluation: [rule_evaluation.md](rule_evaluation.md)

---

## 🎯 Priority

| # | Item | Impact | Effort | Blocks |
|---|------|--------|--------|--------|
| 1 | Qwen2.5-1.5B QLoRA | 🚀🚀🚀 | 1 day | Nothing — same data, same tokens |
| 2 | Multi-couplet generation | 🚀🚀 | 2-3 days | Enables end-rhyme rule |
| 3 | Better training data | 🚀 | Ongoing | Requires data collection |
