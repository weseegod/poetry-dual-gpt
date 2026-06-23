# 📜 Multi-Couplet Đối Thơ — Comprehensive Plan (v2)

> **Status:** Strategy Locked — Ready for Implementation
> **Target:** v2.0 — PoetryDuelGPT 31M (Qwen migration skipped)
> **Baseline:** Stage 2 — R1:Rhyme 50% | R2:Tone 88% | R3:Syllable 71% | Combined: 46%
> **Author:** 2026-05-22

---

## 1. 🎯 What "Đối Thơ" Actually Means

From the user's description of Vietnamese poetry duel rules:

> Two players (A and B) take turns reciting poetry on the spot. Each must:
> - **Nối ý** (follow the idea): Continue or respond to the previous player's meaning — don't go off-topic
> - **Nối vần** (follow the rhyme): The last syllable must rhyme according to the poetic form's rules
> - **Nhanh & duyên** (fast & witty): Poetry should be clever, brief, often romantic or playful
>
> Player A opens with a full couplet. B responds with a full couplet. Then A continues. And so on.

### Lục Bát Duel Rules (the target form)

```
A:  đêm nay gió mát trăng thanh        (6 syllables)
    hỏi em duyên nợ mấy phần với anh    (8 syllables)

B:  duyên em còn nợ anh nhiều          (6 syllables)
    chờ anh xây cầu em mới sang sông   (8 syllables)

Rhyme chain: "anh" (A's pos 8) → "nhiều" (B's pos 6 = "iêu", thông vần with "anh")
             "sông" (B's pos 8) → A's next pos 6 must rhyme with "ông"
```

**Key difference from single-couplet generation:**

| | Single Couplet (v1) | Đối Thơ (v2) |
|---|---|---|
| Input | 1 line (6 syl) | 1-2 couplets (2-4 lines) |
| Output | 1 line (8 syl) | 1 couplet (2 lines) |
| Rhyme source | Pos 6 of input | Pos 8 of last input line |
| Context | 1 line | Full recent history |
| Interaction | One-shot | Turn-based, chained |

---

## 2. 📐 Training Strategy: Sliding Pair Windows

### 2.1 The Core Idea

From a long poem, every adjacent couplet pair becomes a training example. Two window sizes:

| Window | Input | Output |
|--------|-------|--------|
| `window=1` | couplet_k | couplet_{k+1} |
| `window=2` | couplet_k + couplet_{k+1} | couplet_{k+2} |

### 2.2 Concrete Example (Truyện Kiều)

```
Couplet 1:  gặp cơn gia biến lạ dường
            bán mình nó đã tìm đường cứu cha

Couplet 2:  dùng dằng khi bước chân ra
            cực trăm nghìn nỗi, dặn ba bốn lần

Couplet 3:  trót lời nặng với lang quân
            mượn con em nó thúy vân thay lời

Couplet 4:  gọi là trả chút nghĩa người
            sầu này dằng dặc muôn đời chưa quên!
```

**Generated training examples:**

```
Example 1 (w=1):
  Input:  couplet 1 (2 lines)
  Output: couplet 2 (2 lines)

Example 2 (w=2):
  Input:  couplet 1 + couplet 2 (4 lines)
  Output: couplet 3 (2 lines)

Example 3 (w=1):
  Input:  couplet 2 (2 lines)
  Output: couplet 3 (2 lines)

Example 4 (w=2):
  Input:  couplet 2 + couplet 3 (4 lines)
  Output: couplet 4 (2 lines)

Example 5 (w=1):
  Input:  couplet 3 (2 lines)
  Output: couplet 4 (2 lines)
```

**Data yield per poem:** `(N-1)` window=1 examples + `(N-2)` window=2 examples = `2N-3`

Truyện Kiều (1,627 couplets) → ~3,250 examples from one work alone.

### 2.3 Why Both Window Sizes?

- **window=1**: Handles turn 1 of a duel (user just gave 1 couplet). Model learns to continue from minimal context.
- **window=2**: Handles turn 2+ (2 couplets of history). Model learns to use richer context for better nối ý.
- Model is trained with both, so it handles any input length seamlessly.

### 2.4 Why Only 2 Couplets (Not More)?

```
window=1:  ~25 tokens input + ~20 tokens output = ~45 tokens
window=2:  ~50 tokens input + ~20 tokens output = ~70 tokens
window=3:  ~75 tokens input + ~20 tokens output = ~95 tokens
window=4: ~100 tokens input + ~20 tokens output = ~120 tokens
```

window=2 is the sweet spot:
- Enough context for semantic coherence (nối ý)
- Fast training (within block_size=256 with massive room to spare)
- Configurable — bump to 3 or 4 if needed later
- Default to 2 for speed

---

## 3. 📝 Training Data Format

### 3.1 Final Format

```
<|start|> [DOI_THO] [RHYME:X] [TONE:XXXXXX]
  <input lines, joined with <|linebreak|>> <|reply|>
  <output lines, joined with <|linebreak|>> <|end|>
```

### 3.2 Window=1 Example

```
<|start|> [DOI_THO] [RHYME:a] [TONE:BTTBBB]
  gặp cơn gia biến lạ dường <|linebreak|> bán mình nó đã tìm đường cứu cha <|reply|>
  dùng dằng khi bước chân ra <|linebreak|> cực trăm nghìn nỗi dặn ba bốn lần <|end|>
```

### 3.3 Window=2 Example

```
<|start|> [DOI_THO] [RHYME:ân] [TONE:TTBBBT]
  gặp cơn gia biến lạ dường <|linebreak|> bán mình nó đã tìm đường cứu cha <|linebreak|>
  dùng dằng khi bước chân ra <|linebreak|> cực trăm nghìn nỗi dặn ba bốn lần <|reply|>
  trót lời nặng với lang quân <|linebreak|> mượn con em nó thúy vân thay lời <|end|>
```

### 3.4 How Tags Are Extracted

**Always from the LAST couplet of the input** (the couplet immediately before `<|reply|>`):

```python
# For window=2, the last input couplet is couplet 2:
#   dùng dằng khi bước chân ra            ← 6-syllable line
#   cực trăm nghìn nỗi dặn ba bốn lần     ← 8-syllable line

rhyme_6syl = get_rhyme_group(last_6syl.split()[5])    # pos 6 for [TONE] check
rhyme_8syl = get_rhyme_group(last_8syl.split()[7])    # pos 8 for [RHYME] — THIS IS THE CHAIN RHYME
tone_seq   = get_tone_sequence(last_6syl)[:6]          # tone pattern of last 6-syl line

# → [RHYME:ân] [TONE:TTBBBT]
```

**Verification that the rhyme chain works:**

| Example | Input last couplet | [RHYME] from pos 8 | Output pos 6 | Rhyme match |
|---------|-------------------|---------------------|--------------|-------------|
| w=1, ex1 | cặp1: "cha" | [RHYME:a] | "ra" → "a" | ✅ |
| w=2, ex2 | cặp2: "lần" | [RHYME:ân] | "quân" → "ân" | ✅ |
| w=1, ex3 | cặp2: "lần" | [RHYME:ân] | "quân" → "ân" | ✅ |
| w=2, ex4 | cặp3: "lời" | [RHYME:ơi] | "người" → "ươi" | ⚠️ thông vần (acceptable) |

---

## 4. 🆕 New Tokens Required

| Token | Purpose | Count |
|-------|---------|-------|
| `[DOI_THO]` | Signals couplet-to-couplet duel format (distinct from `[LUC_BAT]` single-couplet) | 1 |
| `<\|linebreak\|>` | Separates lines within input/output blocks | 1 |

**Total new tokens: 2**

Everything else (`<|start|>`, `<|reply|>`, `<|end|>`, `[RHYME:*]`, `[TONE:*]`) is reused from the current system.

### Why a Separate `[DOI_THO]` Tag (Not Reusing `[LUC_BAT]`)?

- `[LUC_BAT]` → Input is 1 line, output is 1 line (single couplet mode)
- `[DOI_THO]` → Input is 2-4 lines (1-2 couplets), output is 2 lines (1 couplet)

Giving the model an explicit signal eliminates ambiguity. The model learns: "When I see `[DOI_THO]`, expect linebreak-separated input and output. When I see `[LUC_BAT]`, expect single-line input and output."

Both formats coexist in the training corpus. Backward compatibility is automatic.

---

## 5. 🔬 Current Architecture Analysis

### 5.1 Token Budget

```
Window=1 example: ~50 tokens  (2 input lines + 2 output lines + tags)
Window=2 example: ~75 tokens  (4 input lines + 2 output lines + tags)
```

block_size=256 → **5× slack even for window=2**. No architecture change needed.

### 5.2 What the 31M Model Already Does Well

| Capability | Accuracy | Needed for Đối Thơ? |
|------------|----------|---------------------|
| Tone pattern (B-T-B-B) | 88.1% | Yes — output couplet must follow tone rules |
| Internal rhyme (pos 6→pos 6) | 49.7% | Yes — within the output couplet |
| Syllable count (6→8) | 71.1% | Yes — output must be 6+8 |

The model already understands Lục Bát structure. The only new thing is:
- Input can be a full couplet, not just one line
- Output is a full couplet, not just one line
- The rhyme chain comes from pos 8 instead of pos 6

### 5.3 What the Model Will Learn That's New

| New Capability | How |
|----------------|-----|
| Input = 2-4 lines separated by `<\|linebreak\|>` | Explicit format in training data |
| Output = 2 lines separated by `<\|linebreak\|>` | Explicit format in training data |
| Chain rhyme from pos 8 of last input line | `[RHYME:X]` always from pos 8 in `[DOI_THO]` mode |
| Semantic continuity across couplets | window=2 training provides multi-couplet context |

---

## 6. 🔴 CRITICAL PREREQUISITE: Fix Control Token Fragmentation

**This must be fixed before any training.**

### The Problem

Currently `[RHYME:ong]` is encoded as 5 separate BPE subword tokens:

```
[ → RHY → ME → : → ong → ]
```

The model sees the rhyme signal scattered across 5 positions. This explains the 50% rhyme accuracy — the model is actually performing remarkably well given this handicap.

### The Fix

`train_bpe.py` already has `build_special_tokens()` which collects all patterns and passes them to `BpeTrainer(special_tokens=...)`. After retraining the tokenizer on the new corpus (which includes `[DOI_THO]` and `<|linebreak|>`), all control tokens must encode as single IDs:

```python
tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")

# These MUST all return 1:
assert len(tok.encode("[DOI_THO]").ids) == 1
assert len(tok.encode("[RHYME:ong]").ids) == 1
assert len(tok.encode("[TONE:BBBTTB]").ids) == 1
assert len(tok.encode("[LINK2:B]").ids) == 1
assert len(tok.encode("[DOIAM:BBBBBBB]").ids) == 1
assert len(tok.encode("<|linebreak|>").ids) == 1
```

### Expected Impact on Single-Couplet Baseline

| Metric | Current (fragmented) | Expected (single-token) |
|--------|---------------------|------------------------|
| R1 Rhyme | 49.7% | 60-75% |
| R2 Tone | 88.1% | 88-92% |
| R3 Syllable | 71.1% | 71% (unaffected) |
| Combined | 45.7% | 55-65% |

---

## 7. 🔄 Implementation Plan

### Phase 0: Prerequisite — Fix Fragmentation (4h Colab)

```
1. Add [DOI_THO] and <|linebreak|> to train_bpe.py SPECIAL_TOKENS
2. Add [END_RHYME:*] is NOT needed — [RHYME:*] works for both pos 6 and pos 8,
   since extraction point depends only on context (pos 6 for [LUC_BAT], pos 8 for [DOI_THO])
3. Generate combined corpus: single-couplet + multi-couplet data
4. Retrain tokenizer → verify all control tokens are single IDs
5. Retrain 31M model from scratch on new tokenizer + combined corpus
```

### Phase 1: Data Preparation (4h coding)

```
1. Create src/preprocess_doi_tho.py
   - Read poems_dataset_clean.csv
   - Group lines into couplets (6-8 pairs for Lục Bát, 7-7 for Thất Ngôn)
   - For each poem with ≥ 2 couplets:
     * Generate window=1 examples: couplet_k → couplet_{k+1}
     * Generate window=2 examples: couplet_k+couplet_{k+1} → couplet_{k+2}
   - Extract [RHYME:X] from pos 8 of last input line's 8-syl line
   - Extract [TONE:XXXXXX] from last input line's 6-syl line
   - Output: resources/doi_tho_corpus.txt

2. Merge with existing single-couplet corpus
   - Keep [LUC_BAT] single-couplet format alongside [DOI_THO] format
   - Model trains on both simultaneously → no catastrophic forgetting

3. Check data statistics
   - How many poems have ≥ 2 couplets?
   - How many total đối thơ examples?
   - If < 500K, consider scraping more long poems (Truyện Kiều gives 3,250 alone)
```

### Phase 2: Two-Stage Training (~2h Colab T4)

Same pipeline as current `train.py`:

```
Stage 1 (all genres):
  python src/train.py --corpus resources/doi_tho_corpus.txt --name doi_tho_stage1_
  → 10K steps, batch=192, LR=3e-4

Stage 2 (Lục Bát only):
  python src/train.py --corpus resources/doi_tho_luc_bat.txt
    --resume checkpoints/doi_tho_stage1_best.pt --name doi_tho_stage2_
  → 5K steps, batch=192, LR=1e-4
```

Training speed: ~3 steps/sec on T4 → Stage 1 ~1h, Stage 2 ~30min.

### Phase 3: Inference (3h coding)

```python
def doi_tho_generate(model, tokenizer, user_lines, temperature=0.7, top_k=50):
    """
    User provides 2n lines (n couplets). Model generates 1 couplet.
    """
    # Keep at most last 2 couplets (4 lines)
    lines = user_lines[-4:]
    
    # Extract tags from last couplet
    last_6 = lines[-2]  # 6-syl line
    last_8 = lines[-1]  # 8-syl line
    
    rhyme_8 = get_rhyme_group(last_8.split()[7])     # pos 8 → chain rhyme
    tone_6  = get_tone_sequence(last_6)[:6]            # tone pattern
    
    # Build prompt
    input_text = " <|linebreak|> ".join(lines)
    prompt = f"<|start|> [DOI_THO] [RHYME:{rhyme_8}] [TONE:{tone_6}] {input_text} <|reply|>"
    
    # Generate autoregressively, stop at <|end|>
    tokens = generate(model, tokenizer, prompt, stop_token="<|end|>", 
                      max_new=64, temperature=temperature, top_k=top_k)
    
    # Parse output: split on <|linebreak|>
    response = tokenizer.decode(tokens).replace("<|end|>", "").strip()
    output_lines = [l.strip() for l in response.split("<|linebreak|>") if l.strip()]
    
    # Handle bad responses (empty, < 2 lines, wrong syllable count)
    if len(output_lines) < 2:
        return retry_with_lower_temp(model, tokenizer, prompt)
    
    return output_lines[:2]  # return exactly 2 lines (1 couplet)
```

**Handling bad responses (14.5% in current Stage 2):**
- Retry with lower temperature (0.5) if output is empty or < 2 lines
- Post-generation truncation to correct syllable counts
- Fallback: if all retries fail, use [LUC_BAT] single-couplet mode

### Phase 4: Evaluation (3h coding)

New metrics specific to đối thơ:

| Metric | What It Measures |
|--------|-----------------|
| Chain rhyme accuracy | Pos 6 of output's 6-syl line rhymes with pos 8 of input's 8-syl line |
| Couplet integrity | Output is exactly 2 lines with 6+8 syllables |
| Per-couplet tone | Output couplet follows B-T-B / B-T-B-B internally |
| Semantic continuity | Embedding similarity between input's last line and output's first line |
| Valid response rate | % of generations that produce usable output (≥ 2 lines, correct syllables) |

---

## 8. ⚠️ Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Control token fragmentation** — rhyme/tone tags split across 5+ BPE tokens | 🔴 CRITICAL | Fix tokenizer BEFORE any training. Verify every control token encodes as single ID. |
| **Insufficient long poems** — corpus may have too few poems with ≥ 2 couplets | 🟡 MEDIUM | Run stats first. Scrape Truyện Kiều (1,627 couplets) and other truyện thơ Nôm if needed. |
| **Response quality** — 14.5% empty/short responses get worse with longer output | 🟡 MEDIUM | Retry with lower temp, post-generation truncation, fallback to single-couplet mode. |
| **31M capacity ceiling** — adding chain rhyme + longer context may saturate model | 🟡 MEDIUM | Two-stage training. If ceiling hit, accept that semantic continuity will be limited at 31M. |
| **Catastrophic forgetting** — model forgets single-couplet skills | 🟢 LOW | Train on mixed corpus (both `[LUC_BAT]` and `[DOI_THO]` formats simultaneously). |
| **Thông vần** — some rhyme pairs differ technically but are accepted in poetry (ơi/ươi, inh/in) | 🟢 LOW | Accept as-is. Vietnamese poetry tradition allows this. Don't overfit to strict rhyme equality. |
| **Block size overflow** — context grows with each duel turn | 🟢 LOW | Inference always truncates to last 2 couplets (4 lines). block_size=256 gives massive headroom. |

---

## 9. 📈 Success Criteria

### Minimum Viable (v2.0)

- [ ] **PREREQUISITE**: Control tokens are single IDs (not BPE fragments)
- [ ] R1 Rhyme ≥ 60% (up from 50% — benefit of fixed tokenizer)
- [ ] Chain rhyme accuracy ≥ 20% (33× better than random 0.6%)
- [ ] Output is 2 lines ≥ 80% of the time (couplet integrity)
- [ ] Per-couplet tone accuracy maintained at ≥ 85%
- [ ] Valid response rate ≥ 80% (up from 85.5% for single-line)
- [ ] All existing single-couplet metrics maintained or improved

### Good (v2.1)

- [ ] Chain rhyme accuracy ≥ 40%
- [ ] 3+ turn duels with coherent nối ý (human evaluation)
- [ ] Semantic continuity score > baseline
- [ ] Single-couplet internal rhyme ≥ 65%

### Excellent (v3.0 — may need larger model)

- [ ] Chain rhyme accuracy ≥ 60%
- [ ] Multi-genre support (Song Thất Lục Bát, Thất Ngôn Tứ Tuyệt)
- [ ] Configurable context window (3-4 couplets for richer nối ý)
- [ ] User-specified couplet count for generation

---

## 10. 📁 Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/preprocess_doi_tho.py` | Generate couplet-to-couplet training data with sliding windows |
| `tests/test_doi_tho.py` | Tests for preprocessing, rhyme extraction, inference |
| `evaluate/eval_doi_tho.py` | Chain rhyme + couplet integrity + semantic continuity metrics |
| `documents/multi_couplet_plan.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `src/train_bpe.py` | Add `[DOI_THO]` and `<\|linebreak\|>` to special tokens |
| `src/preprocess.py` | Keep unchanged (single-couplet) — coexist with new preprocessor |
| `src/tones.py` | Add `get_doi_tho_tags()` — extracts rhyme from pos 8, tone from pos 6 |
| `src/sample.py` | Add `--doi_tho` mode: multi-line input, couplet output, chain display |
| `client/server.py` | Auto-detect input type (1 line vs 2+ lines), route to appropriate handler |
| `client/frontend/src/App.jsx` | Multi-line textarea input, couplet-pair response display |

---

## 11. 🎯 Action Items & Timeline

| # | Task | Est. Effort | Depends On |
|---|------|-------------|------------|
| 1 | **Add `[DOI_THO]` and `<\|linebreak\|>` to tokenizer special tokens** | 1h | None |
| 2 | Run stats: count poems with ≥ 2 couplets in current corpus | 30min | None |
| 3 | **Fix control token fragmentation** — retrain tokenizer + verify single IDs | 2h | #1 |
| 4 | Build `preprocess_doi_tho.py` — sliding window data generator | 3h | #2 |
| 5 | Generate `doi_tho_corpus.txt` (combined single + multi format) | 30min (compute) | #4 |
| 6 | Retrain tokenizer on combined corpus | 30min | #5 |
| 7 | **Two-stage training** (Stage 1: all genres, Stage 2: Lục Bát) | 2h (Colab T4) | #5, #6 |
| 8 | Implement `doi_tho_generate()` in sample.py + server.py | 3h | #7 |
| 9 | Frontend: multi-line input, couplet-pair display | 2h | #8 |
| 10 | Evaluation script for chain rhyme + couplet integrity | 3h | #7 |
| 11 | Documentation & report | 1h | #10 |

**Total estimated effort:** ~3-4 days (2 days with parallel work)

**Parallelizable:**
- #1, #2 can start immediately
- #8, #9, #10 can start once #7 is done
- Frontend (#9) can be built against mock API before model is ready

---

## 12. 💡 Summary

### The Strategy in One Sentence

**Train the model to receive 1-2 couplets and output 1 couplet, using sliding windows over existing poems, with chain rhyme extracted from pos 8 of the last input line.**

### Why This Beats the Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Sliding pair (CHOSEN)** | Minimal new tokens (2), reuses existing format, handles any input length, backward compatible | Model only has 2-couplet context | **Best for 31M** |
| Line-by-line chaining | No new format | Loses couplet context, breaks rhyme chain, 8→6 direction unsupported | ❌ Doesn't work |
| Full-poem single pass | Single generation call | New format, `<\|end_poem\|>` needed, inflexible input length, needs larger model | Overengineered |

### Key Design Decisions

1. **`[DOI_THO]` as separate tag** — clean separation from `[LUC_BAT]` single-couplet mode
2. **Window=1 + window=2** — handles both first turn and continued duel
3. **Rhyme from pos 8** — directly encodes the chain rhyme rule into the training data
4. **Mixed corpus** — single-couplet and multi-couplet examples coexist, no forgetting
5. **Inference truncate to 2 couplets** — consistent with training, prevents context overflow
6. **No `<\|end_poem\|>`** — reuse `<\|end\|>`, simpler, proven to work

### The One Non-Negotiable Prerequisite

**Fix control token fragmentation before anything else.** If `[RHYME:ong]` is still 5 BPE tokens, the chain rhyme signal in đối thơ will be just as weak as the internal rhyme signal is today. The tokenizer fix alone should boost single-couplet rhyme from 50% to 60-75%, giving the multi-couplet system a solid foundation to build on.
