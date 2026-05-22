# 📜 Multi-Couplet Generation — Comprehensive Plan

> **Status:** Research & Design Phase — Updated for 31M model
> **Target:** v2.0 (PoetryDuelGPT 31M — Qwen migration skipped)
> **Current model:** Stage 2 — R1:Rhyme 50% | R2:Tone 88% | R3:Syllable 71% | Combined: 46%
> **Author:** 2026-05-22

---

## 1. 🎯 What the User Wants

```
User (2 dòng Lục Bát):
  kiều nhi phận mỏng như tờ
  một lời đã lỗi tóc tơ với chàng!

Model (2 dòng tiếp theo):
  gặp cơn gia biến lạ dường
  bán mình nó đã tìm đường cứu cha
```

The model receives a **multi-line prompt** (one full Lục Bát couplet or more) and produces a **multi-line response** continuing the poem. This isn't just single-couplet response generation — it's **poem continuation** where:

1. The user's last line's final syllable (`chàng` → rhyme group `ang`) must rhyme with syllable 6 of the model's first response line (`dường` → rhyme group `ương` — should be `ang` ideally, but poets take creative liberties)
2. The generated lines must themselves form a valid couplet (internal rhyme, tone pattern)
3. Across stanzas, the poem remains semantically coherent

---

## 2. 📐 Vietnamese Poetry Structure & What Multi-Couplet Unlocks

### 2.1 The Full Lục Bát Chain

```
Stanza 1:  Line 1 (6 syl) ────┐
           Line 2 (8 syl) ──┐ │  internal rhyme (pos 6)
                            │ │
Stanza 2:  Line 3 (6 syl) ─┘ │  END RHYME: pos 6 of line 3
           Line 4 (8 syl)     │  rhymes with pos 8 of line 2
                              │
           ...continues...
```

**The currently-missing END RHYME rule:** Position 6 of the NEXT couplet's first line must rhyme with position 8 of the current couplet's last line. This is the **chain rhyme mechanism** that links stanzas in Lục Bát poetry. Single-couplet training **cannot learn this** because the data never contains adjacent couplets.

### 2.2 Thất Ngôn Bát Cú (8-line poem)

```
Line 1 (7 syl) ── đối âm ── Line 2 (7 syl)
Line 3 (7 syl) ── đối âm ── Line 4 (7 syl)
Line 5 (7 syl) ── đối âm ── Line 6 (7 syl)
Line 7 (7 syl) ── đối âm ── Line 8 (7 syl)
```

End-rhyme links: lines 1-2-4-6-8 share the same end rhyme. Lines 3-4, 5-6, 7-8 each form parallel couplets. Multi-couplet generation is **essential** for this form.

### 2.3 Song Thất Lục Bát (4-line stanza repeating)

```
7 syl ── đối âm ── 7 syl
6 syl ──────────── 8 syl
```

Currently, `preprocess.py` splits these into individual pairs. Multi-couplet would keep the full 4-line stanza intact.

---

## 3. 🔬 Current Architecture Deep Dive

### 3.1 Single-Couplet Data Format (Current)

```
<|start|> [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] prompt_6_syl <|reply|> response_8_syl <|end|>
```

- One pair per line in `poetry_corpus.txt` (942K lines)
- Block size: 256 tokens
- Model never sees adjacent couplets
- End-rhyme between stanzas is **impossible to learn**

### 3.2 What a Typical Training Pair Looks Like Token-Wise

```
<|start|> [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng <|reply|> Phất phơ dưới ngọn nắng hồng ban mai <|end|>

Rough token count: ~25-35 tokens per couplet pair
```

With block_size=256, even 4-couplet sequences (~35 × 4 = 140 tokens) easily fit. **No architecture change needed.**

### 3.3 The Generation Flow (sample.py / server.py)

```
User prompt → auto_tag() → encode → autoregressive loop → decode → response
```

The generation loop samples one token at a time, stops at `<|end|>`. Currently produces exactly one couplet's reply line.

---

## 3.5 ⚡ 31M Model Impact Assessment (Qwen Skipped)

### What stays the same

- **All data format changes** — identical whether 31M or Qwen
- **All code changes** (preprocess, sample, server, tokenizer) — identical
- **All control tokens** — identical
- **block_size=256** — unchanged, fits 4-6 couplets either way
- **Training pipeline** — `src/train.py` works as-is with new corpus

### What's different (and what to do about it)

| Concern | Severity | Why | Fix |
|---------|----------|-----|-----|
| **Control token fragmentation** | 🔴 CRITICAL | `[RHYME:ong]` = 5 BPE subwords → model can't learn rhyme conditioning. This is the #1 reason for 50% rhyme accuracy. Multi-couplet adds `[END_RHYME:X]` which would have the same problem. | **Fix BEFORE multi-couplet:** re-run `train_bpe.py` with all rhyme/tone/doi_am patterns as single special tokens. This alone should boost R1 from 50% → 60-70%. |
| **Model capacity (31M params)** | 🟡 MEDIUM | End-rhyme is an additional rule for the model to learn, on top of internal rhyme, tone pattern, and syllable count. 31M may saturate. | Two-stage training: Stage 1 on single-couplet (master basics) → Stage 2 on multi-couplet (add end-rhyme). If still fails, chained-generation prototype works without learning end-rhyme. |
| **Empty/short responses** | 🟡 MEDIUM | Stage 2 already has 5.8% empty + 3.5% 1-syllable + 5.2% 2-syllable responses. Longer sequences may increase this. | Scheduled sampling already in `train.py`. Post-generation truncation. Temperature 0.7. |
| **Training speed** | 🟢 ADVANTAGE | 31M trains ~3 steps/sec on T4 vs Qwen's ~0.5 steps/sec. Stage 1 (10K steps) ≈ 1h. Stage 2 (5K steps) ≈ 30min. | Fast iteration cycle — can experiment with different data formats quickly. |
| **Vietnamese vocabulary** | 🟡 MEDIUM | 11K BPE tokens vs Qwen's 150K+. Limited vocabulary means more subword fragmentation of poetic words. | Accept as-is. The model already generates coherent Vietnamese despite limited vocab. Multi-couplet doesn't change this. |

### The Prerequisite: Fix Control Token Fragmentation

This is the single highest-impact fix before any multi-couplet work. The current tokenizer encodes `[RHYME:ong]` as 5 separate BPE tokens. The model sees:

```
[ RHY ME : ong ]
```

Instead of a single token `[RHYME:ong]`. This means:
- The rhyme conditioning signal is spread across 5 token positions
- The attention mechanism can't easily associate "RHYME:ong" with "position 6 should rhyme with ong"
- This explains the 50% rhyme accuracy — the model is doing surprisingly well given the handicap

**The fix is already in the codebase.** `train_bpe.py` has `build_special_tokens()` which collects all `[RHYME:*]` patterns and passes them as `special_tokens` to the BPE trainer. The problem is the current tokenizer was trained before this was fully implemented, or the special tokens weren't being passed correctly.

**Verify the fix:** After retraining, check:
```python
tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")
assert len(tok.encode("[RHYME:ong]").ids) == 1  # MUST be 1, not 5
assert len(tok.encode("[TONE:BBBTTB]").ids) == 1
assert len(tok.encode("[END_RHYME:ang]").ids) == 1
```

**Expected impact on single-couplet metrics (pre-multi-couplet):**
| Metric | Current (fragmented) | Expected (single-token) |
|--------|---------------------|------------------------|
| R1 Rhyme | 49.7% | 60-75% |
| R2 Tone | 88.1% | 88-92% |
| R3 Syllable | 71.1% | 71% (unaffected) |
| Combined | 45.7% | 55-65% |

This fixed baseline is the foundation multi-couplet builds on.

---

## 4. 🏗️ Implementation Plan — Two Approaches

### Approach A: "Poem Continuation" (Recommended for v2.0)

The user provides a full couplet (or multiple lines), the model generates the next couplet(s). This is what the user's example shows and is the most practical use case.

**Data format:**
```
<|start|> [POEM:LUC_BAT] [RHYME:ong] [TONE:BBBTTB]
  Thân em như chẽn lúa đòng <|linebreak|> Phất phơ dưới ngọn nắng hồng ban mai <|reply|>
  Thân em như hạt mưa sa <|linebreak|> Hạt vào đài các hạt ra ruộng cày <|end_poem|>
```

### Approach B: "Full Poem Generation" (Ambitious, v3.0+)

Generate an entire Lục Bát or Thất Ngôn bát cú poem from a title/topic. The model generates all stanzas autoregressively with proper inter-stanza rhyming.

**Data format:**
```
<|start|> [FULL:LUC_BAT] [TOPIC:nắng hồng]
  Thân em như chẽn lúa đòng <|linebreak|>
  Phất phơ dưới ngọn nắng hồng ban mai <|linebreak|>
  Thân em như hạt mưa sa <|linebreak|>
  Hạt vào đài các hạt ra ruộng cày <|end_poem|>
```

### Decision: Start with Approach A

Approach A (Poem Continuation) is smaller scope, directly solves the user's need, and enables end-rhyme evaluation. Approach B can be built on top of A later.

---

## 5. 📝 Detailed Implementation: Approach A (Poem Continuation)

### 5.1 New Control Tokens

Add to `train_bpe.py` SPECIAL_TOKENS:

```python
# New multi-couplet tokens (after existing 343 tokens)
"[POEM:LUC_BAT]",       # 343 — multi-couplet Lục Bát continuation
"[POEM:THAT_NGON]",     # 344 — multi-couplet Thất Ngôn continuation
"[END_RHYME:X]",        # 345+ — end-rhyme group from last line of input
"<|linebreak|>",        # last+1 — separates lines within a poem
"<|end_poem|>",         # last+2 — poem-level stop token
```

**Why `[END_RHYME:X]` instead of just `[RHYME:X]`:**

- `[RHYME:X]` = internal rhyme (pos 6 of prompt → pos 6 of response). Used for single couplet.
- `[END_RHYME:X]` = end-rhyme (pos 8 of last input line → pos 6 of first output line). The chain rhyme linking stanzas.

Both are needed because they serve different positions. In a multi-couplet context:

```
Input couplet:   Line 1 (6 syl) ...pos6=syllable_A
                 Line 2 (8 syl) ...pos6=syllable_B  pos8=syllable_C

Model must generate:
                 Line 3 (6 syl) ...pos6 rhymes with syllable_C (end-rhyme!) and follows B-T-B
                 Line 4 (8 syl) ...pos6 rhymes with pos6 of line 3 (internal rhyme) and follows B-T-B-B
```

So the prompt needs:
```
[POEM:LUC_BAT] [RHYME:B] [TONE:BBBBBB] [END_RHYME:C]
```

Where:
- `[RHYME:B]` = rhyme of last line's position 6 (for internal rhyme in the response's 2nd line)
- `[TONE:BBBBBB]` = tone pattern of last input line
- `[END_RHYME:C]` = rhyme of last line's position 8 (for end-rhyme linking to response's 1st line position 6)

### 5.2 Training Data Format

Each training example = one poem split into (input_couplet → continuation_couplets):

```
<|start|> [POEM:LUC_BAT] [RHYME:ong] [TONE:BBTBBB] [END_RHYME:ai]
  Thân em như chẽn lúa đòng <|linebreak|> Phất phơ dưới ngọn nắng hồng ban mai <|reply|>
  Thân em như hạt mưa sa <|linebreak|> Hạt vào đài các hạt ra ruộng cày <|end_poem|>
```

Or with multiple continuation couplets:
```
<|start|> [POEM:LUC_BAT] [RHYME:im] [TONE:BBBTBB] [END_RHYME:au]
  Đói lòng ăn nửa trái sim <|linebreak|> Uống lưng bát nước đi tìm người thương <|reply|>
  Người thương ơi hỡi người thương <|linebreak|> Đi đâu mà để buồn vương trong lòng <|linebreak|>
  Trong lòng nặng những chờ mong <|linebreak|> Nhớ ai da diết mênh mông tháng ngày <|end_poem|>
```

### 5.3 Token Budget Analysis

```
Block size: 256 tokens
Max per couplet: ~35 tokens (tags + 14 syllables + line separators + special tokens)
4 couplets: ~140 tokens
5 couplets: ~175 tokens
10 couplets: ~350 tokens ← exceeds block_size=256

Safe max: 6-7 couplets (still within 256 if we're conservative)
```

**Conclusion:** block_size=256 is fine for 2-6 couplet sequences. For longer poems, we'd need block_size=512, but 256 works for v2.0.

### 5.4 Modified preprocess.py — Multi-Couplet Mode

The key change: instead of splitting poems into individual couplet pairs, keep poems as multi-couplet sequences with input/output split.

```python
def make_multi_couplet_poem(lines, genre, num_input_couplets=1):
    """
    Convert a full poem into training examples:
      input = first N couplets  →  output = remaining couplets
    
    For Lục Bát: each couplet is 2 lines (6-syl + 8-syl)
    For Thất Ngôn: each couplet is 2 lines (7-syl + 7-syl)
    """
    rule = GENRE_RULES[genre]
    p_syl = rule["prompt_syl"]
    r_syl = rule["reply_syl"]
    lines_per_couplet = 2
    
    examples = []
    
    # Must have at least (num_input + 1) couplets
    min_couplets = num_input_couplets + 1
    total_couplets = len(lines) // lines_per_couplet
    
    if total_couplets < min_couplets:
        return []
    
    # Build input from first N couplets
    input_lines = lines[:num_input_couplets * lines_per_couplet]
    output_lines = lines[num_input_couplets * lines_per_couplet:]
    
    # Extract control tokens from LAST line of input
    last_input_line = input_lines[-1]
    
    # Internal rhyme: pos 6 of last input line
    rhyme_tag, tone_tag = get_luc_bat_tags(last_input_line)
    
    # End-rhyme: pos 8 of last input line (the chain link)
    end_rhyme = get_rhyme_group(last_input_line.split()[7]) if len(last_input_line.split()) >= 8 else ""
    end_rhyme_tag = f"[END_RHYME:{end_rhyme}]" if end_rhyme else ""
    
    # Build the formatted string
    if genre == "lục_bát":
        poem_tag = "[POEM:LUC_BAT]"
        extras = f"{rhyme_tag} {tone_tag} {end_rhyme_tag}".strip()
    else:
        poem_tag = "[POEM:THAT_NGON]"
        link2, doi_am = get_that_ngon_tags(last_input_line)
        extras = f"{link2} {doi_am}".strip()
    
    # Input part (the user's lines)
    input_str = " <|linebreak|> ".join(input_lines)
    
    # Output part (the model's lines)
    output_str = " <|linebreak|> ".join(output_lines)
    
    return f"<|start|> {poem_tag} {extras} {input_str} <|reply|> {output_str} <|end_poem|>"
```

### 5.5 Dataset Assembly Strategy

Rather than pre-split inputs/outputs in the preprocessor, **do it in the dataset class** during training. This is more token-efficient and allows dynamic input/output ratios.

**Alternative — Dynamic Windowing in Dataset:**

The `PoetryDataset` already does random windows. If the flat token stream contains multi-couplet poems (joined with `<|linebreak|>` and separated by `<|end_poem|>`), the model naturally learns the multi-couplet structure through next-token prediction.

But this doesn't teach the `<|reply|>` mechanism for multi-couplet. Better approach:

1. **Training:** Each poem → N× (first_k_couplets → remaining_couplets) examples with k=1,2,3...
2. **Inference:** Encode user's N lines with `<|reply|>`, generate until `<|end_poem|>`

### 5.6 Training Examples Per Poem

For a 4-couplet Lục Bát poem (8 lines):
```
Example 1: couplet_1 → couplets_2-4   (1 in, 3 out)
Example 2: couplets_1-2 → couplets_3-4 (2 in, 2 out)
Example 3: couplets_1-3 → couplet_4    (3 in, 1 out)
```

This creates ~3× training examples from existing data and teaches the model variable-length continuation.

For a 20-couplet poem:
```
~19 examples with progressive input length
Many of these will exceed block_size=256 → trim to fit
```

### 5.7 Control Token Injection at Generation Time

The server/sample inference code must handle multi-line input:

```python
def auto_tag_multi(lines_text: str) -> str:
    """
    User input: "kiều nhi phận mỏng như tờ\nmột lời đã lỗi tóc tơ với chàng!"
    Output: "[POEM:LUC_BAT] [RHYME:ơ] [TONE:BBTTBB] [END_RHYME:ang]
             kiều nhi phận mỏng như tờ <|linebreak|> một lời đã lỗi tóc tơ với chàng! <|reply|>"
    """
    lines = [l.strip() for l in lines_text.strip().split('\n') if l.strip()]
    
    if not lines:
        return lines_text
    
    # Detect genre from syllable count pattern
    if len(lines) == 2:
        n1, n2 = len(lines[0].split()), len(lines[1].split())
        if n1 == 6 and n2 == 8:
            genre = "lục_bát"
        elif n1 == 7 and n2 == 7:
            genre = "thất_ngôn"
        else:
            genre = "lục_bát"  # default
    else:
        genre = "lục_bát"
    
    last_line = lines[-1]
    
    if genre == "lục_bát":
        rhyme, tone = get_luc_bat_tags(last_line)
        syls = last_line.split()
        end_rhyme = get_rhyme_group(syls[7]) if len(syls) >= 8 else ""
        end_tag = f"[END_RHYME:{end_rhyme}]" if end_rhyme else ""
        
        extras = f"{rhyme} {tone} {end_tag}".strip()
        tag = f"[POEM:LUC_BAT] {extras}" if extras else "[POEM:LUC_BAT]"
    else:
        link2, doi_am = get_that_ngon_tags(last_line)
        extras = f"{link2} {doi_am}".strip()
        tag = f"[POEM:THAT_NGON] {extras}" if extras else "[POEM:THAT_NGON]"
    
    input_str = " <|linebreak|> ".join(lines)
    return f"<|start|> {tag} {input_str} <|reply|>"
```

### 5.8 Modified Generation Loop

```python
@torch.no_grad()
def generate_multi(model, tokenizer, prompt, max_new=128, temperature=0.75,
                    top_k=50, top_p=0.92, device="cpu"):
    """
    Generate multi-couplet continuation.
    Stops on <|end_poem|> or max_new tokens.
    """
    end_poem_id = tokenizer.token_to_id("<|end_poem|>")
    end_id = tokenizer.token_to_id("<|end|>")  # fallback
    pad_id = tokenizer.token_to_id("<|pad|>")
    
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    new_tokens = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        logits[:, pad_id] = float("-inf")
        
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        
        if top_p is not None:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[:, sorted_idx[mask]] = float("-inf")
        
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        
        if next_id == end_poem_id or next_id == end_id:
            break
        
        new_tokens.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    
    return new_tokens
```

### 5.9 Response Parsing

```python
def parse_multi_response(tokens, tokenizer):
    """Parse generated tokens into structured couplets."""
    text = tokenizer.decode(tokens)
    # Remove control tokens
    text = text.replace("<|end_poem|>", "").replace("<|end|>", "").strip()
    # Split into lines
    lines = [l.strip() for l in text.split("<|linebreak|>") if l.strip()]
    # Group into couplets
    couplets = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            couplets.append((lines[i], lines[i+1]))
        elif i < len(lines):
            couplets.append((lines[i], ""))
    return couplets
```

---

## 6. 📊 New Evaluation Metrics for Multi-Couplet

### 6.1 End-Rhyme Rule (NEW — currently not evaluable)

```
Rule: pos 6 of couplet_k's first line must rhyme with pos 8 of couplet_{k-1}'s last line

Example:
  Couplet 1, Line 2 (pos 8): "chàng" → rhyme group "ang"
  Couplet 2, Line 1 (pos 6): "dường" → rhyme group "ương"
  Match? "ang" vs "ương" → NO (acceptable poetic license, but strict check fails)

Evaluation:
  end_rhyme_accuracy = correct_matches / total_couplet_transitions
```

### 6.2 Inter-Stanza Coherence Score (NEW)

```
Semantic flow: does couplet_{k+1} logically follow couplet_k?
- Uses embedding similarity between last line of k and first line of k+1
- Low similarity → topic drift → poor coherence
- Very high similarity → repetition → poor coherence  
- Moderate similarity → good continuation → good coherence
```

### 6.3 Full-Poem Metrics

| Metric | What | Current (single) | Target (multi) |
|--------|------|------------------|-----------------|
| End-rhyme accuracy | Pos 6 of next line rhymes with pos 8 of prev | N/A | 40-60% |
| Internal rhyme accuracy | Within each generated couplet | 58.4% | 58%+ (same) |
| Tone pattern accuracy | B-T-B / B-T-B-B per couplet | 87.5% | 87%+ (same) |
| Syllable count accuracy | 6-8 per couplet | 78.0% | 78%+ (same) |
| Inter-stanza coherence | Semantic flow between stanzas | N/A | TBD |
| Correct stop rate | Model stops at poem boundary | 85% (on <\|end\|>) | 80%+ (on <\|end_poem\|>) |

---

## 7. 🔄 Migration Path: v1.0 → v2.0 (31M Model)

### ⚡ Prerequisite: Fix R1+R2 Control Token Fragmentation

**Before** any multi-couplet work, fix this: the current control tokens are BPE-fragmented.
`[RHYME:ong]` = 5 BPE subword tokens, not one. Same for `[TONE:BBBTTB]`.
This is the #1 cause of low rhyme accuracy (50%). Multi-couplet adds `[END_RHYME:X]` —
which would be equally fragmented and equally ineffective unless fixed first.

**Fix:** Make all rhyme/tone/doi_am/link2 tokens **single special tokens** (like `[LUC_BAT]` is today).
`train_bpe.py` already has the code for this — the `build_special_tokens()` function
collects all patterns and adds them as special tokens. The issue is likely that
the current tokenizer was trained on single-couplet data only. Re-running with the
multi-couplet corpus (which includes `[END_RHYME:*]`) will fix both.

**Effort:** 30 min to add tokens + 4h Colab to retrain from scratch.
**Expected gain:** R1 rhyme 50% → 60-70%, R2 tone stays at ~88%.

### Phase 1: Data Preparation (1 day)

```
1. Add new special tokens to train_bpe.py
   - [POEM:LUC_BAT], [POEM:THAT_NGON]
   - [END_RHYME:*] groups (same set as [RHYME:*], ~141 tokens)
   - <|linebreak|>, <|end_poem|>
   - CRITICAL: ensure [RHYME:*], [TONE:*], [DOIAM:*], [LINK2:*] are SINGLE tokens

2. Create multi-couplet corpus generator (new file: src/preprocess_multi.py)
   - Reads poems_dataset_clean.csv
   - For each poem: generates (1→N), (2→N), (3→N) examples
   - Outputs: data/poetry_corpus_multi.txt

3. Regenerate tokenizer with ALL special tokens as single IDs
   python src/train_bpe.py --corpus data/poetry_corpus_multi.txt
```

### Phase 2: Training (~6h on Colab T4)

```
4. Two-stage training on PoetryDuelGPT 31M
   Stage 1: python src/train.py --corpus data/poetry_corpus_multi.txt --name multi_stage1_
            → 10K steps, all genres, batch=192
   Stage 2: python src/train.py --corpus data/corpus_luc_bat_multi.txt
            --resume checkpoints/multi_stage1_best.pt --name multi_stage2_
            → 5K steps, Lục Bát only, LR=1e-4
   
   Training speed: ~3 steps/sec on T4 → Stage 1 (~1h) + Stage 2 (~30min)
   The 31M model trains FAST — this is an advantage of staying small.
```

### Phase 3: Inference & UI (1 day)

```
5. Implement chained-generation prototype FIRST (zero retraining — Section 12)
6. Update src/sample.py — multi-line input, multi-couplet output
7. Update client/server.py — new /chat_multi endpoint
8. Update frontend — multi-line input box, couplet-display response
```

### Phase 4: Evaluation (1 day)

```
9. Evaluate multi-couplet rules (end-rhyme, coherence, stop signal)
10. Compare single vs multi on same 173 prompts
11. Write report in documents/multi_couplet_evaluation.md
```

---

## 8. 🔧 Backward Compatibility

All existing single-couplet functionality must remain intact:

```
Single-couplet format:     <|start|> [LUC_BAT] [RHYME:X] [TONE:XXXXXX] prompt <|reply|> reply <|end|>
Multi-couplet format:      <|start|> [POEM:LUC_BAT] [RHYME:X] [TONE:XXXXXX] [END_RHYME:Y] input_lines <|reply|> output_lines <|end_poem|>
```

- Different genre tags (`[LUC_BAT]` vs `[POEM:LUC_BAT]`) ensure no ambiguity
- Model can be trained on BOTH formats simultaneously (mixed corpus)
- Inference decides which tag to use based on whether user input has multiple lines

A user typing a single 6-syllable line gets single-couplet mode. A user typing 2+ lines gets multi-couplet mode. The `auto_tag_multi()` function handles this routing.

---

## 9. ⚠️ Risks & Mitigations (31M-Specific)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Control token fragmentation** (R1/R2 tags are 5 BPE subwords, not single IDs) | **High** | **High** | **PREREQUISITE: fix before multi-couplet.** Ensure `train_bpe.py` makes all `[RHYME:*]`, `[TONE:*]`, `[END_RHYME:*]` single special tokens. Without this, multi-couplet end-rhyme will be as weak as current internal rhyme. |
| **31M capacity ceiling** — model can't learn end-rhyme ON TOP of internal rhyme + tone + syllable | Medium | High | Two-stage training (Stage 1: single-couplet mastery → Stage 2: multi-couplet). If ceiling hit, fall back to chained-generation approach (Section 12) which requires no new model capability. |
| Multi-couplet data is sparse (few poems have 4+ valid couplets) | Medium | Medium | Generate multiple sliding-window examples per poem (3-19 examples per long poem). Truyện Kiều alone has 3,254 lines = 1,627 couplets. |
| End-rhyme is hard for model to learn | Medium | Medium | Special token `[END_RHYME:X]` as strong conditioning signal. Even 30% end-rhyme accuracy is 50× better than random (0.6%). |
| Generated poems drift semantically | High | Medium | Shorter max tokens, temperature=0.7, nucleus sampling top_p=0.9 |
| `<\|linebreak\|>` token breaks BPE subword boundaries | Low | Low | Add as special token, verify single-ID encoding |
| Long sequences overflow block_size=256 | Low | Low | 4 couplets ≈ 140 tokens. Truncate to last 256 during generation. Keep training sequences ≤ 6 couplets. |
| Model doesn't learn to emit `<\|end_poem\|>` reliably | Medium | Medium | Ensure EVERY poem in training data ends with `<\|end_poem\|>`. Fallback to `<\|end\|>` in generation inference. |
| **Empty/short responses increase** (31M model already has 5.8% empty + 3.5% 1-syl responses in Stage 2) | Medium | Medium | Scheduled sampling already in train.py (decays teacher forcing). Lower temperature (0.7) during inference. Post-generation truncation to 8 syl (R3 fix). |
| **Catastrophic forgetting** of single-couplet skills when training on multi-couplet | Low | Medium | Include BOTH single-couplet AND multi-couplet examples in training data (mixed corpus). Or do 2-stage: single first, then multi. |

---

## 10. 📈 Success Criteria

### Minimum Viable (v2.0) — 31M model
- [ ] **PREREQUISITE**: Control tokens fixed (RHYME/TONE/END_RHYME as single IDs, not BPE fragments)
- [ ] R1 Rhyme ≥ 60% (up from 50% due to fixed control tokens)
- [ ] Model generates 2+ couplets when given multi-line input
- [ ] Generated couplets maintain per-couplet tone pattern (R2 ≥ 85%)
- [ ] End-rhyme accuracy > 20% (33× better than random 0.6%)
- [ ] Correct stop on `<|end_poem|>` > 60%
- [ ] Plus: chained-generation prototype delivers immediate UX (Section 12)

### Good (v2.1)
- [ ] End-rhyme accuracy > 35%
- [ ] 3+ couplets coherent continuation
- [ ] Inter-stanza semantic coherence > baseline
- [ ] Internal rhyme (R1) maintained at ≥ 55%
- [ ] All existing single-couplet metrics maintained or improved

### Excellent (v3.0 — likely needs Qwen or larger model)
- [ ] End-rhyme accuracy > 60%
- [ ] Full Thất Ngôn Bát Cú (8 lines) generation
- [ ] Multiple poetic forms (Song Thất Lục Bát, Tứ Tuyệt)
- [ ] User-specified couplet count (`[NUM_COUPLETS:3]`) — this may require model capacity beyond 31M

---

## 11. 📁 Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/preprocess_multi.py` | Generate multi-couplet training data from poems |
| `tests/test_multi_couplet.py` | Tests for multi-couplet preprocessing, generation, evaluation |
| `evaluate/eval_multi_rules.py` | End-rhyme + inter-stanza evaluation metrics |
| `documents/multi_couplet_plan.md` | This document |
| `documents/multi_couplet_evaluation.md` | Post-training evaluation report |

### Modified Files

| File | Changes |
|------|---------|
| `src/train_bpe.py` | Add new special tokens (`[POEM:*]`, `[END_RHYME:*]`, `<\|linebreak\|>`, `<\|end_poem\|>`) |
| `src/preprocess.py` | Add multi-couplet mode alongside existing single-couplet mode |
| `src/sample.py` | Multi-line input handling, multi-couplet generation, response parsing |
| `client/server.py` | New `/chat_multi` endpoint, auto-detect input type |
| `src/dataset.py` | Optional `MultiCoupletDataset` class for structured I/O pairs |
| `src/tones.py` | Add `get_end_rhyme_tag()` function |
| `finetune/finetune.py` | Support multi-couplet corpus flag |
| `client/frontend/src/App.jsx` | Multi-line input textarea, couplet-formatted output display |

---

## 12. 🧪 Quick Prototype (Without Training)

Even without retraining, we can simulate multi-couplet generation by **chaining single-couplet calls**:

```python
def generate_multi_simulated(model, tokenizer, lines, num_couplets=3):
    """Chain single-couplet generation to simulate multi-couplet output."""
    current_lines = list(lines)
    
    for _ in range(num_couplets):
        # Last line as prompt
        prompt = current_lines[-1]
        # Generate next line (single-couplet mode)
        tagged = auto_tag(prompt)
        next_line_tokens = generate(model, tokenizer, tagged, ...)
        next_line = tokenizer.decode(next_line_tokens).replace("<|end|>", "").strip()
        current_lines.append(next_line)
    
    # Group into couplets
    return group_into_couplets(current_lines)
```

This wouldn't learn end-rhyme but would demonstrate the UX and test the response formatting. Worth implementing as a **v1.0 feature** to gather user feedback before investing in full multi-couplet training.

---

## 13. 🎯 Action Items & Timeline

| # | Task | Est. Effort | Depends On | Owner |
|---|------|-------------|------------|-------|
| 1 | Add multi-line input UI to frontend | 2h | None | Frontend |
| 2 | Implement chained-generation prototype | 3h | #1 | Backend |
| 3 | Design & add new control tokens to tokenizer | 2h | None | Backend |
| 4 | Build `preprocess_multi.py` data generator | 4h | #3 | Backend |
| 5 | Generate `poetry_corpus_multi.txt` corpus | 1h (compute) | #4 | Backend |
| 6 | Retrain tokenizer with new special tokens | 30min | #5 | Backend |
| 7 | Train 31M PoetryDuelGPT on multi data (2-stage) | 2-6h (Colab) | #5, #6 | ML |
| 8 | Update inference code (sample.py, server.py) | 4h | #7 | Backend |
| 9 | Evaluation script for end-rhyme | 3h | #7 | ML |
| 10 | Train PoetryDuelGPT on multi data | 2-6h (Colab) | #5, #6 | ML |
| 10b | Fix R1/R2 control token fragmentation (prerequisite) | 4h (Colab) | #3 | ML |
| 11 | Documentation & report | 2h | #9 | All |

**Total estimated effort:** ~3-4 days (parallelizable to ~2 days with 2 people)

---

## 14. 💡 Summary

Multi-couplet generation is **highly feasible** within the current architecture:

1. **No architecture change needed** — block_size=256 fits 4-6 couplets
2. **Same training pipeline** — Qwen2.5-1.5B QLoRA with different data format
3. **Strong backward compatibility** — single-couplet still works via different genre tags
4. **Clear incremental path** — prototype with chaining → train with multi data → full poem generation
5. **Unlocks end-rhyme evaluation** — the one Lục Bát rule we currently can't measure

The key insight is that the `<|reply|>` mechanism scales naturally from single-couplet to multi-couplet — it's the same autoregressive loop, just with a different stop token (`<|end_poem|>` instead of `<|end|>`) and line separators (`<|linebreak|>`) between lines.

**Recommended first step:** Implement the chained-generation prototype (Section 12) immediately — it's 5 hours of work with zero retraining, gives immediate value, and validates the UX before the heavy ML investment.
