# 🔧 v4.2 Roadmap — Semantic Quality Reinforcement

> v4.1 shipped: 76% all-5-pass, 84% rhyme, 92% tone, 90% Trầm-Bổng.
> **Problem**: Metrics are excellent but poems are semantically incoherent — words
> unrelated, BPE fragments, repetition, nonsensical phrases. v3 was better.
>
> **Root cause**: Structural proxy metrics (rhyme, tone, syllable) ≠ poetic quality.
> The model optimizes for what's measured. v4.1 added constraints (beam rhyme masking,
> Trầm-Bổng) without adding capacity or semantic objectives. Result: rule-compliant
> word salad.
>
> **v4.2 strategy**: Keep all v4.1 structural gains. Fix the generation pipeline first
> (no retrain, immediate impact), then add semantic objectives at training time.
> Measure what matters. No external models — fix the 31M model from within.

---

## 📊 Target: What Success Looks Like

v4.2 is a **SUCCESS** if, after Phase 1, a blind human review of 20 outputs shows:

| Criterion | v4.1 (now) | v4.2 Target | Measurement |
|-----------|------------|-------------|-------------|
| Nonsense rate | ~40% of outputs are word salad | < 15% (3/20 max) | Human labels yes/no |
| BPE artifacts visible | ~30% have truncated syllables | < 5% (1/20 max) | Human inspection |
| Adjacent word repeats | "con con con", "gom gom" common | < 5% (1/20 max) | Regex `\b(\w+)\s+\1\b` |
| Reads like Vietnamese | Often no | > 75% of outputs (15/20) | Human judgment 1-5, avg ≥ 3.0 |
| Format correctness | Broken (see RC8) | All prompts use correct format | Automated format check |

v4.2 is a **FAILURE** if after Phase 1 ANY of these are true:
- Nonsense rate still > 25% (more than 5/20 outputs are word salad)
- "con con con"-style triple-repeats still appear
- BPE fragments like "người v", "đong nà" still common (>3/20)

**If Phase 1 fails these criteria, STOP. Do not proceed to Phase 2 or 3.** Fix the
generation pipeline first. Retraining will not fix a broken generation path.

If Phase 1 passes: proceed to Phase 2 (reranking + metrics), then Phase 3 (retrain)
with the refined loss. Phase 3 success: human quality avg ≥ 3.5/5 on 20 blind samples.

---

## 🔍 Problem Catalog — 8 Root Causes

### RC1: Structural Metrics Don't Measure Meaning

| Symptom | Evidence |
|---------|----------|
| Nonsense passes all rules | `"con con con mẹ sinh cha"` → ✅✅✅✅✅ |
| Gibberish with correct rhyme | `"cùng bao quả bóng người v"` passes structure |
| Word repetition ignored | `"gom gom"`, `"câu câu"` not flagged |

The 5-rule evaluation checks form (rhyme group, tone positions, syllable count,
diacritic pairs, line length). Zero awareness of semantic validity.

### RC2: Hard Rhyme Constraint Forces Nonsensical Words

At inference, when generating the rhyme-position syllable, the beam constraint does:

```python
if matching:  # any rhyming candidate exists
    for tid in non_matching:
        logits[:, tid] = float("-inf")  # DELETE everything else
```

If context is `"con con con mẹ sinh ___"` and rhyme group is `"a"`, candidates include
`cha, nhà, xa, ta, ba, la...`. The model MUST pick from these even if none make sense.
It picks `cha` → gets a passing rhyme score → produces nonsense.

### RC3: 31M Parameters at Capacity Ceiling

| Task | Difficulty | Budget share |
|------|-----------|-------------|
| Vietnamese language modeling | Very high | ~60% |
| Rhyme matching (137 groups) | Medium | ~10% |
| Tone patterns (BTB+BTBB) | Low | ~5% |
| Syllable counting (6/8) | Medium | ~10% |
| Trầm-Bổng (diacritic pairs) | Medium-High | ~10% |
| Nhịp điệu | Low | ~5% |

Each new rule taxes the same 31M params. Structure is low-entropy (learned fast),
content is high-entropy (learned slowly). Gradient goes to what's easiest.

### RC4: Content/Structure Gradient Imbalance

Training treats every token equally in cross-entropy loss. The ~10 control tokens
per example get the SAME gradient weight as the ~28 content tokens. But structural
patterns repeat near-identically across 540K examples (low entropy), so the
optimizer saturates them quickly and leaves content under-optimized.

### RC5: Three Divergent Generation Paths

There are THREE separate generation implementations in the codebase:

| Feature | `sample.py` (CLI) | `server.py` (API/UI) | `eval_rules.py` (benchmark) |
|---------|-------------------|---------------------|---------------------------|
| Repetition penalty (-1.2) | ✅ | ✅ | ❌ **MISSING** |
| Rhyme constraint | Hard mask | Hard mask | Hard mask |
| Top-k | 50 | 50 | 50 |
| Top-p | Optional | 0.92 | ❌ |
| Trầm-Bổng in prompt | ✅ | ❌ **MISSING** | ✅ |
| `<\|start\|>` in prompt | ✅ | ❌ **MISSING** | ✅ (couplet mode) |
| Linebreak parsing | Token-ID based | Token-ID based | **`split('  ')`** (fragile) |
| Syllable enforcement | Optional (off) | **Always ON** (hides errors) | N/A |

**The benchmark evaluates a WEAKER generator than what users see**, and the server
uses a DIFFERENT FORMAT than what the model was trained on.

### RC6: BPE Artifacts Invisible to Detector

Current check:
```python
bpe_artifacts = sum(1 for s in syls if len(s) < 2 or not any(
    c in "aăâeêioôơuưy..." for c in s))
```

Tokens like `"la"`, `"nà"`, `"gi"`, `"v"`, `"tr"` all pass — they contain valid
Vietnamese letters and are ≥2 chars. But they're semantically empty subword
fragments that happen to look like syllables.

### RC7: Scheduled Sampling Exposes But Doesn't Fix Content Weakness

v4.1's fix protects control tokens (`is_control = x < 215`) from replacement during
scheduled sampling. Correct — but when the model's own predictions are used for
content tokens, those predictions are already poor. The model learns: "I can get
away with mediocre content as long as structure is right."

### RC8: Trầm-Bổng Tag + `<|start|>` Missing from Server Inference

`server.py` builds the prompt as:
```python
tags = f"{rhyme_tag} {tone_tag}".strip()          # ← missing [TRAMBONG:NH/HN]
prompt = f"[LUC_BAT] {tags} {input_str} <|reply|>"  # ← missing <|start|>
```

But training format is:
```
<|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH] ... <|reply|>
```

A format mismatch between training and inference — the model was trained to expect
`[TRAMBONG:NH]` and `<|start|>`, but at inference time they're absent. The model
has to infer from corrupted context.

---

## 🎯 Solutions — Sorted by Impact

### TIER 1: Maximum Impact, No Retrain (~75 lines, ~2 hours)

These three directly fix the "ridiculous output" problem the user sees.

---

#### 🥇 P1: Unify All Generation Paths (Impact: FOUNDATIONAL)

**Fixes RC5 + RC8 | Effort: ~60 lines | Retrain: No**

Create ONE canonical generation module. All callers import the same function.

**New file: `src/generation.py`** — single source of truth for:

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, *,
             temperature=0.75, top_k=50, top_p=0.92, max_new=64,
             repetition_penalty=1.2,
             rhyme_constraint=True,
             rhyme_mode="soft",        # "soft" (v4.2) or "hard" (v4.1)
             rhyme_logit_boost=2.0,    # soft mode: +X to matching candidates
             pad_id=0, end_id=3, lb_id=9):
    """Single canonical generator used by CLI, server, and eval."""
    ...

def decode_response(tokenizer, new_token_ids, *,
                    enforce_syllables=False,  # OFF by default — measure raw
                    max_lines=2):
    """Single canonical decoder. No T2a re-split. No P3 by default."""
    ...

def build_luc_bat_prompt(couplet_lines, *, include_trambong=True):
    """Single canonical prompt builder — always includes <|start|> + [TRAMBONG]."""
    ...
```

**Files to refactor:**
- `src/sample.py` → import from `generation.py`, delete local `generate()`/`decode_doi_tho()`
- `client/server.py` → import from `generation.py`, delete local `generate()`/`_decode_doi_tho()`
- `evaluate/eval_rules.py` → import from `generation.py`, delete local `_generate()`

**Why this is #1**: Everything else depends on having a correct, unified generation
pipeline. Currently three divergent paths mean you can't even trust evaluation numbers.
Fixing this is the prerequisite for all other work.

---

#### 🥈 P2: Soft Rhyme Constraint (Impact: DIRECTLY fixes the #1 user complaint)

**Fixes RC2 | Effort: ~15 lines | Retrain: No | Blocks: P1**

Replace hard masking with logit biasing:

```python
# OLD (v4.1) — hard mask forces nonsense:
if matching:
    for tid_i in non_matching:
        logits[:, tid_i] = float("-inf")

# NEW (v4.2) — soft bias, lets model override if needed:
rhyme_mask = torch.full((logits.size(-1),), 0.0, device=logits.device)
for tid_i in matching:
    rhyme_mask[tid_i] = rhyme_logit_boost  # +2.0
logits = logits + rhyme_mask.unsqueeze(0)

# Safety valve: if model is very uncertain about ALL candidates (flat distribution),
# fall back to hard masking to prevent complete randomness
probs = F.softmax(logits, dim=-1)
max_prob = probs.max().item()
if max_prob < 0.05 and matching:  # model has no strong opinion
    for tid_i in non_matching:
        logits[:, tid_i] = float("-inf")
```

**Effect**: A rhyming word gets +2.0 logit (~7.4× probability boost). If a non-rhyming
word is overwhelmingly better semantically (gap > 2.0), the model overrides. This
eliminates "forced nonsense" — the model won't pick `cha` for `"con con con mẹ sinh ___"`
unless it genuinely believes `cha` is the best word.

**Expected**: Rhyme drops 2-5 points (78-82%) but "con con con" nonsense → near zero.

---

#### 🥉 P3: Fix Server Format Mismatch (Impact: Users see correctly-formatted prompts)

**Fixes RC8 | Effort: ~10 lines | Retrain: No | Blocks: P1**

Fix `client/server.py` `_build_doi_tho_prompt()`:

```python
# CURRENT (broken):
tags = f"{rhyme_tag} {tone_tag}".strip()
prompt = f"[LUC_BAT] {tags} {input_str} <|reply|>"

# FIXED:
from tones import get_tram_bong_tag
trambong_tag = get_tram_bong_tag(eight_line) if eight_line else "[TRAMBONG:NH]"
tags = f"{rhyme_tag} {tone_tag} {trambong_tag}".strip()
prompt = f"<|start|> [LUC_BAT] {tags} {input_str} <|reply|>"
```

Also fix `_build_doi_tho_prompt` to include `<|start|>` prefix. The model was trained
with `<|start|>` as the first token; its absence means the positional embeddings are
shifted by 1, corrupting all subsequent position-dependent predictions (including
linebreak timing and rhyme-position alignment).

---

### TIER 2: Quality Safety Net, No Retrain (~120 lines, ~3 hours)

These catch bad outputs that slip through Tier 1 fixes.

---

#### P4: Generate-and-Rerank (Impact: Quality filter)

**Effort: ~80 lines | Retrain: No | Blocks: P1**

Generate N=5 candidates, score each, return best:

```python
def generate_best(model, tokenizer, prompt, n_candidates=5, **kwargs):
    best_score = -float('inf')
    best_tokens, best_text = None, None
    for _ in range(n_candidates):
        tokens, text = generate(model, tokenizer, prompt, **kwargs)
        score = score_candidate(text)
        if score > best_score:
            best_score = score
            best_tokens, best_text = tokens, text
    return best_tokens, best_text

def score_candidate(text):
    """Higher = better. No external models needed."""
    syls = text.split()
    n = len(syls)
    if n == 0: return -100.0
    score = 0.0
    # 1. Lexical diversity (unique/total) → weight 2.0
    score += (len(set(syls)) / n) * 2.0
    # 2. BPE artifact penalty → -1.5 per artifact
    bpe = count_bpe_artifacts(syls)
    score -= bpe * 1.5
    # 3. Adjacent repeat penalty → -2.0 per repeat
    repeats = sum(1 for i in range(n-1) if syls[i] == syls[i+1])
    score -= repeats * 2.0
    # 4. Length bonus (penalize too-short outputs)
    if n < 10: score -= (10 - n) * 0.5
    return score
```

**Trade-off**: 5× slower (~1.6s vs 0.3s per request for 31M model). Acceptable for
current scale. Can reduce to N=3 for server if needed.

---

#### P5: BPE Artifact Detection v2 (Impact: Catches invisible garbage)

**Effort: ~40 lines | Retrain: No | Blocks: P1**

Replace character-set check with syllable validity check:

```python
# Build once from training corpus: set of all syllables ever seen
# in 84K Lục Bát poems (~8,000 unique Vietnamese syllables)
VALID_SYLLABLES: Set[str] = load_from_corpus()  # or static list

def is_bpe_artifact(syllable: str) -> bool:
    """A token is a BPE artifact if it's not a real Vietnamese word/syllable."""
    if len(syllable) < 2:
        return True
    # Strip punctuation
    clean = syllable.strip('.,!?;:-"\'()[]')
    if not clean:
        return False  # pure punctuation is fine
    # Check against known Vietnamese syllables
    return clean.lower() not in VALID_SYLLABLES

def count_bpe_artifacts(syllables: list[str]) -> int:
    return sum(1 for s in syllables if is_bpe_artifact(s))
```

**Building VALID_SYLLABLES**: Extract all unique whitespace-delimited tokens from
`data/doi_tho_corpus.txt`, strip control tokens and punctuation. ~8-10K unique
Vietnamese syllables. Any output syllable not in this set → likely BPE fragment.

---

### TIER 3: Training Improvements, Retrain Required (~75 lines, ~5K steps)

These shift the model's gradient budget from structure toward content.

---

#### P6: Content-Weighted Training Loss (Impact: Teaches model content matters more)

**Effort: ~25 lines | Retrain: Yes | Blocks: None**

**Token ID map** (verified from `poetry_bpe.model`):

Training uses shifted targets: `y = row[1:]`. So `<|start|>` at `row[0]` maps
to `x[0]` only — it is **never in `y`** and its weight never activates.

| ID | Token | In `y`? | Generates at inference? | Weight | Reasoning |
|----|-------|---------|------------------------|--------|-----------|
| 0 | `<\|pad\|>` | Yes (tail) | No | N/A | Already `ignore_index=0` |
| 1 | `<\|start\|>` | **NEVER** (at `row[0]` → `x[0]` only) | No | **Any** | Weight never used |
| 2 | `<\|reply\|>` | Yes | No (given in prompt) | **0.3** | Attention sees it in forward pass; loss weight only affects prediction |
| 3 | `<\|end\|>` | Yes | **YES** — model must stop | **1.0** | Model generates this |
| 4-8 | Genre tags | Yes | No (given in prompt) | **0.3** | Forward pass attends; prediction irrelevant |
| 9 | `<\|linebreak\|>` | Yes | **YES** — model must break | **1.0** | Model generates this |
| 10-146 | `[RHYME:*]` | Yes | No (given in prompt) | **0.3** | Forward pass attends; prediction irrelevant |
| 147-212 | `[TONE:*]` | Yes | No (given in prompt) | **0.3** | Forward pass attends; prediction irrelevant |
| 213-214 | `[TRAMBONG:*]` | Yes | No (given in prompt) | **0.3** | Forward pass attends; prediction irrelevant |
| 215+ | Content | Yes | **YES** — actual poetry | **1.0** | This is what we want to improve |

**Only 2 IDs need full weight**: `<|end|>` (3) and `<|linebreak|>` (9).
These are the only control tokens the model actually generates during
inference. Everything else is either already ignored (`<|pad|>`), never in
the target (`<|start|>`), or given in the prompt at inference (all tags +
`<|reply|>`). The model still attends to these tokens through the forward
pass regardless of their loss weight.

```python
# In train.py, after computing per-token loss (shape: B, T)
# Build weight tensor once
GENERATED_CONTROL_IDS = {3, 9}  # <|end|> and <|linebreak|> — the only
                                 # control tokens the model actually generates

token_weight = torch.ones(model.vocab_size, device=device)
for i in range(215):
    if i not in GENERATED_CONTROL_IDS and i != 0:  # skip 0 (pad, already ignored)
        token_weight[i] = 0.3  # tag tokens: reduce gradient 3.3×

# In loss computation:
loss_per_token = F.cross_entropy(logits, y, ignore_index=0, reduction='none')
weights = token_weight[y]  # (B, T) — lookup weight for each target token
loss = (loss_per_token * weights).mean()
```

**Why this is safe**:
- `<|end|>` (3) keeps full weight → model still learns to stop at the right time
- `<|linebreak|>` (9) keeps full weight → model still learns correct line structure
- `<|start|>` (1) is NEVER in `y` (always at `row[0]` → `x[0]` only) → its weight is irrelevant
- `<|reply|>` (2) and all tags (4-8, 10-214): the model sees them in the forward pass
  (attention uses them as conditioning signals). The loss weight only affects how well
  the model *predicts* them — but it never needs to predict them at inference since
  they're given in the prompt. Reducing their weight tells the optimizer: "don't waste
  gradient on predicting these, use it for content instead."

---

#### P7: N-gram Diversity Loss (Impact: Repetition becomes a training objective)

**Effort: ~25 lines | Retrain: Yes | Blocks: None**

Penalize the model during training for putting probability mass on words that
would create repeated bigrams:

```python
def diversity_loss(logits, input_ids, window=16):
    """
    Penalize high probability on tokens that repeat recent content.
    Only applies to content tokens (IDs ≥ 215), not control tokens.
    """
    B, T, V = logits.shape
    total = 0.0
    count = 0
    for t in range(1, T):
        recent = input_ids[:, max(0, t-window):t]
        for b in range(B):
            for prev_id in recent[b]:
                if prev_id >= 215:  # content tokens only
                    prob = F.softmax(logits[b, t], dim=-1)[prev_id]
                    total += prob
                    count += 1
    return total / max(count, 1)

# Add to main loss:
loss = ce_loss + 0.03 * div_loss  # 3% weight
```

---

#### P8: Linebreak Position Reinforcement (Impact: Better syllable counting natively)

**Effort: ~25 lines | Retrain: Yes | Blocks: None**

Add a small logit bonus for `<|linebreak|>` at correct positions in output lines:

```python
# In training loop, after forward pass:
# For each batch item, find <|reply|> position → then linebreak should appear
# at syllable 6 in output Lục, syllable 14 in output Bát
reply_id = 2
lb_id = 9

for b in range(B):
    # Find <|reply|> in this sequence
    reply_positions = (input_ids[b] == reply_id).nonzero(as_tuple=True)[0]
    if len(reply_positions) == 0:
        continue
    reply_pos = reply_positions[0].item()
    
    # After <|reply|>, linebreak should be at syllable 6 (output Lục)
    # and syllable 14 (output Bát). We estimate: ~6 tokens after reply.
    # A simpler approach: add small logit bonus at the token positions
    # where training data shows <|linebreak|>
    # 
    # Even simpler: for output portion, add 0.3 to lb_id logits at 
    # positions where we expect a linebreak
    for target_offset in [7, 16]:  # rough: after reply + ~7 tokens = 6 syllables
        target_pos = reply_pos + target_offset
        if target_pos < T:
            logits[b, target_pos, lb_id] += 0.3  # soft guidance
```

**Alternative (simpler)**: Just apply the bonus wherever the ground-truth `y` is
`<|linebreak|>` — effectively telling the model "your prediction of `<|linebreak|>`
here is extra important":

```python
# After computing loss_per_token (B, T):
lb_positions = (y == lb_id)  # where ground truth is <|linebreak|>
linebreak_bonus = 0.2  # extra weight on getting linebreaks right
weights = token_weight[y].clone()
weights[lb_positions] += linebreak_bonus
loss = (loss_per_token * weights).mean()
```

---

### TIER 4: Measurement (No Retrain, ~150 lines, ~2 hours)

---

#### P9: Semantic Quality Evaluation Suite

**Effort: ~150 lines | Retrain: No | Blocks: P1**

New file: `evaluate/eval_quality.py` — measures what matters:

| Metric | What It Measures | Good | Bad |
|--------|-----------------|------|-----|
| **Lexical diversity** | unique/total syllables | > 0.75 | < 0.5 |
| **Adjacent repeat rate** | syl[i] == syl[i+1] | < 3% | > 8% |
| **Bigram novelty** | % bigrams NOT in last 8 tokens | > 80% | < 60% |
| **BPE artifact rate** | % outputs with subword fragments | < 5% | > 15% |
| **Syllable validity** | % syllables in known-Vietnamese set | > 95% | < 85% |
| **Output completeness** | % outputs with 14±2 syllables | > 90% | < 70% |
| **Empty/short rate** | % prompts yielding <4 syl | 0% | > 5% |
| **Model confidence** | Mean probability of chosen tokens | N/A (track) | N/A |
| **Human blind score** | 1-5 on 20 random outputs | ≥ 3.0 | < 2.5 |

Combined "Poetic Quality Score" = weighted average.

**v4.1 baseline must be measured FIRST**, before any changes, so we can quantify
improvement. Run `eval_quality.py` against current checkpoint + current `eval_rules.py`
generator to establish the true baseline (not the inflated numbers from the
weaker eval generator).

---

## 📋 Implementation Plan

### Phase 1: Fix Generation Pipeline (No Retrain, ~2 hours)

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| P1 | Create `src/generation.py` + refactor 3 callers | 60 lines | — |
| P2 | Soft rhyme constraint | 15 lines | P1 |
| P3 | Fix server format (Trầm-Bổng + `<\|start\|>`) | 10 lines | P1 |
| P5 | BPE artifact detection v2 | 40 lines | P1 |

**After Phase 1**:
- Single generation function used everywhere
- Server format matches training format
- Rhyme constraint no longer forces nonsense
- BPE artifacts are detectable (for P4 reranking)

**Phase 1 SUCCESS GATE** (MUST PASS before continuing):
```bash
# 1. Generate 20 samples with unified generator
PYTHONPATH=. python3 src/sample.py --num_samples 20 --interactive

# 2. Human blind review: score each 1-5, count:
#    - Nonsense outputs: must be < 3/20
#    - Triple-repeats ("con con con"): must be 0/20
#    - BPE fragments: must be < 2/20
#    - Average score: must be ≥ 3.0

# 3. Run structural eval with unified generator
PYTHONPATH=. python3 evaluate/eval_rules.py
# Expect: all-5-pass drops to 55-65% (honest measurement)
# If it stays at 76%: the structural checks are still being gamed
```

If Phase 1 fails the gate: **the generation pipeline is still broken. Fix P1-P3
before proceeding. Retraining will not fix a broken pipeline.**

### Phase 2: Quality Safety Net (No Retrain, ~3 hours)

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| P4 | Generate-and-rerank (N=5) | 80 lines | P1 |
| P9 | Semantic quality eval suite | 150 lines | P1 |

**After Phase 2**:
- Bad candidates filtered by reranking
- Semantic metrics established for before/after comparison

### Phase 3: Training Improvements (Retrain ~5K steps, ~1 day)

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| P6 | Content-weighted loss | 25 lines | — |
| P7 | N-gram diversity loss | 25 lines | — |
| P8 | Linebreak reinforcement | 25 lines | — |

**Training recipe**:
- Start from v4.1 best checkpoint (step 8800)
- Corpus: 540K Lục Bát pairs, window=1 (unchanged)
- Steps: 5,000 fine-tune
- LR: 1e-4 with warmup=100, cosine to 1e-5
- Batch: 192 (unchanged)
- P6 active from step 0
- P7: diversity_loss weight = 0.03
- P8: linebreak bonus = 0.2

**Phase 3 SUCCESS GATE**:
```bash
# 1. Full eval suite
PYTHONPATH=. python3 evaluate/eval_rules.py
PYTHONPATH=. python3 evaluate/eval_quality.py

# 2. Blind human review of 20 samples → avg score ≥ 3.5

# 3. Structural metrics should be:
#    - R1 Rhyme: ≥ 78% (accepting soft constraint drop)
#    - R2 Tone: ≥ 90%
#    - R4 Trầm-Bổng: ≥ 85%
#    - All-5-pass: ≥ 60%

# 4. Semantic metrics should be:
#    - Lexical diversity: ≥ 0.80
#    - Adjacent repeat: < 3%
#    - BPE artifact rate: < 3%
```

---

## 📊 Target Metrics Summary

| Metric | v4.1 Baseline | Phase 1 Target | Phase 3 Target | Go/No-Go |
|--------|--------------|----------------|----------------|----------|
| **Human quality (1-5)** | ~1.5-2.0 | ≥ 3.0 | ≥ 3.5 | 🚦 Phase 1 gate |
| **Nonsense rate** | ~40% | < 15% | < 10% | 🚦 Phase 1 gate |
| **Triple-repeats** | Present | 0 | 0 | 🚦 Phase 1 gate |
| R1 Rhyme | 84% | 78-82% | 78-85% | Monitor |
| R2 Tone | 92% | 90-94% | 90-95% | Monitor |
| R4 Trầm-Bổng | 90% | 85-90% | 85-92% | Monitor |
| All-5-pass | 76% | 55-65% | 60-70% | Monitor |
| Lexical diversity | ~0.89* | > 0.75 | > 0.80 | Monitor |
| Adjacent repeats | ~8% | < 5% | < 3% | Monitor |
| BPE artifacts | ~15% hidden | < 5% | < 3% | Monitor |
| Syllable validity | ~85% | > 92% | > 95% | Monitor |

*\* v4.1 lexical diversity of 0.89 is deceptive: it counts unique syllables across
very short or truncated outputs. The real content diversity is much lower.*

---

## 🚦 Go / No-Go Decision Framework

### Phase 1 Gate (after ~2 hours of work, no retrain)

**GO to Phase 2 if ALL of:**
- [ ] Human blind review: ≥ 15/20 outputs "read like Vietnamese" (score ≥ 3)
- [ ] Triple-repeats ("con con con"): 0 occurrences in 20 samples
- [ ] BPE fragments visible: ≤ 2/20 samples
- [ ] Server format correct: `<|start|>` and `[TRAMBONG:*]` present in prompts
- [ ] Eval uses unified generator: `eval_rules.py` imports from `generation.py`

**NO-GO (stop, fix pipeline first) if ANY of:**
- [ ] Nonsense rate still > 25% (≥ 6/20 outputs are word salad)
- [ ] Triple-repeats still present
- [ ] BPE fragments still common (> 3/20 samples)

### Phase 3 Gate (after retrain)

**GO to ship if ALL of:**
- [ ] Human blind review: ≥ 17/20 score ≥ 3, average ≥ 3.5
- [ ] Lexical diversity ≥ 0.80
- [ ] Adjacent repeat rate < 3%
- [ ] BPE artifact rate < 3%
- [ ] R1 Rhyme ≥ 78%

**PARTIAL SHIP (with caveats) if:**
- [ ] Human blind review: 14-16/20 score ≥ 3, average 3.0-3.4
- [ ] Lexical diversity 0.70-0.79
- [ ] Structural metrics within range

**NO-GO (revert to v4.1) if:**
- [ ] Human quality worse than v4.1
- [ ] Any metric significantly regressed

---

## 📁 Files Changed / Created

| File | Change |
|------|--------|
| `src/generation.py` | **NEW** — Canonical generate + decode + prompt builder + scoring |
| `src/sample.py` | Remove local `generate()`/`decode_doi_tho()`, import from `generation.py` |
| `client/server.py` | Remove local `generate()`/`_decode_doi_tho()`, import from `generation.py`, fix format |
| `evaluate/eval_rules.py` | Remove local `_generate()`, import from `generation.py` |
| `evaluate/eval_quality.py` | **NEW** — 9 semantic quality metrics |
| `src/train.py` | Add P6 (content weights), P7 (diversity loss), P8 (linebreak bonus) |
| `data/valid_syllables.txt` | **NEW** — Set of all known Vietnamese syllables from corpus |
| `documents/roadmap_v4.2.md` | This document |

---

## 🚫 Anti-Patterns Catalog (Cumulative v1→v4.2)

1. **Don't add new rules until current quality is solid.** (v4 Thất Ngôn lesson)
2. **Post-processing hacks lie.** (v4 P3 truncation — metrics ≠ reality)
3. **Weighted loss is a crutch for data scarcity.** (v4 T2b)
4. **Structural metrics ≠ poetic quality.** (v4.1 — 76% pass ≠ good poetry)
5. **Hard constraints at inference force nonsense.** (v4.2 RC2 — beam rhyme masking)
6. **Multiple generation paths create invisible divergence.** (v4.2 RC5 — three gens)
7. **Train/inference format mismatch is a silent killer.** (v4.2 RC8 — missing tags)
8. **Don't measure what's easy; measure what matters.** (v4.2 RC1 — proxy metrics)
9. **Small models need training-time diversity objectives.** (v4.2 RC7 — inference penalties are band-aids)
10. **Control tokens the model never generates can safely receive reduced training weight.** (v4.2 P6 — 210 of 215 control tokens are prompt-only)

---

## 🎓 Design Principles

1. **One generation function.** `src/generation.py` is the single source of truth.
2. **Train what you test, test what you measure.**
3. **Soft constraints > hard constraints at inference.** Bias, don't force.
4. **Format is everything.** Train format = inference format. Always.
5. **Measure raw + processed.** Report both pre- and post-truncation.
6. **Human review trumps all metrics.** Structure + semantics both proxy for it.
7. **No new features without capacity budget.** 31M params is fixed.
8. **Gate every phase.** Don't proceed if the pipeline is still broken.
