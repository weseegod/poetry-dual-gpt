# 📝 Final Exam: PoetryDuel-GPT — Transformer Mastery

> Covers: Attention mechanics, training dynamics, data pipeline, generation, debugging.
> **This is a code-reading + understanding exam.** Answer by tracing through the actual source.
> Answer key at the bottom — self-grade after attempting.

---

## ═══════════════════════════════════════
## Section A: Attention & Transformer Mechanics (25 pts)
## ═══════════════════════════════════════

**A1. Full attention forward pass — trace the shapes.** (6 pts)

Given: `B=4`, `T=128`, `n_embd=512`, `n_head=8`, `head_dim=64`.

```
x = (4, 128, 512)
x → LayerNorm → QKV projection → reshape → split Q/K/V → attention → concat → output proj
```

Trace the shape at each step:
- After LayerNorm:  `(?, ?, ?)`
- After QKV linear: `(?, ?, ?)`
- After split into Q, K, V: each is `(?, ?, ?, ?)`
- After Q @ K^T: `(?, ?, ?, ?)`
- After softmax: `(?, ?, ?, ?)`
- After softmax @ V: `(?, ?, ?, ?)`
- After concat heads: `(?, ?, ?)`
- After output proj: `(?, ?, ?)`

**A2. Why 8 heads instead of 1 big head?** (4 pts)

If you used 1 big attention head (head_dim=512), what would the model lose? Give a concrete example of what different heads might learn in a poetry model.

**A3. Causal mask — the actual matrix.** (4 pts)

For T=4, write out the 4×4 causal mask (1=allowed, 0=masked). Then explain: at generation time (inference), we process one token at a time. Does the causal mask still matter? Why or why not?

**A4. Residual connections — gradient highway.** (5 pts)

```
out = x + attention(ln1(x))
out = out + ffn(ln2(out))
```

The gradient of loss w.r.t `x` flows through TWO paths. Identify both paths. What happens to the gradient if `attention(ln1(x))` outputs garbage (random values)? What happens if we had NO residual connection?

**A5. FeedForward — why 4x expansion?** (3 pts)

Our FFN: `512 → 2048 → 512` (4× expansion). GPT-3 uses 4×. GPT-4 uses 8×. What does the expansion ratio control? Why not just `512 → 512 → 512`?

**A6. Positional encoding — learned vs sinusoidal.** (3 pts)

Our model uses `nn.Embedding(block_size, n_embd)`. What's the difference between learned and sinusoidal? Which one can extrapolate to sequences longer than training? Which one did we pick?

---

## ═══════════════════════════════════════
## Section B: Training Dynamics (25 pts)
## ═══════════════════════════════════════

**B1. Cross-entropy loss — what is it actually?** (4 pts)

```python
loss = F.cross_entropy(logits, targets, ignore_index=0)
```

If at a certain position, the model predicts token 42 with 98% probability and token 42 is correct, what is the loss contribution? If the correct token is 99 and the model predicts 0.01% probability, what is the loss? What does ignore_index=0 do and why?

**B2. Teacher forcing vs autoregressive — the gap.** (5 pts)

During training, the model sees the GROUND TRUTH prefix:
```
[LUC_BAT] Thân em như chẽn lúa đòng <|reply|> hạt gạo trắng trong tỏ tường <|end|>
```
At position "hạt", the model attends to the real "Thân em như chẽn lúa đòng <|reply|>".

During inference (generation), the model sees ITS OWN previous outputs. If the model's first generated token is wrong, what problem does this cause? Why does teacher forcing make training loss look better than real generation quality?

**B3. Your training log — read the signals.** (4 pts)

```
Step  200: val=3.82  lr=1.20e-04  [0/5] 🏆 Best!
Step 4000: val=2.45  lr=2.13e-04  [0/5] 🏆 Best!
Step 8000: val=2.39  lr=4.06e-05  [0/5] 🏆 Best!
```

At step 4000, LR=2.13e-04. At step 8000, LR=4.06e-05 (5× smaller). Yet val loss keeps improving. What does this tell you about the loss landscape? Why doesn't the small LR cause the model to get stuck?

**B4. Gradient clipping — why 1.0?** (4 pts)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

What happens if you set this to 0.1? What happens if you set it to 100.0? What training problem does gradient clipping solve — specifically in Transformers with LayerNorm?

**B5. bfloat16 — where does it NOT help?** (4 pts)

We use bfloat16 for the forward pass. What specific operation MUST stay in float32? (Hint: look for a comment in the model code about this.) Why?

**B6. Weight initialization — why not zeros?** (4 pts)

If you initialized ALL weights to zero, what would happen during the first forward pass? Specifically in the attention layer and FFN — would the outputs be different across positions? Would gradients be different? Would training ever escape?

---

## ═══════════════════════════════════════
## Section C: Data Pipeline & Tokenization (20 pts)
## ═══════════════════════════════════════

**C1. BPE tokenization — what does the model actually see?** (4 pts)

```
Text:    "lúa đòng"
Tokens:  [1234, 567]          (hypothetical)
```

The model never sees "l", "ú", "a" — it sees token IDs 1234 and 567. What's the embedding lookup doing? If "lúa" and "lụa" have completely different token IDs, can the model learn they're similar? How?

**C2. Control tokens — injection mechanism.** (4 pts)

```
Raw prompt:    "Thân em như chẽn lúa đòng"
After tagging: "[LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng"
Tokenized:     [4, 10919, 10890, 23, 456, 78, 902, 1456, 34, 5678, 234]
                 ↑ genre  ↑ rhyme    ↑ tone         ↑ actual poem tokens
```

The control tokens (indices ~10K) are in the SAME vocabulary as poem tokens (~0-10K). The attention mask treats them identically. At generation time, position 0 is `[LUC_BAT]`. How does the model's attention at position 20 (deep into the generated response) know about the rhyme constraint from position 1?

**C3. Vietnamese tone system — what our tone tag captures.** (4 pts)

```
Vietnamese has 6 tones: ngang (B), huyền (B), sắc (T), hỏi (T), ngã (T), nặng (T)
Lục Bát pattern: 1=B, 2=T, 3=B, 4=B, 5=T, 6=T, 7=B, 8=B  →  "BBBTTB" (first 6 only)
```

"Thân em như chẽn lúa đòng" → tones = B B B T T B → tag = `[TONE:BBBTTB]`

What does the model lose by mapping 6 tones into just 2 categories (B/T)? What does it gain?

**C4. Corpus construction — why couplets?** (4 pts)

```
Poem: A B C D E F G H  (8 lines, 4 couplets)
Pairs: A→B, C→D, E→F, G→H
NOT:   B→C, D→E, F→G  (these are CROSS-couplet pairs, excluded by step=2)
```

Why are cross-couplet pairs semantically wrong for Lục Bát? Give an example of what a cross-couplet pair would look like and why it would confuse the model.

**C5. Dataset ratios — what goes wrong if unbalanced?** (4 pts)

```
Stage 1 corpus: 942K pairs (Lục Bát ~70%, Thất Ngôn ~25%, others ~5%)
Stage 2 corpus: 651K pairs (Lục Bát 100%)
```

If Stage 1 was 98% Lục Bát and 2% Thất Ngôn, what would happen to Thất Ngôn quality? What would happen to the model's ability to respond to `[THAT_NGON]` tags? Why does Stage 2 work despite being single-genre?

---

## ═══════════════════════════════════════
## Section D: Generation & Sampling (15 pts)
## ═══════════════════════════════════════

**D1. Temperature — reshaping the distribution.** (4 pts)

```python
logits = logits / temperature   # T=0.75
probs = softmax(logits)
```

If logits = [2.0, 1.0, 0.5] and temperature = 0.1, what are the approximate probabilities? With T=2.0? What does T<1 do to the distribution? What does T>1 do?

**D2. Top-p (nucleus) sampling — choosing the cutoff.** (4 pts)

```python
sorted_probs, sorted_idx = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum > top_p       # top_p = 0.9
```

With probabilities [0.5, 0.3, 0.1, 0.07, 0.03] and top_p=0.9:
- After cumsum: [0.5, 0.8, 0.9, 0.97, 1.0]
- mask > 0.9: [F, F, T, T, T] — keeps first 3, drops last 2

Why not just take top-k=3? What's the advantage of top-p over top-k?

**D3. The generation loop — what state is carried across steps?** (3 pts)

```python
for _ in range(max_tokens):
    logits = model(idx)         # idx grows by 1 each step
    next_token = sample(logits)
    idx = torch.cat([idx, next_token], dim=1)
```

With KV-caching NOT implemented, we recompute the full sequence every step. For a 100-token generation, how many total attention operations occur? (Write the formula.)

**D4. Stopping condition — when does it actually stop?** (4 pts)

```python
if next_id == end_id:
    # Decode only the new tokens (not prompt)
    generated = ids[:, prompt_len:] + new_tokens
    break
```

What are ALL the conditions that end generation? What happens if the model never generates the end token? What happens if it generates end token at position 3 of a 14-syllable Lục Bát line?

---

## ═══════════════════════════════════════
## Section E: Two-Stage Training & Rhyme Conditioning (15 pts)
## ═══════════════════════════════════════

**E1. Why two stages? Why not one?** (4 pts)

You could train on Lục Bát only for 10K steps. Why train on all genres first (Stage 1) then fine-tune on Lục Bát (Stage 2)? What does Stage 1 provide that pure Lục Bát training cannot?

**E2. Catastrophic forgetting in Stage 2.** (4 pts)

During Stage 2, the model only sees Lục Bát examples. After 5000 steps, you test it with a Thất Ngôn prompt. What would you expect to happen, and why?

**E3. Rhyme conditioning — what the model actually learns.** (4 pts)

```
Training examples:
  [RHYME:ong] ... đòng ... <|reply|> ... chồng ...  (ong - ông match)
  [RHYME:ong] ... lòng ... <|reply|> ... giông ...  (òng - ông match)
  [RHYME:ong] ... đồng ... <|reply|> ... sống ...   (ồng - ông... close)
```

The model sees "ong" in the tag but "ông", "ồng" in example responses. Our rhyme system groups by final consonant (ong/ông/ồng all → "ong" group). What will the model learn — exact rhyme or approximate rhyme group? How accurate do you expect it to be?

**E4. The main training loop — a practical scenario.** (3 pts)

You get impatient and change `patience: 5` to `patience: 2`. Training stops at step 2400 with val=3.1. What did you lose by being impatient? What were you hoping to gain?

---

## ═══════════════════════════════════════
## Scoring
## ═══════════════════════════════════════

| Section | Points |
|---------|--------|
| A: Attention & Transformer Mechanics | /25 |
| B: Training Dynamics | /25 |
| C: Data Pipeline & Tokenization | /20 |
| D: Generation & Sampling | /15 |
| E: Two-Stage Training & Rhyme | /15 |
| **Total** | **/100** |

**≥85:** True transformer mastery. You understand the physics, not just the syntax.
**70-84:** Strong grasp. Go do Stage 2 training — you're ready.
**50-69:** Good foundation. Review the sections you missed before training.
**<50:** Re-study the roadmap and code with a focus on WHY, not WHAT.

---

## ═══════════════════════════════════════
## Answer Key
## ═══════════════════════════════════════

<details>
<summary><b>Section A — Attention & Transformer Mechanics</b></summary>

**A1. Shape trace** (6 pts — 0.75 each):

```
x → LayerNorm            → (4, 128, 512)
QKV projection           → (4, 128, 1536)     # 3 × 512 = 1536
Split → Q, K, V          → (4, 8, 128, 64)    # 8 heads × 64 = 512
Q @ K^T                  → (4, 8, 128, 128)   # (128,64) @ (64,128)
+ causal mask + softmax  → (4, 8, 128, 128)   # same shape, now probabilities
attn @ V                 → (4, 8, 128, 64)    # (128,128) @ (128,64)
Concat heads             → (4, 128, 512)      # stack 8 heads
Output proj              → (4, 128, 512)      # same shape
```

**A2** (4 pts): With 1 big head, the model can only attend with ONE attention pattern — it can focus on one relationship per token. With 8 heads, each head can specialize:
- Head 1: syntax (what's the subject? where's the verb?)
- Head 2: rhyme (what was the 6th syllable?)
- Head 3: tone (what's the current tone pattern?)
- Head 4: semantic meaning (what's the poem about?)
- Head 5: position (how far into the line am I?)
- Head 6-8: other patterns

Without multi-head, the attention weights become an average of all these patterns — noisy and unfocused.

**A3** (4 pts):

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

At inference (one token at a time): if you process just `[tok_0]`, the mask is `[1]` → doesn't matter. But if you process incremental growing sequences `[tok_0]`, `[tok_0, tok_1]`, `[tok_0, tok_1, tok_2]`, each step has a different-length mask pointing to the upper-left submatrix. The mechanism is the same — the mask just shrinks. Without KV-cache, the model recomputes full attention anyway.

**A4** (5 pts):

Path 1: `x` directly (identity) — gradient 1.0, no attenuation.
Path 2: through `attention(ln1(x))` — whatever gradient the attention sublayer produces.

If attention outputs garbage: Path 1 still carries the original gradient cleanly. The model can learn to reduce attention's contribution by making its output small, letting the residual dominate. Without residual: gradient ONLY flows through attention → if attention is broken, entire block is broken → training fails.

This is WHY Transformers can be stacked 12, 24, 96 layers deep — residuals guarantee a gradient highway from top to bottom.

**A5** (3 pts): `512 → 512 → 512` has no non-linearity (linear + linear = just another linear). The 4× expansion creates a bottleneck: compress → expand. The hidden dimension (2048) stores intermediate feature representations. Larger expansion = more capacity per token, at the cost of 4× parameters per FFN block. The expansion ratio trades compute for per-token expressiveness.

**A6** (3 pts):
- Learned: `nn.Embedding(block_size, n_embd)` — fixed max length, can't extrapolate beyond training
- Sinusoidal: deterministic function of position — can extrapolate to ANY length (theoretically)
- We picked: learned (simpler, same length as block_size=256)
</details>

<details>
<summary><b>Section B — Training Dynamics</b></summary>

**B1** (4 pts):

CE = -log(p_correct). With p=0.98: loss = -log(0.98) = 0.020. With p=0.0001: loss = -log(0.0001) = 9.21. The loss EXPLODES for confident wrong predictions — it heavily penalizes being confidently wrong.

ignore_index=0: <|pad|> tokens are ignored in loss computation. Without it, the model would waste capacity learning to predict pad tokens (which are meaningless). The loss only counts real tokens.

**B2** (5 pts):

This is the **exposure bias** problem. During training, every token prediction conditions on perfect history. During inference, if the first token is wrong, all subsequent tokens condition on garbage. The error compounds: wrong token 1 → weird attention at step 2 → worse token 2 → worse attention at step 3 → cascade of errors.

Teacher forcing makes train loss overly optimistic. The model never learns to recover from its own mistakes because it never SEES its own mistakes during training.

Solution (not implemented here): scheduled sampling — occasionally feed model its own predictions during training.

**B3** (4 pts):

The loss landscape near a minimum is bowl-shaped — small gradients lead to the bottom. If it were a flat plateau, small LR would stall. But val is still dropping, which means there's still a clear downward slope.

Small LR = fine-grained steps into the valley. Large LR would OVERSHOOT the minimum. The cosine schedule is doing exactly what it should: big steps to find the neighborhood, tiny steps to settle into it.

**B4** (4 pts):

0.1: Too aggressive. Gradients are clipped so hard that learning stalls — every step becomes tiny regardless of what the gradient says. Model moves at a crawl.

100.0: Effectively no clipping. A single bad batch with a spike in gradient norm can send the model to a completely different part of parameter space — unrecoverable.

Transformers with LayerNorm are susceptible to gradient spikes because LayerNorm's gradient includes division by variance — if variance is tiny, the gradient explodes. Clip=1.0 prevents one bad batch from destroying training.

**B5** (4 pts):

Softmax MUST stay in float32. bfloat16 has ~3 decimal digits of mantissa precision. exp(large_value) in softmax overflows bf16's range, and small probability values become 0.0 in bf16, causing log(0) = -inf in cross-entropy. The model code has: `attn = F.softmax(attn, dim=-1, dtype=torch.float32)`.

**B6** (4 pts):

All-zero weights → all Q, K, V projections produce zero vectors → all attention scores are 0 → softmax gives uniform distribution (each token attends equally to all others) → attention output is same for every position → FFN output (two linear layers with zero weights) is zero → all positions have identical output → gradients are identical for all positions → symmetry is never broken → training can NEVER escape.

Weight initialization MUST break symmetry. Normal distributions (μ=0, σ=0.02) ensure every neuron computes a different function from step 0.
</details>

<details>
<summary><b>Section C — Data Pipeline & Tokenization</b></summary>

**C1** (4 pts):

The embedding lookup is: `tok_emb.weight[token_id]` — a direct index into a matrix. "lúa" (id=1234) and "lụa" (id=9999) look up DIFFERENT rows of the embedding matrix — they start completely unrelated.

But during training, the model sees both in similar contexts (they're both nouns, both appear in rice/poetry metaphors, both have similar tone patterns). The gradients push their embedding vectors in similar directions — they become neighbors in embedding space. This is **distributional semantics**: words that appear in similar contexts get similar embeddings.

**C2** (4 pts):

Bidirectional attention! At generation time, position 20 CAN attend to position 1 (the `[RHYME:ong]` token) because the causal mask only prevents attending to the FUTURE (positions 21+), not the past. The attention at position 20 sees:

```
Positions 0-20:    [LUC_BAT] [RHYME:ong] [TONE:...] Thân em ... <|reply|> ...
                         ↑ position 20's attention can look back here
Positions 21+:     MASKED (future)
```

The rhyme constraint propagates through EVERY attention layer's EVERY head — the model learns to route the rhyme signal from position 1 through its hidden state to influence position 26 (the 6th syllable of the response).

**C3** (4 pts):

Loses: Distinction between tone pairs within B/T. "Sắc" (T) and "nặng" (T) are VERY different sounds — but our system treats them identically. The model can't learn that "sắc" words tend to follow "huyền" words, only that "T" words tend to follow "B" words.

Gains: Drastically simpler conditioning. 2 categories × 6 positions = 2⁶ = 64 possible tone patterns, learnable. 6 categories × 6 positions = 6⁶ = 46,656 — too sparse for the model to learn from 942K pairs.

**C4** (4 pts):

A Lục Bát poem:
```
A: Trăm năm trong cõi người ta         (6-syl)
B: Chữ tài chữ mệnh khéo là ghét nhau  (8-syl)
C: Trải qua một cuộc bể dâu            (6-syl)
D: Những điều trông thấy mà đau đớn lòng (8-syl)
```

Cross-couplet pair (B→C): "Chữ tài chữ mệnh..." → "Trải qua một cuộc..." — these are from DIFFERENT couplets about different things. The rhyme is wrong (nhau doesn't rhyme with dâu). The model would learn that sometimes rhymes matter, sometimes they don't — confusing.

**C5** (4 pts):

98/2 Thất Ngôn → the model barely sees Thất Ngôn examples → `[THAT_NGON]` tag has almost no training signal → the model treats it like noise → Thất Ngôn generation is garbage.

Stage 2 works because: it's FINE-TUNING (not training from scratch). The model already knows Vietnamese grammar, vocabulary, and poetic structure from Stage 1. Stage 2 just narrows focus to Lục Bát rhyme/tone rules. The Lục Bát knowledge from Stage 1 provides the foundation.
</details>

<details>
<summary><b>Section D — Generation & Sampling</b></summary>

**D1** (4 pts):

T=0.1: logits=[20, 10, 5] → after softmax ≈ [0.99995, 0.00004, ~0]. Almost DETERMINISTIC — always picks the highest token. Safe but boring.

T=2.0: logits=[1, 0.5, 0.25] → after softmax = [0.47, 0.28, 0.22]. MUCH flatter — allows creative but risky choices.

T<1 sharpens: amplifies differences, makes model more confident (and more repetitive).
T>1 flattens: reduces differences, makes model more exploratory (and more likely to produce nonsense).

**D2** (4 pts):

Top-k=3 always picks exactly 3 tokens regardless of probability distribution. Problem: [0.9, 0.05, 0.02, 0.01, 0.005, ...] → top-3 keeps the 0.02 and 0.01 tokens even though the 0.9 dominates. [0.33, 0.33, 0.33, 0.01] → top-3 drops the 0.01 even though the distribution is flat.

Top-p adapts: when one token dominates, it keeps fewer candidates. When distribution is flat, it keeps more. It's distribution-aware — the cutoff depends on how confident the model is.

**D3** (3 pts):

Step 1: 1 forward pass on length 1 → 1 attention computation
Step 2: 1 forward pass on length 2 → 2 attention computations (token 0→0, token 1→0,1)
Step N: 1 forward pass on length N → N attention computations

Total attention ops = 1 + 2 + 3 + ... + N = N(N+1)/2. For N=100: 5050 computations. With KV-cache: only N computations (no recomputation). Our model doesn't have KV-cache → O(N²) generation → slow for long sequences.

**D4** (4 pts):

Stop conditions:
1. Model generates `<|end|>` token (token_id=3)
2. Reaches max_tokens limit (prevents infinite loops)
3. (Implicit) The loop doesn't stop from max_new_tokens — it just ends

If model never generates end: runs until max_tokens, then returns whatever it generated (might be truncated mid-line).

If end at position 3: model stops early → response is only 3 characters → broken poem. This is the model COLLAPSING — it learned to end ASAP. Usually caused by: too-low temperature, or the model's training data had many short responses, or the control tokens are confusing it.

The server code ALSO checks: strip `<|reply|>` and `<|end|>` from the decoded text for display.
</details>

<details>
<summary><b>Section E — Two-Stage Training & Rhyme Conditioning</b></summary>

**E1** (4 pts):

Pure Lục Bát training: model only sees Lục Bát → only learns 6→8 pattern → has NO concept of other genres → the `[LUC_BAT]` tag is meaningless (everything is Lục Bát) → `[THAT_NGON]` tag means nothing.

Stage 1 provides: general Vietnamese language knowledge (grammar, vocabulary, word order), multi-genre awareness (learns that `[THAT_NGON]` means 7→7, `[LUC_BAT]` means 6→8), and a strong initialization for Stage 2 to build upon. It's pre-training → fine-tuning, same paradigm as GPT.

**E2** (4 pts):

Catastrophic forgetting: Stage 2's Lục Bát gradient updates override the Thất Ngôn knowledge from Stage 1. After 5000 steps of Lục Bát-only training, Thất Ngôn generation will be WORSE than after Stage 1. The model may still output 7 syllables (the `[THAT_NGON]` tag still triggers some memory), but rhyme quality, fluency, and coherence will degrade.

This is why we keep TWO checkpoints: stage1_best.pt (multi-genre) and stage2_best.pt (Lục Bát specialist). Use different checkpoints for different genres.

**E3** (4 pts):

The model learns APPROXIMATE rhyme groups — not exact rhymes. Since ong/ông/ồng are all in the "ong" group and appear with the SAME `[RHYME:ong]` tag, the model learns that all three endings are acceptable responses. This is actually GOOD — it gives the model creative flexibility while keeping the rhyme constraint satisfied.

Expected accuracy: 60-80% on exact rhyme, 85-95% on rhyme group. The model will sometimes pick a syllable from a completely different group (sampling noise at T=0.75), but most responses should roughly follow the rhyme constraint.

**E4** (3 pts):

You lost 7600 more training steps that could have improved val loss from 3.1 → ~2.39. With patience=2, a single temporary plateau (2 evaluations = 400 steps with no improvement) kills training. The cosine LR had barely begun to decrease — you traded a better model for 30 minutes of wall-clock time, and got a model that's 2× worse (perplexity difference: exp(3.1) / exp(2.39) = ~4× worse token predictions).

What you hoped to gain: faster iteration. The lesson: patience is a hyperparameter tuned to the expected loss curve. 5 at 200-interval = 1000 steps = 10% of training. That's already on the aggressive side for Stage 1.
</details>
