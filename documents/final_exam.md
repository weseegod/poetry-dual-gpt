# 📝 Final Exam: PoetryDuel-GPT

> Covers everything from architecture through improvements.
> Open book. Open code. The goal is understanding, not memorization.

---

## Section A: Architecture (20 pts)

**A1.** The model uses pre-norm (LayerNorm BEFORE attention/FFN). Why? What would break with post-norm? (4 pts)

**A2.** Weight tying: `self.head.weight = self.tok_emb.weight`. Why does this save parameters? How many params does it save in your 30M model (vocab=10,922, n_embd=512)? (4 pts)

**A3.** `attention = Q @ K^T / sqrt(head_dim)`. Why divide by sqrt(head_dim)? What happens if you don't? (4 pts)

**A4.** The causal mask is a `block_size × block_size` lower-triangular matrix. What happens if you feed a sequence longer than block_size without cropping? (4 pts)

**A5.** GELU vs ReLU in the FeedForward. Why GELU? What problem does ReLU have in deep networks? (4 pts)

---

## Section B: Training Loop (20 pts)

**B1.** In `train()`, we do `with torch.autocast(device_type=dev, dtype=torch.bfloat16)`. Why bfloat16 instead of float16? What specific problem does float16 have? (4 pts)

**B2.** `optimizer.zero_grad()` is called BEFORE `loss.backward()`, not after. What happens if you call it after? (4 pts)

**B3.** The cosine LR schedule has warmup (steps 0→500) then decay (500→10000). Why warmup? What happens without it? (4 pts)

**B4.** We separate parameters into `decay` (weights, dim ≥ 2) and `no_decay` (biases, LayerNorm, dim < 2). Why? What happens if you apply weight_decay to LayerNorm? (4 pts)

**B5.** Patience = 5 at eval_interval = 200 means the model stops if val loss doesn't improve for 1000 steps. Is this too aggressive or too lenient for 10K total steps? (4 pts)

---

## Section C: Data Pipeline (20 pts)

**C1.** `clean_data.py` uses `unicodedata.normalize("NFC", text)`. What problem does this solve? What happens if you skip it? (4 pts)

**C2.** The original CSV stores lines with `<\n>` (literal newline) as separator. Our old code used `split(" <\n> ")` with spaces. Why did this silently fail? (4 pts)

**C3.** `make_pairs()` uses `range(0, len(lines)-1, 2)` — step by 2. What problem did this fix compared to the original `range(len(lines)-1)`? (4 pts)

**C4.** Our corpus has `[LUC_BAT] [RHYME:ong] [TONE:BBBTTB] prompt <|reply|> response <|end|>`. The model already learned `[LUC_BAT]` → "generate 8 syllables." How will it learn `[RHYME:ong]` → "6th syllable should rhyme with 'ong'"? Explain the mechanism. (4 pts)

**C5.** The clean CSV is 136K poems, corpus is 942K pairs. Why more pairs than poems? Show the math for a 16-line Lục Bát poem. (4 pts)

---

## Section D: Generation & Evaluation (20 pts)

**D1.** `auto_tag("Thân em như chẽn lúa đòng")` should produce what string? Trace through the logic. (4 pts)

**D2.** During generation, we set `logits[:, pad_id] = -inf`. Why not just `continue` when pad is sampled? What specific bug does `continue` cause? (4 pts)

**D3.** `logits = logits[:, -1, :]` — why `-1`? Why not `logits[:, 0, :]` or use all positions? (4 pts)

**D4.** The `/chat` endpoint decodes only `new_tokens`, not the full sequence. Why? What would happen if we decoded `ids + new_tokens`? (4 pts)

**D5.** After Stage 2 training, you run `sample.py` and get a 7-syllable response to a 6-syllable prompt. Is this a bug? What should you check? (4 pts)

---

## Section E: Debugging (20 pts)

**E1.** Training starts. Loss is 9.3 (random), drops to 5.0 after 100 steps, then gets stuck at 4.8 for the next 500 steps. What's the most likely issue? (4 pts)

**E2.** You run Stage 2 with `--resume checkpoints/stage1_best.pt`. The error says: `RuntimeError: Error(s) in loading state_dict: Missing key(s): tok_emb.weight...` What happened? How do you fix it? (4 pts)

**E3.** Colab disconnects at step 3500 of Stage 1. You re-run Cell 3. What happens? What files does the resume logic look for? (4 pts)

**E4.** After training, `sample.py` produces: `"<|reply|> <|end|>"` (empty response). List 3 possible causes. (4 pts)

**E5.** `checkpoints/` contains: `stage1_best.pt`, `stage1_final.pt`, `stage1_step_5000.pt`, `stage2_best.pt`, `stage2_final.pt`. Which one should you use for the chat UI? Which one has the lowest validation loss? (4 pts)

---

### Answer key (self-grade)

<details>
<summary>Section A</summary>

**A1** (4 pts): Pre-norm keeps the residual path clean — gradients flow directly to early layers through the skip connection. Post-norm normalizes the residual output, which attenuates the gradient signal to early layers. Pre-norm = stable training. Post-norm = vanishing gradients in deep networks.
- Full: gradient flows through residual un-normalized (2 pts) + post-norm kills early-layer signal (2 pts)

**A2** (4 pts): Instead of two separate matrices (tok_emb: 10922×512 and head: 512×10922), they share one. Saves 10922×512 ≈ 5.6M params.
- Correct explanation of weight tying (2 pts) + correct param count ~5.6M (2 pts)

**A3** (4 pts): Without scaling, large head_dim (64) makes dot products large → softmax saturates → gradients vanish. Dividing by √64=8 keeps the variance at 1.0, softmax stays in a useful range.
- Identifies softmax saturation problem (2 pts) + understands sqrt(d) keeps variance stable (2 pts)

**A4** (4 pts): The causal mask is hardcoded to block_size×block_size. Positions beyond block_size have no valid attention entries → all scores become -inf after masking → softmax produces NaN or uniform distribution → garbage output.
- Mask is fixed size (2 pts) + positions beyond block_size attend to nothing (2 pts)

**A5** (4 pts): ReLU(-2.0) = 0.0 with gradient 0 — dead neuron, never recovers. GELU(-2.0) ≈ -0.045 — small negative, gradient flows, neuron can recover when context changes. In 6+ layer networks, dead ReLU neurons accumulate; GELU prevents this.
- ReLU dead neuron problem (2 pts) + GELU's negative tail allows recovery (2 pts)
</details>

<details>
<summary>Section B</summary>

**B1** (4 pts): float16 has only 5 exponent bits → max value ~65,504. Gradients can exceed this → overflow to inf → NaN propagation → training dies. bfloat16 has 8 exponent bits (same as float32) → same range, never overflows. No GradScaler needed.
- float16 overflow problem (2 pts) + bfloat16 same range as float32 (2 pts)

**B2** (4 pts): If you call zero_grad() after backward(), you've already accumulated gradients in .grad. The next backward() ADDS to them instead of replacing → effectively using stale gradients from previous step → corrupted updates.
- Accumulation problem (2 pts) + stale gradients (2 pts)

**B3** (4 pts): Without warmup, the optimizer's momentum buffers (m and v in AdamW) start at 0. Early steps with large LR + zero momentum → unstable updates. Warmup lets momentum build gradually before hitting peak LR. After a few hundred steps, momentum buffers are populated and training is stable.
- Momentum buffers start at zero (2 pts) + warmup lets them stabilize (2 pts)

**B4** (4 pts): Weight decay penalizes large weights (L2 regularization). Applying it to biases forces them toward 0 — forces the model through origin, which is an arbitrary restriction. Applying it to LayerNorm's γ pushes outputs toward 0 — defeats the purpose of LayerNorm (which exists to keep values in a learnable range).
- Biases: forces model through origin (2 pts) + LayerNorm: defeats its purpose (2 pts)

**B5** (4 pts): Slightly aggressive. 1000 steps is 10% of training. If the model plateaus at step 3000, it has 7000 more steps to potentially improve. For Lục Bát fine-tuning where the model starts from a good checkpoint, 5 eval intervals (1000 steps) is reasonable. For Stage 1 from scratch, it might stop too early — the model could be in a temporary plateau before a breakthrough.
- Acknowledges it stops at 10% (2 pts) + reasoning about when it might be too aggressive (2 pts)
</details>

<details>
<summary>Section C</summary>

**C1** (4 pts): Vietnamese can be encoded as NFD (decomposed: tổ = t + o + combining-hook + combining-dot, 5 chars) or NFC (composed: tổ = single char). The tokenizer sees them as different words → vocabulary doubles, training data fragments. NFC normalization ensures consistent representation.
- Explains NFC vs NFD (2 pts) + impact on tokenizer (2 pts)

**C2** (4 pts): The CSV contains `<\n>` with LITERAL newline characters (no spaces). `split(" <\n> ")` looks for space-<-newline->-space which never matches. Result: the entire poem is treated as one line → syllable counts are wildly wrong → most pairs rejected or garbage.
- Identifies wrong separator pattern (2 pts) + consequence (1 big line, wrong syllables) (2 pts)

**C3** (4 pts): `range(len(lines)-1)` pairs EVERY adjacent line: 0→1, 1→2, 2→3. For Lục Bát (6-8-6-8), half the pairs are 8→6 (response as prompt) — semantically wrong. Step-by-2 pairs only 0→1, 2→3 (6→8 couplets only). Halves data but every pair is genuine.
- Identifies 8→6 wrong pairs (2 pts) + step=2 fixes it (2 pts)

**C4** (4 pts): Same mechanism as `[LUC_BAT]` → 8 syllables. During training, the model sees thousands of examples where `[RHYME:ong]` appears before a prompt whose 6th syllable is "ong", and the response's 6th syllable also rhymes with "ong". The attention mechanism learns the correlation: when `[RHYME:ong]` is in the context, bias position 6 of the response toward tokens in the "ong" rhyme group. It's learned from co-occurrence statistics, not explicit rules.
- Same mechanism as genre tags (2 pts) + learned from co-occurrence (2 pts)

**C5** (4 pts): Each poem produces multiple training pairs. A 16-line Lục Bát poem = 8 couplets (16 lines ÷ 2). With strict syllable matching, most couplets become valid pairs. 136K poems × avg ~7 pairs = ~942K pairs. 16-line poem: step=2 gives 8 pairs if all syllable counts are correct.
- Pairs > poems because each poem has multiple couplets (2 pts) + 16/2 = 8 pairs (2 pts)
</details>

<details>
<summary>Section D</summary>

**D1** (4 pts): `auto_tag("Thân em như chẽn lúa đòng")` → 6 syllables → Lục Bát path → extracts rhyme from "đòng" (position 6) → `[RHYME:ong]` → extracts tone sequence "BBBTTB" → `[TONE:BBBTTB]` → result: `"[LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng"`
- Correct tag path (2 pts) + correct final string (2 pts)

**D2** (4 pts): With `continue`, the loop skips appending the pad token AND doesn't break. The context hasn't changed → next forward pass sees the SAME input → model samples pad AGAIN → infinite loop. Eventually max_tokens exhausted → empty output. `-inf` makes pad un-sampleable — the model MUST pick a real token.
- Infinite loop explanation (2 pts) + -inf is the correct fix (2 pts)

**D3** (4 pts): `logits[:, -1, :]` takes the LAST position's predictions. Position T predicts token T+1. We only need the next token after the current sequence, not predictions at every position. `logits[:, 0, :]` would predict token 1 from token 0 — not what comes after the full sequence.
- Last position = next token prediction (2 pts) + why not position 0 (2 pts)

**D4** (4 pts): `ids` (prompt tokens) + `new_tokens` (generated tokens) = full poem. Decoding that would return the user's prompt repeated back to them. The UI should show only the bot's response. We decode only `new_tokens` (the response part).
- Would echo user's prompt (2 pts) + we decode only new_tokens for UI cleanliness (2 pts)

**D5** (4 pts): Not necessarily a bug. Check: (1) Did auto_tag incorrectly detect 7 syllables? Count the actual prompt syllables. (2) Is the model from Stage 1 (all genres) instead of Stage 2? Stage 1 has strong Thất Ngôn capability. (3) Is this consistent across multiple samples, or a one-off sampling variation? Temperature=0.75 allows occasional wrong-syllable outputs even from a good model.
- Three distinct debugging checks (4 pts), partial for 2 checks (2 pts)
</details>

<details>
<summary>Section E</summary>

**E1** (4 pts): Model has reached its capacity limit for the current data. At 30M params with 942K pairs, loss ~4.8 might be the floor. Check: (1) is the LR still high? Should be decaying. (2) are there NaN gradients? Check with torch.isnan. (3) try reducing batch size — sometimes large batches converge to sharp minima and stall.

**E2** (4 pts): The checkpoint was trained with old attribute names (token_embedding, lm_head, qkv_proj). Current model uses new names (tok_emb, head, qkv). Fix: the `--resume` code already has key remapping logic. Check it's being applied — make sure `resume_from` is not None and the remapping loop runs. If the checkpoint is from a COMPLETELY different architecture (e.g., n_embd=384 vs 512), you can't resume — must retrain.

**E3** (4 pts): The resume logic in Cell 3 checks for `checkpoints/stage1_step_*.pt`. If found, it resumes from the latest one. The step counter, optimizer state, and model weights all continue from where they left off. LR schedule restarts from the beginning (a minor inefficiency — but training continues). If no step checkpoints exist (Colab wiped the VM), training starts from scratch.

**E4** (4 pts): Any 3 of: (1) Pad token not suppressed — model samples token 0 repeatedly. (2) Model collapsed during training — all outputs are <|end|>. (3) Wrong tokenizer — special token indices shifted. (4) Temperature too low (0.1) — model always picks <|end|>. (5) Checkpoint corrupted — weights are garbage.

**E5** (4 pts): Chat UI should use `stage2_best.pt` — it's the fine-tuned Lục Bát model with the lowest validation loss. `stage2_final.pt` is the last checkpoint (might not be the best). `stage1_best.pt` is the all-genre model (good Thất Ngôn but not specialized). Lowest val loss: compare `stage2_best.pt` and `stage2_final.pt` — `stage2_best.pt` has the lower one by definition (it's saved when val_loss improves).
</details>

---

### Scoring

| Section | Points | Your score |
|---------|--------|------------|
| A: Architecture | /20 | |
| B: Training | /20 | |
| C: Data Pipeline | /20 | |
| D: Generation | /20 | |
| E: Debugging | /20 | |
| **Total** | **/100** | |

**≥80:** You understand the full pipeline. Go train.
**60-79:** Solid grasp, review the missed sections before training.
**<60:** Re-study the documents for the sections you missed.
