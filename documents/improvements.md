# 🚀 Improvement Strategies for PoetryDuelGPT

> Your model generates 8 syllables with correct B-T-B-B tones. The *form* is right — now we improve the *content* and fix bugs.

---

## 🔴 Critical Bugs (fix these first)

### 1. Pad token infinite loop in generation (`sample.py`)

**File:** `src/sample.py`, inside `generate()`

Current bug:
```python
if next_id == pad_id: continue   # same context → model picks pad AGAIN → loops forever
```

Fix: suppress pad token BEFORE softmax so it's never sampled:
```python
# In generate(), right after /temperature and before top-k:
logits[:, pad_id] = float("-inf")
```

This bug also exists in `model.py`'s `generate()` method (used only in `__main__` test).

**How to reproduce:** Run `sample.py` 3-4 times. Occasionally the response is empty — that's the model hitting the pad loop until `max_new_tokens` is exhausted.

### 2. No checkpoint resume (Colab-friendly)

**File:** `src/train.py`, `train()` function

Current: always starts from scratch. Colab disconnects = lost progress.

Add:
- `--resume` CLI argument pointing to a `.pt` file
- Load `model_state_dict`, `optimizer_state_dict`, and `step` from checkpoint
- Continue training from that step

Key snippet to add after model creation:
```python
if args.resume:
    ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    print(f"Resumed from step {step}")
```

---

## 🟡 High Impact / Low Effort

### 3. Train longer — 15K-20K steps

**File:** `src/train.py`, `CONFIG["train"]["max_steps"]`

Val loss plateaued at step 3000 but cosine LR hadn't fully decayed. More steps at very low LR (near 1e-5) allow final fine-grained weight adjustments.

```python
"train": {"max_steps": 15000, "batch_size": 192, "eval_interval": 500}
```

Also increase `warmup_steps` proportionally (200→500 for 15K steps, ~3%).

### 4. True semantic couplets, not sliding windows

**File:** `src/preprocess.py`, `make_pairs()`

Current code pairs EVERY adjacent line (`range(len(lines)-1)`). For Lục Bát with 6-8-6-8 pattern, this creates:

```
Pair 0: line(6-syl) → line(8-syl)  ✅ genuine couplet
Pair 1: line(8-syl) → line(6-syl)  ❌ response as prompt, wrong semantics
Pair 2: line(6-syl) → line(8-syl)  ✅ genuine couplet
Pair 3: line(8-syl) → line(6-syl)  ❌ wrong again
```

Fix: change to step-by-2:
```python
for i in range(0, len(lines) - 1, 2):
```

This halves training data from 688K → ~344K pairs, but every pair is a true 6→8 Lục Bát couplet. Quality > quantity.

### 5. Add top-p (nucleus) sampling

**File:** `src/sample.py`, `generate()` function

Current: top-k=50 is a blunt cutoff — always keeps exactly 50 tokens regardless of confidence.

Top-p adapts: keeps enough tokens to reach cumulative probability p.
- Confident model (one token at 0.9): keeps ~1 token → sharp output
- Uncertain model (flat distribution): keeps many tokens → varied output

Implementation sketch:
```python
def top_p_filter(logits, p=0.92):
    """Set logits to -inf for tokens outside the nucleus."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > p
    mask[..., 1:] = mask[..., :-1].clone()  # shift: keep first token past threshold
    mask[..., 0] = False                     # always keep top token
    logits[sorted_idx[mask]] = float("-inf")
    return logits
```

Add `--top_p` CLI arg. When `top_p` is set, skip `top_k`. Allow both to coexist (apply top-p after top-k).

### 6. Tune generation temperature + top-k

**File:** `src/sample.py`, `--temperature` arg

Current T=0.75 is a guess. Sweep and observe:

| T | Behavior |
|---|----------|
| 0.5 | Safe, may repeat |
| 0.6 | Slightly varied, mostly coherent |
| 0.75 | Current — occasionally creative |
| 0.85 | More varied, sometimes wanders |
| 0.95 | Creative, occasionally nonsense |

Also try `--top_k 30` and `--top_k 80` to find the sweet spot.

---

## 🟢 Medium Impact / Medium Effort

### 7. Clean training data — strict syllable filter

**File:** `src/preprocess.py`, `make_pairs()`

Current: ±1 syllable tolerance for noisy data:
```python
p_ok = 5 <= count_syllables(prompt) <= 7
r_ok = 7 <= count_syllables(reply) <= 9
```

Lục Bát is *defined* as exactly 6→8. Tightening removes garbage lines mislabeled as Lục Bát:
```python
p_ok = count_syllables(prompt) == 6
r_ok = count_syllables(reply) == 8
```

Fewer but cleaner pairs. Combine with #4 (true couplets) for maximum data quality.

### 8. Scale the model up

**File:** `src/train.py`, `CONFIG` dict

| Config | Params | Training time (L4) | VRAM |
|--------|--------|--------------------|------|
| `n_embd=384, n_layer=6` | 14.8M | ~43 min (5K steps) | ~4GB |
| `n_embd=512, n_layer=8` | ~30M | ~1.5 hr | ~8GB |
| `n_embd=768, n_layer=8` | ~45M | ~2.5 hr | ~14GB |
| `n_embd=768, n_layer=12` | ~60M | ~3.5 hr | ~16GB |

Start with `n_embd=512, n_layer=8` — it fits on Colab's T4 (16GB) and L4 (24GB).

### 9. Clean training pairs — remove comma from prompt

**File:** `src/preprocess.py`, `make_pairs()`

Current format adds a comma after the prompt:
```python
pairs.append(f"{START} {TAG} {prompt}, {REPLY} {END}")
#                                  ↑ this comma
```

This comma is noise — the model has to learn to ignore it or reproduce it. For cleaner training, remove it:
```python
pairs.append(f"{START} {TAG} {prompt} {REPLY} {END}")
```

Update `sample.py` default prompt to match (remove trailing comma).

---

## 🔵 Higher Effort / Long Term

### 10. Rhyme conditioning

Lục Bát rhyme rule: syllable 6 of the prompt must rhyme with syllable 6 of the reply (vần lưng).

Add rhyme group tags to training data:
```
<|start|> [LUC_BAT] [VAN:ong] Trăm năm trong cõi người ta, <|reply|> Chữ tài chữ mệnh khéo là ghét nhau. <|end|>
```

Steps:
1. Extract last syllable of prompt → get its rhyme group (e.g., "ta" → "a")
2. Inject `[VAN:{rhyme}]` after the genre tag
3. During generation, the model sees `[VAN:ong]` and is conditioned to output rhyming responses
4. Add `[VAN:...]` tokens to `SPECIAL_TOKENS` in `train_bpe.py` (or let BPE learn them)

This is a genuine quality booster — rhyme is the defining feature of Lục Bát.

### 11. Two-stage training

Stage 1: Train on ALL 198K poems (all genres) — model learns Vietnamese grammar, vocabulary, and general poetic structure.

Stage 2: Fine-tune on Lục Bát only — model specializes in the 6→8 format.

Implementation:
1. Modify `preprocess.py` to generate pairs for all genres, not just Lục Bát
2. Train stage 1 model (~15K steps)
3. Save checkpoint
4. Fine-tune on Lục Bát pairs only (~5K steps with lower LR: 1e-4)

### 12. Data cleaning pipeline

The CSV has noise: HTML artifacts, broken lines, wrong genre labels. Build:

```
data/poems_dataset.csv
  → remove HTML/empty <|> tags
  → validate line count vs genre
  → normalize Unicode (NFC vs NFD)
  → remove duplicate poems
  → filter poems < 4 lines
  → save clean version
```

### 13. Multi-genre support

Currently only `[LUC_BAT]` tag is used. The code already reserves `[TU_TUYET]` (id=5) and `[THAT_NGON_BAT_CU]` (id=6).

Add pairs for these genres in `preprocess.py`:
- Thất ngôn tứ tuyệt: 7-syllable × 4 lines
- Thất ngôn bát cú: 7-syllable × 8 lines

Model can then respond in multiple poetic forms when given different genre tags.

---

## 📊 Implementation Order

```
Phase 1 — Stability (do now)
  □ 1. Fix pad token loop in sample.py
  □ 2. Add checkpoint resume to train.py

Phase 2 — Better Poetry (next)
  □ 3. Train longer (15K steps)
  □ 4. True couplets (step=2 in preprocess.py)
  □ 5. Add top-p sampling
  □ 6. Sweep temperature + top-k

Phase 3 — Cleaner Data
  □ 7. Strict syllable filter (6 and 8 exactly)
  □ 8. Scale model up (n_embd=512, n_layer=8)
  □ 9. Remove comma from prompt format

Phase 4 — Advanced
  □ 10. Rhyme conditioning
  □ 11. Two-stage training (all genres → Lục Bát fine-tune)
  □ 12. Data cleaning pipeline
  □ 13. Multi-genre support
```

---

## 🧪 How to Evaluate Each Improvement

After each change, generate 5 samples with `python src/sample.py --num_samples 5` and check:

| Metric | Target |
|--------|--------|
| Syllable count | prompt=6, response=8 (exactly) |
| Response tones | pos2=B, pos4=T, pos6=B, pos8=B |
| Rhyme | prompt's 6th syllable rhymes with response's 6th syllable |
| Semantic coherence | response relates to prompt, not random words |
| Diversity | 5 runs → 5 different responses |
| No empty output | never `<|reply|>` followed by nothing |
