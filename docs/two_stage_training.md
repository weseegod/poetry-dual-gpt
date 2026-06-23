# 🏋️ Two-Stage Training

> Train on all genres first, then fine-tune on Lục Bát — like GPT-3 → ChatGPT.

---

## Why Two-Stage?

```
Stage 1: All genres (136K poems, 942K pairs)
         → Model learns: Vietnamese grammar, poetic vocabulary, general patterns
         → "This is how Vietnamese poetry works"

Stage 2: Lục Bát only (~651K pairs)
         → Model specializes: 6→8 format, tone patterns, rhythm
         → "This is how Lục Bát specifically works"
```

**Single-stage (what you have now):** 30M model, trained only on mixed genres. It splits its limited capacity across Lục Bát AND Thất Ngôn AND patterns. Result: mediocre at both.

**Two-stage:** Stage 1 builds a strong Vietnamese poetry foundation. Stage 2 focuses all capacity on Lục Bát. Result: better Lục Bát while still understanding Thất Ngôn as a "secondary skill."

---

## What You Already Have

```
✅ Clean data:    resources/poems_dataset_clean.csv  (136K poems, no HTML, no dupes)
✅ Multi-genre:   [LUC_BAT] + [THAT_NGON] tokens both working
✅ Corpus:        resources/poetry_corpus.txt  (942K pairs, strict syllables)
✅ Tokenizer:     tokenizer/poetry_bpe.model  (10,785 vocab, indices 0-7 = special)
✅ Training code: src/train.py with cosine LR, mixed precision, best.pt saving
✅ 30M model:     n_embd=512, n_head=8, n_layer=8, block_size=256
```

---

## Implementation

### Step 1: Generate two separate corpora

You need one corpus with ALL genres (Stage 1) and one with only Lục Bát (Stage 2).

**Option A: Filter at preprocessing time (simpler)**

```bash
# Stage 1 corpus — all genres (already have this!)
# resources/poetry_corpus.txt = 942K pairs ([LUC_BAT] + [THAT_NGON])

# Stage 2 corpus — Lục Bát only
python -c "
with open('resources/poetry_corpus.txt') as f:
    lines = f.readlines()
luc_bat = [l for l in lines if '[LUC_BAT]' in l]
with open('resources/corpus_luc_bat.txt', 'w') as f:
    f.writelines(luc_bat)
print(f'Lục Bát pairs: {len(luc_bat):,}')
"
```

**Option B: Modify preprocess.py to accept a genre filter** (cleaner, reusable)

Add `--genre` flag to preprocess.py so you can run:
```bash
python src/preprocess.py --output data/corpus_all.txt               # all genres
python src/preprocess.py --output resources/corpus_luc_bat.txt --genre lục_bát  # Lục Bát only
```

### Step 2: Train Stage 1 (all genres, 15K steps)

```bash
python src/train.py --mode train
```

Config (already in `train.py`):
```python
"train": {
    "max_steps": 15000,
    "batch_size": 192,
    "eval_interval": 500,
}
# Model: n_embd=512, n_head=8, n_layer=8  (~30M params)
# LR: 3e-4 → warmup 500 steps → cosine → 1e-5
# Output: checkpoints/best.pt, checkpoints/step_5000.pt, checkpoints/step_10000.pt, checkpoints/final.pt
```

**Expected on L4:** ~3 hours. Val loss should drop from ~9.3 → ~2.5-2.7.

After completion, **copy final.pt somewhere safe:**
```bash
cp checkpoints/final.pt checkpoints/stage1_base.pt
```

### Step 3: Train Stage 2 (fine-tune on Lục Bát only, 5K steps)

This is the key step. You need to:

1. Start from the Stage 1 model weights
2. Train on Lục Bát pairs only
3. Use a lower learning rate (fine-tuning, not pretraining)

**What needs to change in train.py:**

The current training loop always starts from scratch. You need to add `--resume`:

```python
# In train.py — add to argparse:
p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint for fine-tuning")

# In train(), after model + optimizer creation:
start_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("step", 0)
    print(f"📂  Resumed from step {start_step} ({args.resume})")
    # Reset LR schedule for fine-tuning phase
    cfg["learning_rate"] = 1e-4  # lower LR for fine-tuning
    cfg["warmup_steps"] = 100
```

Then run:
```bash
# Point to Lục Bát-only corpus, resume from Stage 1
python src/train.py \
    --mode train \
    --resume checkpoints/stage1_base.pt \
    --steps 5000
```

**Expected on L4:** ~1 hour. Val loss should go from ~2.5 → ~2.2-2.4 on Lục Bát specifically.

### Step 4: Compare

Generate 5 samples from Stage 1 model vs Stage 2 model with the same prompts:

```bash
# Stage 1 (all-genre)
python src/sample.py --checkpoint checkpoints/stage1_base.pt --num_samples 5

# Stage 2 (fine-tuned Lục Bát)
python src/sample.py --checkpoint checkpoints/final.pt --num_samples 5
```

The Stage 2 model should produce:
- Cleaner 6→8 syllable structure
- More accurate B-T-B-B tones
- More "Lục Bát feeling" vocabulary
- Still capable of Thất Ngôn (from Stage 1 foundation)

---

## Colab Notebook

Updated cell structure for two-stage training:

```python
# Cell 4: Clean + Preprocess
!python src/clean_data.py
!python src/preprocess.py              # → resources/poetry_corpus.txt (all genres)
!python src/train_bpe.py

# Cell 5: Quick Test (verify everything works)
!python src/train.py --mode test

# Cell 6a: Stage 1 — Pretrain on ALL genres (~3 hours)
!python src/train.py --mode train
# After completion:
!cp checkpoints/final.pt checkpoints/stage1_base.pt

# Cell 6b: Stage 2 — Prepare Lục Bát corpus
import subprocess
subprocess.run("""
python -c "
with open('resources/poetry_corpus.txt') as f:
    lines = f.readlines()
luc_bat = [l for l in lines if '[LUC_BAT]' in l]
with open('resources/corpus_luc_bat.txt', 'w') as f:
    f.writelines(luc_bat)
print(f'Lục Bát pairs: {len(luc_bat):,}')
"
""", shell=True)

# Cell 6c: Stage 2 — Fine-tune on Lục Bát only (~1 hour)
!python src/train.py --resume checkpoints/stage1_base.pt --steps 5000

# Cell 7: Generate + Compare
!python src/sample.py --checkpoint checkpoints/stage1_base.pt --num_samples 3
!python src/sample.py --checkpoint checkpoints/final.pt --num_samples 3
```

---

## Files Changed

```
train.py:
  + --resume flag
  + Load checkpoint weights + optimizer state
  + Lower LR for fine-tuning mode

Optional (cleaner):
  preprocess.py:
    + --genre flag to filter output corpus
    + --output flag (already exists)
```

---

## Expected Results

| Metric | Single-stage (current) | Stage 1 (all) | Stage 2 (fine-tuned) |
|--------|----------------------|---------------|---------------------|
| Val loss | ~2.5-2.7 | ~2.5-2.7 | ~2.2-2.4 |
| 6→8 accuracy | ~85% | ~80% | ~95% |
| B-T-B-B tones | ~60% | ~55% | ~80% |
| Vocabulary | Mixed | Rich, varied | Lục Bát-focused |
| Thất Ngôn quality | Mediocre | Good | Still good (from Stage 1) |

---

## Why This Order Before Rhyme Conditioning

```
Two-stage first → establishes quality baseline
Rhyme conditioning later → builds on that baseline

If you do rhyme first, you're adding complexity to a model
that hasn't reached its potential yet.

Two-stage first means you can measure:
  "Here's how good the model is without rhyme help"
vs
  "Here's how much rhyme conditioning adds"
```

---

## Quick Start

```bash
# 1. Train Stage 1
python src/train.py --mode train
cp checkpoints/final.pt checkpoints/stage1_base.pt

# 2. Filter Lục Bát corpus
python -c "
with open('resources/poetry_corpus.txt') as f:
    lines = f.readlines()
lb = [l for l in lines if '[LUC_BAT]' in l]
with open('resources/corpus_luc_bat.txt', 'w') as f:
    f.writelines(lb)
print(f'{len(lb):,} Lục Bát pairs')
"

# 3. Fine-tune Stage 2
python src/train.py --resume checkpoints/stage1_base.pt --steps 5000

# 4. Compare
python src/sample.py --checkpoint checkpoints/stage1_base.pt --num_samples 3
python src/sample.py --checkpoint checkpoints/final.pt --num_samples 3
```
