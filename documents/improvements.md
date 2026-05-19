# 🚀 Improvement Strategies for PoetryDuelGPT

> Your model generates 8 syllables with correct B-T-B-B tones. The *form* is right — now we improve the *content*.

---

## 🔴 High Impact / Low Effort (do these first)

### 1. Train longer — 15K–20K steps instead of 5K

Your val loss plateaued at step 3000 but the cosine LR was still at ~1e-4.
Let it decay fully to 1e-5 for further fine-grained learning.

```
Config change:  "max_steps": 15000  (was 5000)
```

### 2. Tune generation temperature

At T=0.75, the model picks "safe" common tokens → generic output.

| Temperature | Behavior |
|-------------|----------|
| 0.5 | Very deterministic, may repeat itself |
| 0.6 | Safe but slightly varied |
| 0.75 | Current setting — balanced |
| 0.9 | Creative, may produce nonsense |
| 1.2 | Very random, rarely coherent |

Sweep `--temperature 0.5 0.6 0.7 0.8 0.9` and read the output to find the best.

### 3. Use top-p (nucleus) sampling instead of top-k

Top-k=50 is a blunt cutoff. Top-p adapts dynamically:

```bash
python src/sample.py --top_k 0 --top_p 0.92
```

When the model is confident (few tokens dominate), top-p keeps only those.
When uncertain (spread out), top-p keeps more options. Better than fixed k=50.

---

## 🟡 Medium Impact / Medium Effort

### 4. Clean the training data

688K pairs include noisy lines. Tighten syllable acceptance in `preprocess.py`:

```python
# Before (loose):
prompt_ok = 5 <= prompt_syl <= 7
reply_ok = 7 <= reply_syl <= 9

# After (strict):
prompt_ok = prompt_syl == 6
reply_ok = reply_syl == 8
```

Fewer but cleaner pairs → model learns the 6→8 pattern sharper.

### 5. Use true semantic couplets, not sliding windows

Current code pairs *every* adjacent line. But Lục Bát couplets are every 6-8 pair:

```python
# In make_luc_bat_pairs(), change:
i += 1    # sliding window: 1-2, 2-3, 3-4...
# to:
i += 2    # true couplets: 1-2, 3-4, 5-6...
```

Halves your data but every pair is a genuine semantic couplet.

### 6. Scale the model up

| Config | Params | Trade-off |
|--------|--------|-----------|
| `n_embd=384, n_layer=6` | 14.8M | Current — fast, small |
| `n_embd=512, n_layer=8` | ~30M | Better capacity, 1.5× training time |
| `n_embd=768, n_layer=8` | ~45M | README target, 2× training time |

---

## 🟢 High Impact / Higher Effort

### 7. Add rhyme conditioning

Lục Bát rhyme rule: the 6th syllable of line 1 must rhyme with the 6th syllable of line 2.

Add a rhyme group tag to training data:
```
<|start|> [LUC_BAT] [VAN:ong] Trăm năm trong cõi người ta, <|reply|> Chữ tài chữ mệnh khéo là ghét nhau. <|end|>
                                                           ^ shell of "ta" rhymes with "nhau"
```

The model learns: `[VAN:ong]` conditions it to generate rhyming responses.

### 8. Two-stage training

Stage 1: Train on all 198K poems → model learns Vietnamese grammar & vocabulary
Stage 2: Fine-tune on Lục Bát only → model specializes in 6→8 poetry

This is how GPT-3 → ChatGPT works (pretrain then fine-tune).

---

## 📊 Priority Order

1. **More steps + tune temperature** — zero code changes, just config
2. **Clean data** — one-line change in `preprocess.py`
3. **True couplets** — one-line change in `preprocess.py`
4. **Top-p sampling** — add to `sample.py`
5. **Bigger model** — config change in `train.py`

Start with #1 and #2 — these alone noticeably improve quality.
