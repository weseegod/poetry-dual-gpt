# 🗺️ PoetryDuel-GPT Learning Roadmap

> Build a ~45M-parameter Vietnamese poetry generator from scratch with raw PyTorch.
> Zero HuggingFace model wrappers. Every matrix multiplication is yours.

---

## 📋 Project Overview

| Aspect | Detail |
|--------|--------|
| **Goal** | Build a model that accepts a Vietnamese poetic line and generates a rule-compliant response |
| **Model** | Decoder-only Transformer (GPT-style), autoregressive |
| **Params** | ~45M (embed 384, 6 layers, 6 heads, vocab 12K) |
| **Framework** | Raw PyTorch (`torch.nn`) — no `transformers`, no `trl`, no Keras |
| **Data** | Vietnamese poetry corpus → turn-based dialogue pairs |
| **Hardware** | Single GPU (NVIDIA L4 / T4 / any ~16GB VRAM) |

---

## 🎯 Phased Training Strategy

**Don't train on all 198K poems at once.** Start small, validate, then scale.

```
Phase 1: Lục Bát only (89,943 poems, 45% of data)
  → 1 genre, 1 rule (6→8 syllables + B-T-B tone)
  → Fastest iteration loop, easiest to debug
  → Goal: model reliably generates 8-syllable responses to 6-syllable prompts

Phase 2: Add bảy chữ / thất ngôn bát cú (46,586 poems, 23%)
  → 2 genres now, model learns [GENRE] tag means different rules
  → Goal: model switches between 8-syllable and 7-syllable output

Phase 3: Add remaining genres (tám chữ, năm chữ, thơ tự do, etc.)
  → Full 198K dataset
  → Goal: model handles all Vietnamese poetic forms
```

**Why this order matters:**
- Lục Bát has the most data AND the simplest rules — perfect for proving the pipeline works
- If the model can't do Lục Bát, it definitely can't do more complex forms
- Each phase adds exactly one new capability, so failures are easy to isolate
- Training time scales linearly: Phase 1 is ~2hr on L4, Phase 3 is ~6hr

---

## 📁 File Map

```
poetry-dual-gpt/
├── README.md                       ← Project pitch (already exists)
├── requirements.txt                ← Dependencies (add as you go)
├── .gitignore                      ← Ignore venv/, checkpoints/, generated files
│
├── data/                           ← 📦 Data files (not code)
│   └── poems_dataset.csv           ← Raw Vietnamese poetry corpus
│
├── src/                            ← 🧠 All source code
│   ├── preprocess.py               ← Phase 1B: raw poetry → training pairs
│   ├── train_bpe.py                ← Phase 1C: train custom BPE tokenizer
│   ├── dataset.py                  ← Phase 3B: PyTorch Dataset class
│   ├── model.py                    ← Phase 2: the Transformer (5 classes)
│   ├── train.py                    ← Phase 3: training loop + mixed precision
│   └── sample.py                   ← Phase 4: generation + Vietnamese rule checks
│
├── documents/                      ← 📖 Documentation
│   └── roadmap.md                  ← YOU ARE HERE — learning guide
│
├── checkpoints/                    ← 💾 Saved model weights (gitignored)
└── tokenizer/                      ← 🔤 Generated tokenizer files (gitignored)
    └── poetry_bpe.model            ← Phase 1C output: saved vocabulary
```

---

## 🔤 Phase 1: Data & Tokenization

**Goal:** Convert raw poetry into tokenized sequences the model can consume.

### 1A — Explore the dataset (10 min)

**First, see what you have:** Run `python src/dataset.py` to print genre distribution, author stats, sample poems, etc. This tells you the Lục Bát dominates (89K poems) and that 161K poems are missing author/period — still valid for training.

### 1B — Understand the data format (10 min)

Read the existing `README.md`. Pay attention to:
- What the input/output looks like (the control token format)
- The poetic genres: Lục Bát, Tứ Tuyệt, Thất Ngôn Bát Cú
- Expected training format: `<|start|> [GENRE] line1, <|reply|> line2 <|end|>`

### 1C — `src/preprocess.py` (30-60 min) — *Lục Bát first!*

**File:** `src/preprocess.py` (open it — comments are your guide)

**Phase 1 filter:** Only process `genre == 'lục bát'` rows. You can filter in pandas:
```python
df = pd.read_csv('data/poems_dataset.csv')
df_luc_bat = df[df['genre'] == 'lục bát']  # 89,943 poems
```

**What to implement:**
1. Read `poems_dataset.csv`, filter for `genre == 'lục bát'`
2. Parse poem `content` column (lines separated by ` <\n> ` marker)
3. Create (prompt, reply) pairs: each 6-syllable line → next 8-syllable line
4. Wrap with control tokens: `<|start|> [LUC_BAT] prompt, <|reply|> reply <|end|>`
5. Write one pair per line to `data/poetry_corpus.txt` (gitignored, generated)

**Concepts learned:**
- Data structuring for causal language modeling
- Control tokens as "instructions" to the model
- Vietnamese syllable counting

**To test:** Create a small sample file with 2-3 poems manually and run the script.

### 1D — `src/train_bpe.py` (45-90 min)

**File:** `src/train_bpe.py` (open it — full BPE walkthrough in comments)

**What to implement:**
1. Define 7 special tokens: `<|pad|>`, `<|start|>`, `<|reply|>`, `<|end|>`, `[LUC_BAT]`, `[TU_TUYET]`, `[THAT_NGON_BAT_CU]`
2. Initialize a BPE tokenizer from HuggingFace `tokenizers`
3. Train it on `data/poetry_corpus.txt` with vocab_size=12000
4. Save to `tokenizer/poetry_bpe.model` (gitignored, generated)

**Concepts learned:**
- Byte-Pair Encoding (how subword tokenization works)
- Why custom tokenizers matter for non-English languages
- Special tokens and their roles (pad, start, end, control)

**Dependencies to install:** `pip install tokenizers`

### 1E — Where does the raw data come from? (already done — you have poems_dataset.csv)

**Option A: Download from HuggingFace (recommended)**

The README mentions two real datasets:
- `roots_vi_vietnamese_poetry`
- `vietnamese-poetry-corpus`

Use the `datasets` library to fetch them:
```python
# pip install datasets
from datasets import load_dataset

# This dataset has ~370K lines of Vietnamese poetry
ds = load_dataset("onghocit/roots_vi_vietnamese_poetry", split="train")

# Write poems to a raw text file for preprocess.py to consume
with open("data/raw_poetry.txt", "w", encoding="utf-8") as f:
    for row in ds:
        # Each row typically has a 'text' or 'poem' field
        poem = row.get("text") or row.get("poem") or ""
        f.write(poem.strip() + "\n\n")  # blank line separates poems
```

**Option B: Build a sample dataset manually**

Create `data/sample_raw.txt` with 20-30 Vietnamese poems (Lục Bát, Tứ Tuyệt) so you can test the pipeline end-to-end before scaling to the full dataset. See `data/preprocess.py` comments for the expected format.

**Option C: Use the sample generator built into preprocess.py**

`preprocess.py` already includes a built-in sample generator function — add a CLI flag `--sample` to trigger it. This gives you ~30 Lục Bát and Tứ Tuyệt examples for testing.

### 1F — Understand Dataset vs DataLoader (15 min) — *read now, implement in Phase 3*

In PyTorch, data feeding follows a two-layer design:

```
  Raw files  ──→  Dataset  ──→  DataLoader  ──→  Model
                 (what data)     (how to serve)
```

**`torch.utils.data.Dataset`** — the "what"
- A class you subclass. Must implement:
  - `__len__()` → total number of samples
  - `__getitem__(idx)` → return the idx-th (input, target) pair as tensors
- Owns the data. Knows how to access it. Does NOT know about batching.

**`torch.utils.data.DataLoader`** — the "how"
- Wraps a Dataset. Handles:
  - Batching (stack multiple samples into one tensor)
  - Shuffling (randomize order each epoch)
  - Parallel loading (`num_workers`) to keep GPU busy
  - Pin memory (`pin_memory=True`) for faster CPU→GPU transfer

**For causal language modeling**, there are two common Dataset patterns:

*Pattern 1: Sequential chunks* (simpler — what Karpathy's nanoGPT uses)
```
Giant flat tensor: [tok1, tok2, tok3, ..., tokN]
Each sample: a random contiguous window of length block_size+1
  x = data[start : start+block_size]
  y = data[start+1 : start+block_size+1]    # shifted by 1
```

*Pattern 2: Per-example Dataset* (more formal — what HuggingFace uses)
```
Each example is a separate line/sequence, tokenized individually.
DataLoader pads them to equal length within each batch.
```

**We'll use Pattern 1** — it's simpler, more memory-efficient for our scale, and avoids wasting compute on padding tokens.

### ✅ Phase 1 Checkpoint

Run this and get reasonable output:
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")
print(tok.get_vocab_size())  # → 12000
encoded = tok.encode("<|start|> [LUC_BAT] Trăm năm")
print(encoded.ids)  # → list of token IDs
print(tok.decode(encoded.ids))  # → back to text
```

And verify your preprocessed data:
```bash
wc -l data/poetry_corpus.txt   # should show many lines (one per pair)
head -3 data/poetry_corpus.txt # inspect the format
```

---

## 🧠 Phase 2: The Transformer Model

**Goal:** Build the entire model architecture from scratch. This is the core.

**File:** `src/model.py` (open it — full concept explanations in comments)

### What you're building (5 classes):

```
Class 1: MultiHeadAttention      ← The "magic" of Transformers
Class 2: FeedForward             ← Per-position processing (MLP)
Class 3: TransformerBlock         ← Attention + FFN + norms + residuals
Class 4: PoetryDuelGPT            ← The complete model
Utility: count_parameters         ← Verify ~45M params
```

### 2A — Study the concepts first (30-60 min)

Read the comment block at the top of `model.py` **thoroughly**. It explains:
- What a language model is
- The Transformer architecture diagram
- How self-attention works (Q, K, V)
- What a causal mask is
- Why residual connections and LayerNorm matter
- Weight tying

Then watch/read these (optional but helpful):
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (video)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original paper (just the decoder part)

### 2B — Implement `MultiHeadAttention` (60-90 min)

The hardest single component. Take your time.

**Steps:**
1. `__init__`: Create Q, K, V linear projections + output projection
2. Register a causal mask as a buffer (lower triangular matrix)
3. `forward(x)`: 
   - Project x → Q, K, V
   - Reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
   - Compute attention: `softmax(QK^T / sqrt(d_k) + mask) × V`
   - Combine heads back: (B, n_head, T, head_dim) → (B, T, C)
   - Output projection

**Debug tip:** Add `assert` statements to check shapes at each step.

### 2C — Implement `FeedForward` (10 min)

Simple: two linear layers with GELU activation. Expand 4×, contract back.

### 2D — Implement `TransformerBlock` (15 min)

Assemble: `x = x + attn(ln1(x))`, then `x = x + ffn(ln2(x))`.

### 2E — Implement `PoetryDuelGPT` (30-45 min)

The full model:
1. Token embedding + Position embedding (add them together)
2. Stack of N TransformerBlocks
3. Final LayerNorm + LM head (linear → vocab_size)
4. Weight tying: `lm_head.weight = token_embedding.weight`
5. Weight initialization: normal(0, 0.02) for linears, zeros for biases
6. `forward(idx, targets)`: returns `(logits, loss)`

### ✅ Phase 2 Checkpoint

```python
from src.model import PoetryDuelGPT, count_parameters

model = PoetryDuelGPT(vocab_size=12000, n_embd=384, n_head=6, n_layer=6, block_size=256)
total, _ = count_parameters(model)
print(f"Params: {total:.1f}M")  # Should be ~45M

# Test forward pass
import torch
x = torch.randint(0, 12000, (2, 64))  # batch=2, seq=64
logits, loss = model(x, targets=x)
print(logits.shape)  # → (2, 64, 12000)
print(f"Loss: {loss.item():.4f}")  # → ~9.4 (ln(12000) ≈ random)
```

---

## 🏋️ Phase 3: Training

**Goal:** Make the model learn poetry through gradient descent.

**File:** `src/train.py` (open it — all training concepts explained in comments)

### 3A — Understand the concepts (20-30 min)

Read the comment block at the top of `train.py`. Key topics:
- The training objective: predict next token
- Cross-entropy loss (why initial loss ≈ ln(vocab_size) ≈ 9.4)
- Mixed precision (bfloat16 vs float16)
- AdamW optimizer and parameter grouping
- Cosine LR schedule with warmup
- Gradient clipping

### 3B — Implement the Dataset & DataLoader (45-60 min)

This is where you bridge the gap between tokenized text and model-ready tensors.

**Step 1: `load_and_tokenize()`** — convert corpus to one giant tensor

```
Input:  data/poetry_corpus.txt (one preprocessed pair per line)
Output: torch.LongTensor of shape (total_tokens,)

Process:
  1. Load tokenizer from file
  2. For each line in corpus:
     - tokenizer.encode(line).ids → list of ints
     - Extend accumulator list
  3. Convert to torch.tensor
```

**Step 2: Split into train/val**

Hold out ~10% of tokens for validation:
```python
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]
```

**Step 3: `get_batch(data, batch_size, block_size)`** — sample random windows

```
Input:  data tensor of shape (N,), batch_size, block_size
Output: x (B, T), y (B, T) where y[:, i] = x[:, i+1]

Algorithm:
  1. Pick batch_size random start indices in range [0, N - block_size - 1]
  2. For each start index i:
       chunk = data[i : i+block_size+1]    # one extra token for target
       x_row = chunk[:block_size]           # tokens 0..T-1
       y_row = chunk[1:block_size+1]        # tokens 1..T  (shifted by 1)
  3. Stack all rows into (B, T) tensors
  4. Move to GPU
```

**Step 4 (optional but educational): Wrap in a proper PyTorch Dataset class**

```python
from torch.utils.data import Dataset, DataLoader

class PoetryDataset(Dataset):
    """
    Wraps a flat token tensor into a Dataset.
    Each __getitem__ returns one (x, y) window.
    """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # number of possible windows
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:self.block_size]
        y = chunk[1:]
        return x, y

# Usage:
train_ds = PoetryDataset(train_data, block_size)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)

# Then in training loop:
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    ...
```

**Key data loading concepts you're learning here:**
- **Batching:** stacking multiple sequences together. The GPU processes them in parallel — that's why batch_size=64 is much faster than batch_size=1.
- **Shuffling:** randomizing order each epoch. Prevents the model from memorizing sequence order instead of learning patterns.
- **Pin memory:** allocates CPU memory in a way that speeds up the CPU→GPU copy (`pin_memory=True` + `non_blocking=True`).
- **num_workers:** spawns subprocesses to load data in parallel while GPU is computing. Rule of thumb: `num_workers = 4` for most setups.
- **Why shift by 1 (next-token prediction):** the model sees tokens [t0, t1, t2, t3] and must predict [t1, t2, t3, t4]. This is called "teacher forcing" — at each position, the model predicts the next token. Cross-entropy loss is computed at EVERY position simultaneously.

### 3C — Implement LR schedule (10 min)

`get_lr(step, warmup, max_steps, max_lr, min_lr)` — the cosine-with-warmup formula.

### 3D — Implement the training loop (60-90 min)

`train(config)`:
1. Load tokenized data
2. Initialize model, move to GPU
3. Set up optimizer with parameter grouping (weight decay on weights, not biases/norms)
4. Set up mixed precision context
5. For each step:
   - Get batch
   - Forward pass under `autocast`
   - Backward pass
   - Clip gradients
   - Optimizer step + scheduler step
   - Log loss every N steps
6. Periodically evaluate on validation set
7. Save checkpoints

### 3E — Implement checkpointing (15 min)

`save_checkpoint()` / `load_checkpoint()`: Save model + optimizer + step so you can resume.

### ✅ Phase 3 Checkpoint

```bash
python -m src.train --epochs 3 --batch_size 64 --device cuda
```

Expected:
- Initial loss: ~9.4
- Loss decreases steadily
- After 3 epochs: validation loss should be ~1.4-2.0
- Training time: ~2 hours on L4, ~4-6 hours on T4

---

## 🎨 Phase 4: Generation & Evaluation

**Goal:** Generate poetry and verify it follows Vietnamese rules.

**File:** `src/sample.py` (open it — full generation and tone-check explanations)

### 4A — Implement sampling strategies (30 min)

1. `sample_with_temperature()`: Scale logits by 1/temperature
2. `sample_top_k()`: Keep only the k highest logits
3. `sample_top_p()`: Nucleus sampling (keep tokens until cumulative prob ≥ p)

### 4B — Implement the generation loop (45-60 min)

`generate(model, tokenizer, prompt, ...)`:
1. Tokenize prompt
2. Loop max_new_tokens times:
   - Crop context to last `block_size` tokens
   - Forward pass → get logits at last position
   - Temperature → top-k → top-p → softmax → multinomial sample
   - Append new token
   - Break if `<|end|>` token generated
3. Decode the full sequence

### 4C — Implement Vietnamese rule checking (30-45 min)

1. `count_syllables()`: Split by spaces (each word = one Vietnamese syllable)
2. `get_tone_type()`: Classify as Bằng or Trắc based on diacritics
3. `check_syllable_count()`: Verify 6→8 for Lục Bát, 7→7 for Tứ Tuyệt
4. `check_tone_alignment()`: Verify B-T-B pattern at positions 2,4,6 for Lục Bát
5. `evaluate_generation()`: Parse output, run all checks, print results

### 4D — Interactive mode (15 min)

CLI loop: user types a line → model responds → evaluate.

### ✅ Phase 4 Checkpoint

```bash
python -m src.sample --prompt "[LUC_BAT] Thân em như chẽn lúa đòng đòng," --temperature 0.75
```

Expected output:
```
[Input Prompt]: [LUC_BAT] Thân em như chẽn lúa đòng đòng,
[Model Rebuttal]: Phất phơ dưới ngọn nắng hồng ban mai.
==================================================
* Metric Evaluation *
Syllable Verification: PASS (6-word prompt -> 8-word response)
Tone Map Alignment: Bằng - Trắc Match Confirmed.
```

---

## 🔬 Phase 5 (Optional Advanced): Improvements

Once the basic pipeline works:

| Challenge | What to do | Learning value |
|-----------|------------|----------------|
| **Better tone checking** | Implement full B-T tables for all genres + rhyme detection | Vietnamese linguistics, regex |
| **Flash Attention** | Replace naive attention with `F.scaled_dot_product_attention` | 2-3× faster training |
| **Rotary Position Embeddings (RoPE)** | Replace learned position embeddings with rotary | Modern LLMs (Llama, Mistral) use this |
| **KV Cache** | Cache K,V from previous steps during generation | Makes generation 10× faster |
| **Dataset augmentation** | Download actual `roots_vi_vietnamese_poetry` from HuggingFace | Working with real datasets |
| **WandB logging** | Add Weights & Biases for loss curves and sampling | ML experiment tracking |
| **Gradient accumulation** | Simulate larger batches on small GPU memory | Common technique in LLM training |

---

## 📚 Key Learning Resources

| Resource | Type | Topic |
|----------|------|-------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Blog | Attention mechanism visuals |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Video (2h) | Karpathy builds nanoGPT from scratch |
| [nanoGPT repo](https://github.com/karpathy/nanoGPT) | Code | Reference implementation (~300 lines) |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Paper | Original Transformer (focus on decoder) |
| [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) | Blog+Code | Line-by-line explanation |
| [PyTorch docs: nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | Docs | See how PyTorch implements it (but build yours) |
| [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/quicktour) | Docs | BPE tokenizer training |
| [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html) | Docs | torch.cuda.amp guide |

---

## 🐛 Debugging Tips

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss = NaN | Exploding gradients, LR too high | Lower LR, check gradient clipping, verify mask |
| Loss doesn't decrease | Wrong masking, wrong data shapes | Verify `targets = input[:, 1:]`, check mask is causal |
| All generated tokens are the same | Temperature too low, or model didn't train | T > 0.7, check loss actually decreased |
| CUDA out of memory | Batch too big, block_size too large | Reduce batch_size to 32 or 16, use grad accumulation |
| Shape mismatch errors | Transpose/view wrong | Print shapes at every step, add assert statements |

---

## 📊 Success Criteria

### Phase 1 (Lục Bát only)
- [ ] `model.py` compiles and produces correct shape outputs
- [ ] Parameter count is ~45M
- [ ] Initial loss ≈ 9.4 (close to random guessing)
- [ ] Training loss decreases steadily on 89K Lục Bát poems
- [ ] Final validation loss < 2.0
- [ ] Generation produces 8-syllable responses to 6-syllable prompts (>50% of the time)
- [ ] Some outputs show proper B-T-B tone alignment (positions 2,4,6)

### Phase 2 (+ bảy chữ)
- [ ] Model switches output length based on `[LUC_BAT]` vs `[BẢY_CHỮ]` tag
- [ ] 7-syllable output for bảy chữ prompts (>50% of the time)
- [ ] Lục Bát quality does NOT degrade (no catastrophic forgetting)

### Phase 3 (full dataset)
- [ ] Generation works across all poetic forms
- [ ] Vietnamese syllables are grammatically valid (>90% of the time)

---

**Start at Phase 1 and work through sequentially. Each file contains all the conceptual knowledge you need in its comments. Good luck!**
