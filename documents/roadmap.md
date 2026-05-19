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

## 🏗️ Architecture Deep Dive

### Layer-by-layer diagram (14.8M params)

```
INPUT: token IDs (B, T)  e.g. (192, 256)
┌──────────────────────────────────────────────┐
│ EMBEDDINGS                                   │
│   Token Embedding   10581×384 = 4,063,104    │
│   Position Embedding  256×384 =    98,304    │
│   → tok + pos = (B, T, 384)                  │
└──────────────────────────────────────────────┘
    │
    ├── BLOCK 0 ─────────────────── 1,773,600
    │   ├── LayerNorm           (768)
    │   ├── MultiHeadAttention
    │   │   QKV proj  384×1152 = 442,368
    │   │   Out proj   384×384 = 147,456
    │   │   Total attn         = 589,824
    │   │   + residual (0 params)
    │   ├── LayerNorm           (768)
    │   ├── FeedForward
    │   │   fc1       384×1536 = 589,824 + 1,536 bias
    │   │   fc2      1536×384  = 589,824 + 384 bias
    │   │   Total FFN          = 1,181,568
    │   │   + residual (0 params)
    │   └── Total block        = 1,773,600
    │
    ├── BLOCK 1 ─────────────────── 1,773,600
    ├── BLOCK 2 ─────────────────── 1,773,600
    ├── BLOCK 3 ─────────────────── 1,773,600
    ├── BLOCK 4 ─────────────────── 1,773,600
    ├── BLOCK 5 ─────────────────── 1,773,600
    │
    ▼  (B, T, 384) — same shape throughout!
┌──────────────────────────────────────────────┐
│ OUTPUT                                        │
│   Final LayerNorm                     (768)  │
│   LM Head (tied to token embedding)      (0) │
│   → logits: (B, T, 10581)                    │
└──────────────────────────────────────────────┘
```

### Parameter breakdown

```
FeedForward     ████████████████████████████  7.09M  (47.9%)
Token Embedding ██████████████████████        4.06M  (27.5%)
Attention       ████████████████              3.54M  (23.9%)
Position Embed  █                              0.10M  ( 0.7%)
LayerNorms      █                              0.01M  ( 0.1%)
                ─────────────────────────────
Total                                        14.80M
```

**Key insights:**
- FeedForward is the heaviest (48%) — expanding 384→1536→384 costs `384×1536×2` per block
- Attention is surprisingly light (24%) — the combined QKV projection saves params vs 3 separate matrices
- Embeddings cost 27% because vocab_size directly controls size
- Residuals, GELU, and Dropout cost ZERO params — they're pure operations
- Weight tying makes LM Head free — saves ~4M params
- Same tensor shape (B,T,384) flows through every block — this is why stacking works

---

## 🧪 Why LayerNorm?

### The problem: values drift out of control

Without LayerNorm, each layer multiplies its input by weight matrices:

```
Input:   [0.5, -0.3, 1.2, ...]     (reasonable range)
  ↓ × W (random normal, std=0.02)
Block 1: [2.1, -4.7, 8.3, ...]     (drifting wider)
  ↓ × W
Block 2: [15.2, -32.1, 47.8, ...]   (exploding!)
  ↓ × W  
Block 6: [1421.3, -893.2, ...]      (Gradient = NaN → training dies)
```

Each matrix multiply amplifies the values. After 6 layers, the numbers are so large that softmax saturates (outputs 0 or 1 only) and gradients become NaN.

### The fix: force mean=0, std≈1 after each block

```python
# LayerNorm(x) = γ * (x - mean) / std + β
#                ↑ scale      ↑ normalize    ↑ shift

mean = x.mean(dim=-1, keepdim=True)    # average across 384 features
std  = x.std(dim=-1, keepdim=True)     # spread across 384 features
x_norm = (x - mean) / (std + 1e-5)    # → mean=0, std=1
return γ * x_norm + β                  # learnable scale + shift
```

`γ` (gamma) and `β` (beta) are 384 learned numbers each — the model can undo the normalization if it wants, but in practice it keeps values stable.

### Why pre-norm (before attention) not post-norm (after)

```
Pre-Norm (modern, used here):
  x = x + Attention( LayerNorm(x) )   ← normalize FIRST, then process

Post-Norm (old, less stable):
  x = LayerNorm( x + Attention(x) )   ← process FIRST, then normalize
```

Pre-norm gives a "clean" input to each sublayer. The residual path (`+ x`) stays un-normalized, providing a gradient highway. Post-norm normalizes the residual too, which can kill the gradient signal.

### Visual: what LayerNorm does to a 384-dim vector

```
Before LN:  [-12.3,  0.5,  45.2,  -8.1,  3.7, ...]  scattered all over
After LN:   [ -0.8,  0.3,   1.2,  -0.5,  0.1, ...]  centered around 0
```

Same information, different scale. The attention + FFN layers expect clean, centered inputs.

---

## 🧠 Model Insights (from Q&A)

### 1. Token Embedding = learned word space
After training, similar words cluster together in the 384-dim space.
`"thương"` and `"yêu"` both appear near `"em"`, `"anh"`, `"lòng"` → vectors drift together.
The model learns relationships purely from context — no dictionary needed.

### 2. Attention = 6 parallel "lenses" (n_head=6)
Each head sees 64 dims of the 384. Multiple heads let the model learn
DIFFERENT types of relationships (syllable count, tone, rhyme, punctuation).
One big 384-dim head would average everything into one blurry relationship.

### 3. FeedForward = the "thinker" (48% of all params)
Attention finds connections. FFN interprets what those connections MEAN.
The GELU non-linearity is critical — without it, stacking more attention layers
just collapses to ONE big linear transform. GELU lets the model learn curves.

### 4. LayerNorm = clean input before every sublayer
Forces mean≈0, std≈1 so weight matrices get the values they were initialized for.
Pre-norm (normalize BEFORE attention/FFN) keeps the residual path un-normalized
→ gradient highway to early layers. Post-norm kills this signal.

### 5. Causal mask prevents CHEATING, not bad meaning
Without mask during training: token 0 can see token 5 (the answer) → loss=0 instantly,
model learns nothing. During generation: there IS no future to look at → model panics.

### 6. Shifted target (y = x shifted by 1) makes training T× faster
Every position produces a loss. With only the last token as target, positions 0..T-1
never get gradients → training is T× slower.

### 7. Shape invariant: (B, T, C) throughout
T stays T throughout all blocks. Only C changes at the final head (384 → vocab_size).
Position embedding is a separate (T, C) tensor broadcast-added — token IDs never change.

### 8. Pad token (id=0) must be suppressed during generation
If model samples `<|pad|>` and code does `continue`, it loops forever sampling pad.
Fix: set `logits[:, pad_id] = -inf` so it's never picked.

---

## 🏋️ Training Insights (from Q&A)

### 1. bfloat16 vs float16 (mixed precision)
`bfloat16` has the same exponent range as float32 → no overflow, no NaN.
`float16` has only 5 exponent bits → values > 65,504 overflow to `inf`.
`bfloat16` needs no GradScaler; `float16` requires constant rescaling to survive.

### 2. model.train() vs model.eval() — only affects Dropout
`.eval()` does NOT stop gradients. It only tells `nn.Dropout` to sleep.
Without `.eval()` during validation: 10% neurons randomly dead → val loss looks
worse than it is → you might stop training too early.

### 3. Gradient clipping = survival mechanism
Without clipping: one bad batch with gradient=50.0 sends a weight from 0.7 to 50.7.
Next softmax overflows → NaN → every weight becomes NaN → model is dead.
With clipping (max=1.0): 50.0 scaled to 1.0 → weight goes 0.7→1.7 → survives.
All gradients are scaled proportionally, so direction is preserved.

### 4. Optimizer state in checkpoints = momentum memory
AdamW keeps running averages of past gradients (momentum). Without saving
optimizer state, resuming training loses all momentum — first few steps are wasted
rebuilding direction from scratch.

### 5. Cosine LR decay
Warmup (steps 0-200): ramp up slowly → prevents early instability.
Decay (steps 200-5000): cosine curve down to 1e-5 → settles into minimum.
Constant LR would bounce around the minimum, never settling.

### 6. drop_last=True on DataLoader
If 100 samples and batch_size=64: two batches (64 + 36).
The 36-sample batch has noisier gradients (fewer samples → worse loss estimate).
`drop_last=True` drops the incomplete batch → every batch is exactly 64 samples.

### 7. Plateau = model capacity limit
Val loss flatlines at ~2.65 for 14.8M model on this data.
Not a bug — the model is saying "I've learned everything I can at this size."
To go lower: bigger model or cleaner data.

---

## 💧 Dropout: How it works

`nn.Dropout(p=0.1)` randomly sets 10% of values to ZERO on every forward pass.

### Concrete example

```
Input vector (8 numbers):
  [ 0.5, -0.3,  1.2,  0.8, -0.1,  0.4, -0.9,  0.7 ]

Dropout(p=0.1) randomly picks ~10% to kill:
  [ 0.5, -0.3,    0,  0.8, -0.1,  0.4, -0.9,    0 ]  ← 2 of 8 zeroed
  
Then scales survivors up by 1/(1-p) = 1/0.9 = 1.111:
  [ 0.56, -0.33,   0,  0.89, -0.11, 0.44, -1.0,   0 ]
  
Why scale? The sum stays roughly the same:
  Before: sum = 2.3
  After:  sum ≈ 2.3  (compensates for missing 10%)
```

### Why it works

Without dropout: the model relies on specific neurons becoming "experts."
Neuron #47 always activates for "thân em" → model overfits to that neuron.

With dropout: neuron #47 sometimes disappears. Other neurons MUST learn to help.
Result: knowledge is spread across many neurons → no single point of failure →
model generalizes instead of memorizing.

### During training vs inference

```
Training:   Dropout kills 10% → forces redundancy → learns robust patterns
Inference:  Dropout disabled → all neurons active → uses full knowledge

Think of it like: during practice, the athlete wears weights on their ankles.
During the race, weights come off — they're faster.
```

### Where dropout lives in this model

```python
# In MultiHeadAttention.__init__:
self.drop_attn = nn.Dropout(0.1)   # kills 10% of attention weights
self.drop_out  = nn.Dropout(0.1)   # kills 10% of attention outputs

# In FeedForward:
self.dropout = nn.Dropout(0.1)     # kills 10% of FFN outputs

# Total: 3 dropout layers × 2 positions per block × 6 blocks = 36 dropout points
# During training: ~3-4 neurons zeroed at each point
# During eval/generation: all zero → full model
```

---

## 🧪 Final Exam Insights

### 1. Cross-entropy loss values tell a story
```
Loss = -ln(P_correct)

9.3 → random guessing among 10,581 tokens
9.2 → model is 0.01% sure, right (still terrible)
4.6 → model is 1% sure, right
3.2 → model has an unfair advantage (weight tying, easy special tokens)
2.7 → model learning grammar + structure
2.65 → PLATEAU — model capacity exhausted at 14.8M
1.0 → model is 37% confident (good for poetry, not for classification)
0.0 → model is 100% sure, right (overfitting?)
∞   → model is 0% sure (P=0), should never happen outside bugs
```

### 2. `ignore_index=-1` in cross_entropy
Positions with `targets == -1` are skipped — no loss, no gradient.
Used for padding: padded positions don't penalize the model.
Currently unused (no padding in our flat-tensor approach) — safety net only.

### 3. Bias is useless in attention, necessary in FFN
Attention: bias adds same offset to ALL scores → softmax result unchanged.
FFN: bias shifts the output curve → lets model fit data that doesn't pass through origin.

### 4. AdamW betas: (0.9, 0.95) not (0.9, 0.999)
```
beta1=0.9:  momentum — smooth gradient direction
beta2=0.95: variance — forget old gradient magnitudes in ~20 steps
            (default 0.999 → forgets in ~1000 steps)

Transformers need 0.95: loss landscape changes rapidly.
NLP needs faster adaptation than images.
```

### 5. GELU vs ReLU — dead neurons
```
GELU(-2.0) = -0.045  → small negative, gradient flows, can recover
ReLU(-2.0) = 0.0     → completely dead, gradient = 0 FOREVER

In deep networks: dead ReLU neurons accumulate → 20-30% wasted params.
GELU prevents this with smooth negative tail.
```

### 6. `@torch.no_grad()` — TWO effects
1. Stops gradient computation (saves compute)
2. Stops Autograd graph storage (saves memory, 3-5× less VRAM during eval)
Does NOT affect dropout — still need `model.eval()`.

### 7. Argmax vs Multinomial (why we sample)
`argmax` always picks the most likely token → identical output every time.
`multinomial` samples from the distribution → 3 runs → 3 different poems.
Temperature controls how spread out that distribution is.

### 8. Causal mask has a HARD limit
Mask is fixed at `block_size × block_size`. If sequence exceeds block_size
without cropping: positions beyond 256 attend to nothing → garbage output.
Must `idx[:, -block_size:]` before every forward pass.

### 9. Pad token loop during generation
```
If model samples token 0 (<|pad|>) and code does `continue`:
  → same context → model picks pad AGAIN → infinite loop
Fix: logits[:, pad_id] = -inf  (never sample pad)
```

### 10. CUDA OOM debugging order
```
1. Reduce batch_size     (192→128→64)   ← immediate relief
2. Reduce block_size     (256→128)      ← cuts attention O(T²)
3. Use gradient accumulation             ← simulate big batch without memory
4. Reduce model size     (n_embd/n_layer)
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
