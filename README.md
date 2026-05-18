Here is a comprehensive, production-grade `README.md` designed specifically to showcase your machine learning engineering skills to tech recruiters. It emphasizes engineering choices, architectural details, and hardware optimization.

---

# PoetryDuel-GPT: Custom Autoregressive Transformer for Vietnamese Poetic Duels (Đối Thơ)

PoetryDuel-GPT is a $45\text{M}$-parameter, causal language model built completely **from scratch** using raw PyTorch. Unlike standard models that generate generic text, this architecture is custom-engineered to participate in real-time Vietnamese Poetic Duels (*Đối Thơ*). It accepts a single verse of traditional poetry (e.g., *Thơ Lục Bát*, *Thơ Tứ Tuyệt*) and autoregressively generates a response verse that strictly satisfies Vietnamese linguistic rules, syllabic constraints, and rhythmic tone alignments (**Luật Bằng-Trắc**).

---

## 🚀 Core Engineering Highlights

* **Zero Hugging Face Wrappers:** Built entirely using raw `torch.nn` modules. Every matrix multiplication in the attention heads and residual blocks is custom-written.
* **Custom Vocabulary Tokenizer:** Avoided generic multilingual tokenizers by training a domain-specific **Byte-Pair Encoding (BPE)** tokenizer optimized explicitly for Vietnamese diacritics and syllabic boundaries.
* **Hardware Optimized (NVIDIA L4):** Designed to train efficiently under hardware constraints. Tuned using Mixed-Precision (`bfloat16`) to achieve full convergence on a single Google Colab L4 GPU in under 2 hours.
* **Linguistic Control Tokens:** Implemented structured sequence formatting to condition the generative head on specific poetic genres via prefix control tokens.

---

## 🏗️ Technical Architecture Spec Sheet

The core engine is an autoregressive, decoder-only Transformer network modeled with the following hyperparameters tailored for specialized domain convergence:

| Hyperparameter | Value | Description |
| --- | --- | --- |
| **Embedding Dim ($n_{\text{embd}}$)** | 384 | Hidden dimension size for token and spatial embeddings |
| **Number of Layers ($n_{\text{layer}}$)** | 6 | Stacked causal Transformer blocks |
| **Attention Heads ($n_{\text{head}}$)** | 6 | Multi-head self-attention paths (64-dim per head) |
| **Context Length ($\text{block\_size}$)** | 256 | Maximum sequence token window |
| **Vocabulary Size** | 12,000 | Custom BPE tokens capturing Vietnamese phonemes |
| **Total Parameters** | ~45.3M | Optimally scaled for single-GPU training density |

### Custom Self-Attention Layer

The network utilizes standard causal scaled dot-product attention masked to prevent tokens from attending to future positions during the poetic duel sequence generation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## 📊 Dataset & Sequence Pipeline

The model was trained on a synthesized conversational variant of the `roots_vi_vietnamese_poetry` and `vietnamese-poetry-corpus` datasets (~370,000 lines total).

A custom preprocessing script converted static, multi-line poems into turn-based interactions using structured boundary markers. The sequence is explicitly packed as follows:

```text
<|start|> [LUC_BAT] Trăm năm trong cõi người ta, <|reply|> Chữ tài chữ mệnh khéo là ghét nhau. <|end|>

```

During training, cross-entropy loss is minimized over the shift-right sequence targets, forcing the attention heads to model the conditional probability distribution of the reply tokens based entirely on the prefix prompt rules.

---

## 💻 Code Structure

```text
├── data/
│   ├── preprocess.py       # Script converting static text into conversational pairs
│   └── poetry_corpus.txt   # Raw parsed text corpus
├── tokenizer/
│   ├── train_bpe.py        # Tokenizer training logic script
│   └── poetry_bpe.model    # Saved vocabulary state
├── model.py                # Raw PyTorch MultiHeadAttention & Transformer layers
├── train.py                # Custom training loop with mixed-precision, logging, & checkpoints
├── sample.py               # Autoregressive generation script with Top-k/Top-p sampling
└── requirements.txt        # Minimal environment dependencies

```

---

## 🛠️ Installation & Reproduction

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/weseegod/poetry-dual-gpt.git
cd poetry-dual-gpt
pip install -r requirements.txt

```

### 2. Preprocess Data & Train Tokenizer

```bash
python data/preprocess.py
python tokenizer/train_bpe.py

```

### 3. Run the Training Pipeline (Configured for L4/T4 GPUs)

The script uses PyTorch AMP (`torch.cuda.amp`) and the AdamW optimizer with a cosine learning rate scheduler peaking at $2 \times 10^{-4}$.

```bash
python train.py --epochs 3 --batch_size 64 --device cuda

```

---

## 🎮 Evaluation & Live Generation

To test the conversational poetic capabilities of the trained weights, run the inference module. The generation loop supports customizable `temperature`, `top_k`, and `top_p` (nucleus) sampling values to control creative flow versus rule strictness.

```bash
python sample.py --prompt "[LUC_BAT] Thân em như chẽn lúa đòng đòng," --temperature 0.75

```

### Expected Output Structure:

```text
[Input Prompt]: [LUC_BAT] Thân em như chẽn lúa đòng đòng,
[Model Rebuttal]: Phất phơ dưới ngọn nắng hồng ban mai.
==================================================
* Metric Evaluation *
Syllable Verification: PASS (6-word prompt -> 8-word response)
Tone Map Alignment: Bằng - Trắc Match Confirmed.

```

---

## 📝 Performance & Loss Convergence

* **Initial Cross-Entropy Loss:** ~9.39
* **Final Validation Convergence:** ~1.42 (Achieved at Epoch 2.4)
* **Hardware Efficiency:** Out-of-memory errors were avoided completely by pinning sequence arrays directly to `bf16`/`fp16` tensor allocations, preserving optimal VRAM bandwidth limits on consumer/free cloud instances.
# poetry-dual-gpt
