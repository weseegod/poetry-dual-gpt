<div align="center">

# 🎭 PoetryDuel-GPT

*Vietnamese Lục Bát Poetry Generation — from scratch Transformers to QLoRA Instruct models*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Models](https://img.shields.io/badge/🤗%20HF-Models-yellow)](https://huggingface.co/grindytech)

</div>

Two approaches to the same problem — generating rule-compliant Lục Bát poetry couplets:

| | v5.1 Instruct (QLoRA) | v4.2.3 (From Scratch) |
|---|---|---|
| **Base model** | Qwen2.5-1.5B-Instruct | 31.5M custom GPT |
| **Method** | QLoRA fine-tuning | Trained from scratch |
| **Params** | 1.5B (18.5M trainable) | 31.5M |
| **All-5 accuracy** | 70.0% | 91.4% |
| **HF Model** | [grindytech/poetry-dual-gpt-instruct-v5.1](https://huggingface.co/grindytech/poetry-dual-gpt-instruct-v5.1) | [grindytech/poetry-dual-gpt](https://huggingface.co/grindytech/poetry-dual-gpt) |

> 🚀 Try it live at **[doitho.net](https://doitho.net)**

---

## 🆕 v5.1 Instruct (QLoRA on Qwen2.5-1.5B-Instruct)

Instruction-tuned with chat template. Given a Lục line, generates the matching Bát line.

### Evaluation (30 prompts, temp=0.6)

| Rule | Accuracy | Target |
|------|----------|--------|
| R1 Rhyme (vần lưng) | 80.0% | 85% |
| R2 Tone (B-T-B-B) | 97.5% | 90% |
| R3 Syllable count (=8) | 100% | 90% |
| R4 Trầm-Bổng | 93.3% | 85% |
| R5 Rhythm | 100% | 90% |
| **All 5 rules** | **70.0%** | 70% |

### Quick Start

```bash
# Setup & train
bash run.sh setup
bash run.sh train --batch-size 8 --max-steps 7000

# Resume from checkpoint
bash run.sh train --batch-size 8 --max-steps 10000 --resume checkpoints/instruct_best

# Evaluate
python evaluate/eval_instruct.py --checkpoint checkpoints/instruct_best
```

See [`checkpoints/instruct_best/README.md`](checkpoints/instruct_best/README.md) for Python inference code.

---

## 🏗️ v4.2.3 — From-Scratch Transformer

**PoetryDuel-GPT** is a GPT-style autoregressive Transformer built from scratch in raw PyTorch — zero HuggingFace wrappers. It performs **đối thơ** (Vietnamese poetry dueling): given one or two lines of Lục Bát poetry, the model responds with a matching couplet that follows rhyme, tone, and syllable constraints.

---

## ✨ Features

- **Pure PyTorch** — No `transformers`, no `trl`, just `torch.nn`
- **5 Lục Bát rules** enforced via control tokens: rhyme (vần lưng), tone (bằng-trắc), syllable count, trầm-bổng (diacritic pairs), rhythm
- **Generation pipeline** — Soft rhyme biasing, repetition penalty, top-k/top-p, generate-and-rerank
- **FastAPI + React** chat UI included
- **Colab-ready** — Single-click training from scratch

---

## 📊 Results

Evaluated on 116 held-out Lục Bát couplet prompts (v4.2.3):

| Rule | Accuracy | Random Baseline |
|------|----------|-----------------|
| **All 5 rules simultaneously** | **91.4%** | 0.0% |
| Rhyme (vần lưng) | 92.2% | 0.6% |
| Tone (B-T-B-B pattern) | 100.0% | 6.2% |
| Syllable (exact 6+8) | 100.0% | 7.0% |
| Trầm-Bổng (Ngang ≠ Huyền) | 99.1% | 50.0% |
| Rhythm (nhịp điệu) | 100.0% | 7.0% |

| Quality Metric | Value |
|----------------|-------|
| Lexical diversity | 0.962 |
| Average response length | 14.0 syllables |
| Empty/short response rate | 0.0% |

### Sample Output

```
Input:  Công cha như núi thái sơn
        Nghĩa mẹ như nước trong nguồn chảy ra

Output: Dù cho vật đổi sao dời
        Nhưng dân vẫn Việt bốn phương rạng ngời

Input:  Thân em như chẽn lúa đòng
        Phất phơ dưới ngọn nắng hồng ban mai

Output: Bao la trời đất cao vời
        Tình em tha thiết muôn lời nhớ thương
```

---

## 🏗️ Model Architecture

| | Specification |
|---|---|
| **Type** | Decoder-only Transformer (GPT-style) |
| **Parameters** | 31.5M |
| **Layers** | 8 |
| **Attention heads** | 8 |
| **Embedding dimension** | 512 |
| **Context window** | 256 tokens |
| **Vocabulary** | 12,000 Byte-Level BPE tokens |
| **Training framework** | Raw PyTorch (`torch.nn`) |
| **Mixed precision** | bfloat16 |
| **Optimizer** | AdamW, cosine LR schedule, gradient clipping |

### Training Data Format

```
<|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]
  câu lục <|linebreak|> câu bát <|reply|>
  câu lục <|linebreak|> câu bát <|end|>
```

Control tokens condition the model on rhyme target, tone pattern, and tonal contrast — enabling rule-compliant generation without external rule engines.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/weseegod/poetry-dual-gpt.git
cd poetry-dual-gpt
pip install -r requirements.txt
```

### Download Checkpoint

```bash
# Option 1: From HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('grindytech/poetry-dual-gpt', 'doi_tho_best.pt', local_dir='checkpoints')
hf_hub_download('grindytech/poetry-dual-gpt', 'poetry_bpe.model', local_dir='tokenizer')
"

# Option 2: Train from scratch (see Training section)
```

### Generate Poetry

```bash
# Single prompt
python src/sample.py --checkpoint checkpoints/doi_tho_best.pt

# Interactive mode
python src/sample.py --interactive
```

### Chat UI

```bash
cd client
python start.py
# Opens at http://localhost:8000
# Frontend at http://localhost:3000
```

---

## 🔬 How It Works

### Control Token Conditioning

The model is trained with an explicit format that encodes poetic rules as special tokens:

```
<|start|> [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] [TRAMBONG:NH]
```

During inference, tags are extracted from the user's input couplet:
- `[RHYME:X]` — Target rhyme group from position 8 of the input's second line
- `[TONE:XXXXXX]` — Bằng/Trắc tone pattern from the input's first line
- `[TRAMBONG:NH/HN]` — Trầm-Bổng (Ngang→Huyền or Huyền→Ngang)

The model attends to these tags via standard self-attention, learning to generate output whose rhyme, tone, and diacritic patterns match the constraints.

### Lục Bát Rules Enforced

| Rule | Description | Implementation |
|------|-------------|----------------|
| **R1: Vần lưng** | Position 6 of output Lục rhymes with position 6 of output Bát | Soft rhyme biasing during generation |
| **R2: Bằng-Trắc** | BTB (Lục) + BTBB (Bát) tone pattern at even positions | `[TONE:XXXXXX]` control token |
| **R3: Syllable** | Exact 6+8 syllable count | Learned from training data |
| **R4: Trầm-Bổng** | Position 6 & 8 of Bát must differ in Ngang vs Huyền | `[TRAMBONG:NH/HN]` control token |
| **R5: Nhịp điệu** | 2/2/2 or 4/4 rhythm structure | Implicit in data |

### Generation Features

- **Soft rhyme constraint** — Rhyme candidates get +2.0 logit boost instead of hard masking
- **Repetition penalty** — Recent 16 tokens penalized (-1.2) for lexical diversity
- **Top-k + Top-p** — Dual filtering for quality/diversity balance
- **Generate-and-rerank** — N candidates scored by lexical diversity, BPE validity, repetition

---

## 📁 Project Structure

```
poetry-dual-gpt/
├── src/                    # Model, training, generation, preprocessing
│   ├── model.py            # PoetryDuelGPT — decoder-only Transformer
│   ├── train.py            # Training loop with cosine LR, mixed precision
│   ├── generation.py       # Canonical generation (CLI + API + eval)
│   ├── sample.py           # CLI demo with interactive mode
│   ├── dataset.py          # Example-aligned batching, curriculum datasets
│   ├── tones.py            # Tone classification, rhyme extraction
│   ├── preprocess.py       # Data preprocessing pipeline
│   └── train_bpe.py        # BPE tokenizer training
├── client/                 # FastAPI backend + React frontend
│   ├── server.py           # /chat endpoint with model inference
│   └── frontend/           # React chat UI
├── evaluate/               # Evaluation suite
│   ├── eval_rules.py       # 5-rule structural evaluation
│   └── eval_quality.py     # Semantic quality metrics
├── tests/                  # Unit tests
├── scripts/                # Deployment scripts
└── checkpoints/            # Model weights (download from HF)
```

---

## 🔧 Training

### From Scratch

```bash
# 1. Download & preprocess dataset
python src/preprocess.py

# 2. Train BPE tokenizer (12K vocab)
python src/train_bpe.py --corpus data/poetry_corpus.txt

# 3. Train the model (~10K steps, ~3h on T4)
python src/train.py --mode train --name doi_tho_ --corpus data/poetry_corpus.txt
```

### Google Colab

Use `colab/colab_train.ipynb` for one-click training on free Colab T4 GPU.

### Training Config

| Parameter | Value |
|-----------|-------|
| Max steps | 10,000 |
| Batch size | 192 |
| Learning rate | 3e-4 → 1e-5 (cosine) |
| Warmup | 500 steps |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |

---

## 📚 Dataset

Training data derived from [**phamson02/vietnamese-poetry-corpus**](https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/) on HuggingFace.

- **84,000 Lục Bát poems** preprocessed into 540,000 window-1 couplet-to-couplet pairs
- Rhyme (from position 8), tone (BTB pattern), and Trầm-Bổng tags extracted per example
- Example-aligned batching — zero cross-poem noise

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{poetry_dual_gpt,
  author = {Thanh},
  title = {PoetryDuel-GPT: A Transformer for Vietnamese Poetry Dueling},
  year = {2026},
  url = {https://github.com/weseegod/poetry-dual-gpt},
}
```

---

## 🙏 Acknowledgments

- [**phamson02/vietnamese-poetry-corpus**](https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/) for the training dataset
- [**Andrej Karpathy's nanoGPT**](https://github.com/karpathy/nanoGPT) for architectural inspiration
- The Vietnamese poetry community for preserving Lục Bát tradition
