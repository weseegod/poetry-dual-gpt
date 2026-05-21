# 🎭 PoetryDuel-GPT — Vietnamese Poetry Generator

A 31M-parameter GPT-style Transformer built from scratch in raw PyTorch for Vietnamese poetry generation (Lục Bát 6→8, Thất Ngôn 7→7). Zero HuggingFace wrappers. Trained with rhyme, tone, and tonal contrast conditioning via 335 custom special tokens.

**[v1.0](https://github.com/weseegod/poetry-dual-gpt/releases/tag/v1.0)** — [Colab Notebook](https://colab.research.google.com/github/weseegod/poetry-dual-gpt/blob/main/colab/colab_train.ipynb)

---

## 🚀 Quick Start

```bash
git clone https://github.com/weseegod/poetry-dual-gpt.git
cd poetry-dual-gpt
pip install -r requirements.txt

# Download pretrained checkpoint + tokenizer (from Releases) or train:
# python src/train_bpe.py && python src/train.py

# Generate poetry
PYTHONPATH=. python src/sample.py --checkpoint checkpoints/stage2_best.pt

# Chat UI
cd client && python start.py
```

---

## 📊 Model

| | |
|---|---|
| **Params** | 31.2M |
| **Architecture** | Decoder-only Transformer, 8 layers, 8 heads, n_embd=512, block_size=256 |
| **Vocabulary** | 11,392 BPE tokens (335 special control tokens) |
| **Training** | Two-stage: Stage 1 (all genres, 10K steps) → Stage 2 (Lục Bát, 5K steps) |
| **Hardware** | Colab T4/L4, ~3 hours total |
| **Mixed precision** | bfloat16 |

## 🎯 Rule Accuracy (v1.0, 173 novel prompts)

| Rule | Token | Accuracy | vs Random |
|------|-------|----------|-----------|
| **Rhyme** (vần lưng) | `[RHYME:X]` | **58.4%** | 93× |
| **Tone** (B-T-B-B) | `[TONE:XXXXXX]` | **87.5%** | 14× |
| **Syllable** (exact 8) | form + truncation | **78.0%** | 12× |
| **Đối Âm** (tonal contrast) | `[DOIAM:XXXXXXX]` | **69.4%** | 1.4× |
| **All 3 rules pass** | — | **50.9%** | — |

[Full evaluation](documents/rule_evaluation.md)

## 🎭 Sample Output

```
Prompt:  Thân em như chẽn lúa đòng
Tags:    [LUC_BAT] [RHYME:ong] [TONE:BBBTTB]
Output:  anh về em lúa trổ đòng vàng hong phơi

Prompt:  Rủ nhau xuống biển mò cua
Output:  tôi đây vẫn giữ canh chua cá vàng
```

## 🏗️ Training Format

```
[LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng <|reply|> response_8_syl <|end|>
[THAT_NGON] [LINK2:B] [DOIAM:TTBBTTB] Lom khom dưới núi tiều vài chú <|reply|> response_7_syl <|end|>
```

All control tokens (`[RHYME:X]`, `[TONE:XXXXXX]`, `[DOIAM:XXXXXXX]`, `[LINK2:X]`) are single special token IDs — not BPE subwords. 335 total, auto-collected from the corpus.

## 📁 Project Structure

```
src/
  model.py          # Transformer (attention, FFN, blocks)
  train.py          # Training loop with resume, patience, mixed precision
  sample.py         # Autoregressive generation + rule checking
  preprocess.py     # CSV → training pairs with control tokens
  train_bpe.py      # BPE tokenizer with 335 special tokens
  tones.py          # Vietnamese tone classification + rhyme extraction
  clean_data.py     # Data cleaning pipeline
  dataset.py        # PyTorch Dataset + DataLoader
evaluate/
  eval_rules.py     # Per-rule evaluation on novel prompts
client/
  server.py         # FastAPI backend
  frontend/         # React chat UI
  start.py          # Launch backend + frontend
colab/
  colab_train.ipynb # One-click Colab training with verification gates
documents/
  roadmap.md        # Learning guide (Transformer from scratch)
  improvements.md   # v2.0 roadmap
  rhyme_conditioning.md  # Rule implementation details
  rule_evaluation.md     # v1.0 evaluation results
checkpoints/        # Trained model weights
data/               # poems_dataset_clean.csv, poetry_corpus.txt
tokenizer/          # poetry_bpe.model
```

## 🛤️ v2.0 Roadmap

See [improvements.md](documents/improvements.md)

| # | Item | Impact |
|---|------|--------|
| 1 | Qwen2.5-1.5B QLoRA | 50× more params, pretrained Vietnamese |
| 2 | Multi-couplet generation | Full 4-8 line poems |
| 3 | Better training data | Ca dao, Truyện Kiều, modern poetry |

## 📄 License

MIT
