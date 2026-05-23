# 🎭 PoetryDuel-GPT — Vietnamese Đối Thơ AI

A 31M-parameter GPT-style Transformer built from scratch in raw PyTorch for Vietnamese poetry dueling (đối thơ). Zero HuggingFace wrappers. Input a Lục Bát couplet, get a matching response couplet back — with rhyme chaining, tone pattern conditioning, and syllable enforcement.

**Live at: [doitho.net](https://doitho.net)**

---

## 🚀 Quick Start

```bash
git clone https://github.com/weseegod/poetry-dual-gpt.git
cd poetry-dual-gpt
pip install -r requirements.txt

# Train tokenizer + model (or download checkpoint from Releases)
python src/train_bpe.py --corpus data/doi_tho_corpus.txt
python src/train.py --mode train --name doi_tho_ --corpus data/doi_tho_corpus.txt

# Generate poetry
python src/sample.py --checkpoint checkpoints/doi_tho_best.pt

# Chat UI
cd client && python start.py
```

---

## 📊 Model

| | |
|---|---|
| **Params** | 31.5M |
| **Architecture** | Decoder-only Transformer, 8 layers, 8 heads, n_embd=512, block_size=256 |
| **Vocabulary** | 12,000 ByteLevel BPE tokens |
| **Training** | Single-stage, 541K đối thơ pairs, example-aligned batches, batch=192 |
| **Format** | `[DOI_THO] [RHYME:X] [TONE:XXXXXX]` with `<\|linebreak\|>` line separator |

## 🎯 Results (v3, 15 diverse prompts)

| Rule | Accuracy |
|------|----------|
| **Stress test** (valid output, 14 prompts) | **100%** |
| **BPE collapse** (garbled tokens) | **0%** |
| **Rhyme** (vần, pos 6 matches `[RHYME:X]`) | **80%** |
| **Tone** (B-T-B-B pattern) | **97%** |
| **Syllable** (exact 6+8 enforced) | **93%** |

## 🎭 Sample Output

```
Input:   Thân em như chẽn lúa đòng
         Phất phơ dưới ngọn nắng hồng ban mai

Output:  Mẹ già ăn sắn ngô đồng nương thu
         Con ăn cơm nước, anh về chợ quê

Input:   Công cha như núi thái sơn

Output:  Sơn hà hoa đẹp ngọt hơn hoa sen
         Rừng xanh xanh thẳm bên miền hương linh

Input:   Gió đưa cành trúc la đà
         Tiếng chuông Trấn Vũ canh gà Thọ Xương

Output:  Ba đình chuông vọng tơ vương
         Nghe như tiếng vọng chùa thương quyện hòa
```

## 🧠 How It Works

The model sees `[RHYME:X]` and `[TONE:XXXXXX]` tags extracted from the user's input couplet. It learns via attention to condition its response on these tags — position 6 of the response 8-syl line must rhyme with `[RHYME:X]`, and positions 2,4,6,8 must follow the B-T-B-B tone pattern.

Multi-couplet input gets independent responses per couplet (C1→C3, C2→C4). Syllables are enforced to 6/8 with post-processing truncation. Repetition is penalized during sampling for lexical diversity.

## 📚 Dataset

Training data derived from [phamson02/vietnamese-poetry-corpus](https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/) on HuggingFace. We preprocess Lục Bát poems into couplet-to-couplet sliding pairs with rhyme and tone tags injected.

## 📄 License

MIT
