---
language: vi
tags:
  - vietnamese
  - poetry
  - transformer
  - gpt
  - pytorch
  - text-generation
  - luc-bat
  - doi-tho
license: mit
model-index:
  - name: poetry-dual-gpt
    results:
      - task:
          type: text-generation
        metrics:
          - type: rhyme-accuracy
            value: 92.2
          - type: tone-accuracy
            value: 100.0
          - type: syllable-accuracy
            value: 100.0
          - type: tram-bong-accuracy
            value: 99.1
          - type: all-5-rules
            value: 91.4
datasets:
  - phamson02/vietnamese-poetry-corpus
---

# PoetryDuel-GPT

A 31.5M-parameter GPT-style Transformer built from scratch in raw PyTorch for Vietnamese poetry dueling (đối thơ). Zero HuggingFace wrappers — `torch.nn` only.

## Model Description

PoetryDuel-GPT performs **đối thơ** (Vietnamese poetry dueling): given a Lục Bát couplet, it generates a matching response couplet that follows all five classic Lục Bát rules — rhyme (vần lưng), tone pattern (bằng-trắc), syllable count (6+8), trầm-bổng (diacritic contrast), and rhythm (nhịp điệu).

The model uses **control token conditioning**: special tokens like `[RHYME:ong]`, `[TONE:BBBTTB]`, and `[TRAMBONG:NH]` are injected into the prompt, and the model learns via self-attention to condition its output on these constraints.

## Intended Use

- Vietnamese poetry generation / đối thơ
- Educational tool for learning Lục Bát poetry rules
- NLP research on constrained text generation with small language models

## Performance

Evaluated on 116 held-out Lục Bát couplet prompts (v4.2.3):

| Rule | Accuracy |
|------|----------|
| All 5 rules simultaneously | **91.4%** |
| R1: Rhyme (vần lưng) | 92.2% |
| R2: Tone (B-T-B-B) | 100.0% |
| R3: Syllable (6+8) | 100.0% |
| R4: Trầm-Bổng | 99.1% |
| R5: Rhythm | 100.0% |

## Training Data

Derived from [phamson02/vietnamese-poetry-corpus](https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/):
- 84,000 Lục Bát poems → 540,000 couplet-to-couplet training pairs
- Rhyme, tone, and Trầm-Bổng tags extracted per example
- Example-aligned batching to prevent cross-poem noise

## Training Procedure

- **Hardware:** Google Colab T4 GPU (~3 hours)
- **Steps:** 10,000 training steps
- **Batch size:** 192
- **Optimizer:** AdamW (β₁=0.9, β₂=0.95), cosine LR 3e-4 → 1e-5, warmup=500
- **Precision:** bfloat16 mixed precision
- **Tokenizer:** 12,000 BPE tokens trained on the poetry corpus

## Usage

```python
import torch
from tokenizers import Tokenizer
from src.model import PoetryDuelGPT
from src.generation import build_prompt, generate, decode_response

# Load model
tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")
model = PoetryDuelGPT(vocab_size=tok.get_vocab_size())
ckpt = torch.load("checkpoints/doi_tho_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate
prompt = build_prompt("Thân em như chẽn lúa đòng\nPhất phơ dưới ngọn nắng hồng ban mai")
gen_ids, output = generate(model, tok, prompt, temperature=0.75)
response = decode_response(tok, gen_ids)
print(response)
```

## Limitations

- Only supports Lục Bát genre (6-8 syllable couplets)
- Trained on classical/folk Vietnamese poetry — may not handle modern slang
- 31.5M parameter ceiling — semantic richness limited compared to larger models
- No multi-turn conversation support; each generation is independent

## Citation

```bibtex
@software{poetry_dual_gpt,
  author = {Thanh},
  title = {PoetryDuel-GPT: A Transformer for Vietnamese Poetry Dueling},
  year = {2026},
  url = {https://github.com/weseegod/poetry-dual-gpt},
}
```
