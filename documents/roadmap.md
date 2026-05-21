# 🗺️ PoetryDuel-GPT Roadmap

> v1.0 shipped — 31M-param Vietnamese poetry generator with rhyme, tone, and tonal contrast conditioning.

---

## 📋 Project

| | |
|---|---|
| **Model** | Decoder-only Transformer, 31.2M params, n_embd=512, 8 layers, 8 heads |
| **Vocabulary** | 11,392 BPE tokens (335 special control tokens) |
| **Training** | Two-stage: Stage 1 (all genres, 10K steps) → Stage 2 (Lục Bát, 5K steps) |
| **Data** | 135K poems → 942K training pairs |
| **Framework** | Raw PyTorch (`torch.nn`), zero HuggingFace |
| **Hardware** | Colab T4/L4, ~3 hours |

## 🎯 Rules (v1.0)

| Rule | Token | Accuracy |
|------|-------|----------|
| Rhyme (vần lưng) | `[RHYME:X]` | 58.4% |
| Tone (B-T-B-B) | `[TONE:XXXXXX]` | 87.5% |
| Syllable (exact 8) | truncation | 78.0% |
| Đối Âm (tonal contrast) | `[DOIAM:XXXXXXX]` | 69.4% |

Full: [rule_evaluation.md](rule_evaluation.md)

## 📁 Structure

```
src/          # Model, training, generation, preprocessing, tokenizer
evaluate/     # Per-rule evaluation scripts
client/       # FastAPI + React chat UI
colab/        # One-click training notebook
documents/    # Docs (roadmap, evaluation, rule design)
checkpoints/  # Trained weights
tokenizer/    # BPE model
data/         # Corpus + CSV
```

## 🛤️ v2.0

See [roadmap_v2.md](roadmap_v2.md) — Qwen2.5-1.5B QLoRA, multi-couplet generation, better data.
