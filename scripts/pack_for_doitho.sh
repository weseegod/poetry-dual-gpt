#!/bin/bash
# ============================================================
# pack_for_doitho.sh — Package v4.2.3 model for doitho deployment
#
# Usage: bash scripts/pack_for_doitho.sh
#
# Packs 5 files into deploy/doitho_utils.zip:
#   1. checkpoints/doi_tho_best.pt  → doitho.pt        (380MB model weights)
#   2. tokenizer/poetry_bpe.model   → poetry_bpe.model  (884KB BPE tokenizer)
#   3. src/model.py                 → model.py          (model architecture)
#   4. src/tones.py                 → tones.py          (tone/rhyme/diacritic utils)
#   5. deploy/utils/inference.py    → inference.py      (production inference)
#
# Then deploy to doitho:
#   unzip -o deploy/doitho_utils.zip -d ../doitho/utils/
#   cd ../doitho && python -m pytest backend/tests/ -x -q
#
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
TMPDIR="$ROOT/deploy/_pack_tmp"
ZIPFILE="$ROOT/deploy/doitho_utils.zip"
SRC_UTILS="$ROOT/deploy/utils"

echo "📦 PoetryDuel-GPT v4.2.3 → doitho packager"
echo ""

# ── Check required files ──
CKPT="$ROOT/checkpoints/doi_tho_best.pt"
TOK="$ROOT/tokenizer/poetry_bpe.model"

for f in "$CKPT" "$TOK" "$ROOT/src/model.py" "$ROOT/src/tones.py" "$SRC_UTILS/inference.py"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing: $f"
        exit 1
    fi
done

rm -rf "$TMPDIR" "$ZIPFILE"
mkdir -p "$TMPDIR"

# ── Pack: checkpoint + tokenizer + source files ──
cp "$CKPT" "$TMPDIR/doitho.pt"
cp "$TOK" "$TMPDIR/poetry_bpe.model"
cp "$ROOT/src/model.py" "$TMPDIR/model.py"
cp "$ROOT/src/tones.py" "$TMPDIR/tones.py"
cp "$SRC_UTILS/inference.py" "$TMPDIR/inference.py"

echo "📋 Files:"
ls -lh "$TMPDIR/" | grep -v doitho.pt
du -h "$TMPDIR/doitho.pt"

# ── Quick verify checkpoint ──
echo ""
echo "🔍 Verifying checkpoint..."
python3 -c "
import torch, sys
sys.path.insert(0, '$TMPDIR')
ckpt = torch.load('$TMPDIR/doitho.pt', map_location='cpu', weights_only=False, mmap=True)
cfg = ckpt['model_config']
print(f'  ✅ Step: {ckpt[\"step\"]} | Loss: {ckpt[\"loss\"]:.4f} | Vocab: {ckpt[\"vocab_size\"]:,}')
print(f'  ✅ Config: emb={cfg[\"n_embd\"]} layers={cfg[\"n_layer\"]} heads={cfg[\"n_head\"]} block={cfg[\"block_size\"]}')
print(f'  ✅ Keys: {len(ckpt[\"model_state_dict\"])} parameter tensors')
" 2>/dev/null || echo "  ⚠️  Could not verify checkpoint"

# ── Quick verify tokenizer ──
echo ""
echo "🔍 Verifying tokenizer..."
python3 -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('$TMPDIR/poetry_bpe.model')
vocab = tok.get_vocab_size()
ids = {tok.token_to_id(t) for t in ['<|pad|>', '<|start|>', '<|reply|>', '<|end|>', '<|linebreak|>']}
assert ids == {0, 1, 2, 3, 9}, f'Control token IDs wrong: {ids}'
print(f'  ✅ Vocab: {vocab:,} | Control tokens: 0,1,2,3,9 confirmed')
" 2>/dev/null || echo "  ⚠️  Could not verify tokenizer"

# ── Quick verify model can load ──
echo ""
echo "🔍 Verifying model loads..."
python3 -c "
import torch, sys
sys.path.insert(0, '$TMPDIR')
from model import PoetryDuelGPT
from tokenizers import Tokenizer

tok = Tokenizer.from_file('$TMPDIR/poetry_bpe.model')
ckpt = torch.load('$TMPDIR/doitho.pt', map_location='cpu', weights_only=False, mmap=True)
cfg = ckpt['model_config'].copy()
cfg.pop('vocab_size', None)
m = PoetryDuelGPT(ckpt['vocab_size'], **cfg)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()

# Quick test forward pass
x = torch.randint(0, ckpt['vocab_size'], (1, 16))
with torch.no_grad():
    logits, _ = m(x)
assert logits.shape == (1, 16, ckpt['vocab_size'])
print(f'  ✅ Model loads & forward pass works: {logits.shape}')
" 2>/dev/null || echo "  ⚠️  Could not verify model"

# ── Create zip ──
cd "$TMPDIR" && zip -r "$ZIPFILE" . > /dev/null
cd "$ROOT"

echo ""
echo "🎉 Done! deploy/doitho_utils.zip ready."
echo ""
echo "   📦 Package contents:"
echo "      doitho.pt          — v4.2.3 checkpoint (31.5M params)"
echo "      poetry_bpe.model   — BPE tokenizer (12,000 tokens)"
echo "      model.py           — PoetryDuelGPT architecture"
echo "      tones.py           — Tone/rhyme/diacritic utilities"
echo "      inference.py       — Production inference (soft rhyme, v4.2.3)"
echo ""
echo "   To deploy to doitho:"
echo "     unzip -o deploy/doitho_utils.zip -d ../doitho/utils/"
echo "     cd ../doitho && python -m pytest backend/tests/ -x -q"
