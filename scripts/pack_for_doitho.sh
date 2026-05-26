#!/bin/bash
# ============================================================
# pack_for_doitho.sh — Package v4.1 model for doitho deployment
#
# Usage: bash scripts/pack_for_doitho.sh
#
# Creates deploy/doitho_utils.zip containing everything doitho needs.
# Then: unzip -o deploy/doitho_utils.zip -d ../doitho/utils/
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
TMPDIR="$ROOT/deploy/_pack_tmp"
ZIPFILE="$ROOT/deploy/doitho_utils.zip"
SRC_UTILS="$ROOT/deploy/utils"

echo "📦 PoetryDuel-GPT v4.1 → doitho packager"
echo ""

# ── Check required files ──
CKPT="$ROOT/checkpoints/doi_tho_best.pt"
TOK="$ROOT/tokenizer/poetry_bpe.model"

for f in "$CKPT" "$TOK"; do
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

# ── Quick verify ──
python3 -c "
import torch, sys
sys.path.insert(0, '$TMPDIR')
ckpt = torch.load('$TMPDIR/doitho.pt', map_location='cpu', weights_only=False, mmap=True)
print(f'  Step: {ckpt[\"step\"]} | Loss: {ckpt[\"loss\"]:.4f} | Vocab: {ckpt[\"vocab_size\"]:,}')
print(f'  Config: emb={ckpt[\"model_config\"][\"n_embd\"]} layers={ckpt[\"model_config\"][\"n_layer\"]}')
" 2>/dev/null || echo "  ⚠️  Could not verify checkpoint"

# ── Create zip ──
cd "$TMPDIR" && zip -r "$ZIPFILE" . > /dev/null
cd "$ROOT"

echo ""
echo "🎉 Done! deploy/doitho_utils.zip ready."
echo ""
echo "   To deploy to doitho:"
echo "     unzip -o deploy/doitho_utils.zip -d ../doitho/utils/"
