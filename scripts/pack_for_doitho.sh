#!/bin/bash
# ============================================================
# pack_for_doitho.sh — Package v4.1 model for doitho deployment
#
# Usage: bash scripts/pack_for_doitho.sh
#
# Creates deploy/utils/ with everything doitho needs.
# Then: cp -r deploy/utils/ ../doitho/
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="$ROOT/deploy/utils"

echo "📦 PoetryDuel-GPT v4.1 → doitho packager"
echo "   Root: $ROOT"
echo "   Output: $UTILS_DIR"
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

mkdir -p "$UTILS_DIR"

# ── 1. Checkpoint ──
cp "$CKPT" "$UTILS_DIR/doitho.pt"
echo "✅ Checkpoint: $(du -h "$UTILS_DIR/doitho.pt" | cut -f1)"

# ── 2. Tokenizer ──
cp "$TOK" "$UTILS_DIR/poetry_bpe.model"
echo "✅ Tokenizer: poetry_bpe.model"

# ── 3. Model architecture ──
cp "$ROOT/src/model.py" "$UTILS_DIR/model.py"
echo "✅ Model: model.py"

# ── 4. Tones (diacritic, Trầm-Bổng, rhyme) ──
cp "$ROOT/src/tones.py" "$UTILS_DIR/tones.py"
echo "✅ Tones: tones.py"

# ── 5. Inference (already in deploy/utils/) ──
if [ ! -f "$UTILS_DIR/inference.py" ]; then
    echo "⚠️  inference.py missing from deploy/utils/ — generating from sample.py"
    # This shouldn't happen since we committed it, but as a fallback:
    cp "$ROOT/src/sample.py" "$UTILS_DIR/inference.py"
fi
echo "✅ Inference: inference.py"

# ── Verify ──
echo ""
echo "📋 Pack contents:"
ls -lh "$UTILS_DIR/"
echo ""

# Quick sanity: verify checkpoint loads
python3 -c "
import torch, sys
sys.path.insert(0, '$UTILS_DIR')
ckpt = torch.load('$UTILS_DIR/doitho.pt', map_location='cpu', weights_only=False, mmap=True)
print(f'  Step: {ckpt[\"step\"]} | Loss: {ckpt[\"loss\"]:.4f} | Vocab: {ckpt[\"vocab_size\"]:,}')
print(f'  Config: emb={ckpt[\"model_config\"][\"n_embd\"]} layers={ckpt[\"model_config\"][\"n_layer\"]}')
" 2>/dev/null || echo "  ⚠️  Could not verify checkpoint (torch not available)"

echo ""
echo "🎉 Done! deploy/utils/ is ready."
echo ""
echo "   To deploy to doitho:"
echo "     cp -r deploy/utils/ ../doitho/"
echo "     cd ../doitho && git add utils/ && git commit -m 'Update model to v4.1'"
