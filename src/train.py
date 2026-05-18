"""
train.py — Training loop for PoetryDuelGPT with mixed precision.

Phase 1: Lục Bát only. Single GPU. Banner + trend tracking + best model save.

Modes:
  "test"  — 200 steps, batch=8, quick smoke test
  "train" — 5000 steps, batch=64, full training

Run:
    python src/train.py              # full training
    python src/train.py --mode test  # quick test
"""

import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import PoetryDuelGPT
from dataset import PoetryDataset, tokenize_corpus, get_dataloaders


ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",  # "test" or "train"

    # Model (dynamic vocab from tokenizer)
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "block_size": 256,
    "dropout": 0.1,

    # Paths
    "corpus_path": "data/poetry_corpus.txt",
    "tokenizer_path": "tokenizer/poetry_bpe.model",
    "checkpoint_dir": "checkpoints",

    # Test mode
    "test": {
        "max_steps": 200,
        "batch_size": 8,
        "val_fraction": 0.05,
        "log_interval": 10,
        "eval_interval": 100,
        "save_interval": 200,
    },

    # Train mode
    "train": {
        "max_steps": 5000,
        "batch_size": 64,
        "val_fraction": 0.05,
        "log_interval": 50,
        "eval_interval": 500,
        "save_interval": 1000,
    },

    # Optimizer
    "learning_rate": 3e-4,
    "min_lr": 1e-5,
    "weight_decay": 0.1,
    "warmup_steps": 200,
    "grad_clip": 1.0,

    # Mixed precision
    "dtype": "bfloat16",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Apply mode settings
MODE = CONFIG["mode"]
SETTINGS = CONFIG.get(MODE, CONFIG["train"])


def _setting(key, default=None):
    """Get setting: mode-specific override, or top-level CONFIG value."""
    return SETTINGS.get(key, CONFIG.get(key, default))


# ═══════════════════════════════════════════════════════════════
#  LR SCHEDULE
# ═══════════════════════════════════════════════════════════════

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, device, num_batches=None):
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
        batches += 1
        if num_batches and batches >= num_batches:
            break

    model.train()
    return total_loss / max(1, batches)


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, step, loss, vocab_size, config, filename):
    ckpt_dir = ROOT / config["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "vocab_size": vocab_size,
        "model_config": {
            "n_embd": config["n_embd"],
            "n_head": config["n_head"],
            "n_layer": config["n_layer"],
            "block_size": config["block_size"],
            "dropout": config["dropout"],
        },
    }, path)
    return path


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(max_lines=None):
    """
    Args:
        max_lines: if set, only use first N lines from corpus (for quick testing)
    """
    device = CONFIG["device"]
    max_steps = _setting("max_steps")
    batch_size = _setting("batch_size")
    log_interval = _setting("log_interval")
    eval_interval = _setting("eval_interval")
    save_interval = _setting("save_interval")

    # ── Banner ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🎭  PoetryDuelGPT — Phase 1 Training (Lục Bát)")
    print(f"{'='*60}")
    print(f"   Mode:       {MODE.upper()}")
    print(f"   Device:     {device}")
    if device == "cuda":
        print(f"   GPU:        {torch.cuda.get_device_name(0)}")
        print(f"   VRAM:       {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    print(f"   Max steps:  {max_steps:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Block size: {CONFIG['block_size']}")
    print(f"   Tokens/step: {batch_size * CONFIG['block_size']:,}")
    print(f"   Peak LR:    {CONFIG['learning_rate']}")
    print(f"   Warmup:     {CONFIG['warmup_steps']} steps")
    print(f"   Mixed prec: {CONFIG['dtype']}")
    print(f"   Model:      n_embd={CONFIG['n_embd']}, n_head={CONFIG['n_head']}, n_layer={CONFIG['n_layer']}")

    # ── Load tokenizer ──────────────────────────────────────
    tok_path = ROOT / CONFIG["tokenizer_path"]
    print(f"\n📖  Loading tokenizer: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<|pad|>")
    end_id = tokenizer.token_to_id("<|end|>")
    print(f"    Vocab: {vocab_size:,} | Pad={pad_id} | End={end_id}")

    # ── Load & tokenize data ────────────────────────────────
    corpus = ROOT / CONFIG["corpus_path"]
    print(f"\n📦  Loading corpus: {corpus}")
    with open(corpus, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"    Lines: {len(lines):,}")

    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        print(f"    Limited to {max_lines:,} lines for quick test")

    print("    Tokenizing...")
    data = tokenize_corpus(lines, tokenizer)

    # ── DataLoaders ─────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data,
        block_size=CONFIG["block_size"],
        batch_size=batch_size,
        val_fraction=_setting("val_fraction", 0.05),
        num_workers=2 if device == "cuda" else 0,
    )
    batches_per_epoch = len(train_loader)
    print(f"    ~{batches_per_epoch:,} batches/epoch")

    # ── Model ───────────────────────────────────────────────
    print(f"\n🧠  Creating model...")
    model = PoetryDuelGPT(
        vocab_size=vocab_size,
        n_embd=CONFIG["n_embd"],
        n_head=CONFIG["n_head"],
        n_layer=CONFIG["n_layer"],
        block_size=CONFIG["block_size"],
        dropout=CONFIG["dropout"],
    )
    model.to(device)

    # ── Optimizer ───────────────────────────────────────────
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": CONFIG["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG["learning_rate"],
        betas=(0.9, 0.95),
    )

    # ── Training ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀  TRAINING START")
    print(f"{'='*60}\n")

    model.train()
    step = 0
    best_val_loss = float("inf")
    total_train_loss = 0.0
    train_batches = 0
    t0 = time.time()

    train_iter = iter(train_loader)

    while step < max_steps:
        # Refresh iterator if exhausted (new epoch)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Learning rate
        lr = get_lr(step, CONFIG["warmup_steps"], max_steps,
                    CONFIG["learning_rate"], CONFIG["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        step += 1
        total_train_loss += loss.item()
        train_batches += 1

        # ── Logging ─────────────────────────────────────────
        if step % log_interval == 0:
            avg_loss = total_train_loss / train_batches
            elapsed = time.time() - t0
            tokens_per_sec = (step * batch_size * CONFIG["block_size"]) / max(1, elapsed)
            print(f"   Step {step:5d}/{max_steps} | "
                  f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                  f"{tokens_per_sec:.0f} tok/s | "
                  f"{elapsed/60:.1f}min")
            total_train_loss = 0.0
            train_batches = 0

        # ── Evaluation ──────────────────────────────────────
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, num_batches=20)
            trend = "📉" if val_loss < best_val_loss else "➡️"
            best_val_loss = min(best_val_loss, val_loss)
            print(f"   --- Eval  step={step} | val_loss={val_loss:.4f} {trend} | best={best_val_loss:.4f} ---")

        # ── Save checkpoint ─────────────────────────────────
        if step % save_interval == 0:
            path = save_checkpoint(model, optimizer, step, loss.item(),
                                   vocab_size, CONFIG, f"step_{step}.pt")
            print(f"   💾  Saved: {path.name}")

    # ── Final save ──────────────────────────────────────────
    final_path = save_checkpoint(model, optimizer, step, loss.item(),
                                 vocab_size, CONFIG, "final.pt")
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"✅  TRAINING COMPLETE!")
    print(f"   Total time:  {elapsed/60:.1f} min")
    print(f"   Total steps: {step:,}")
    print(f"   Best val:    {best_val_loss:.4f}")
    print(f"   Final model: {final_path}")
    print(f"   To generate: python src/sample.py")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PoetryDuelGPT")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["test", "train"],
                       help="test = 200 steps | train = 5000 steps")
    parser.add_argument("--steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_lines", type=int, default=None, help="Limit corpus lines (for quick testing)")
    args = parser.parse_args()

    if args.mode:
        CONFIG["mode"] = args.mode
        MODE = args.mode
        SETTINGS = CONFIG.get(MODE, CONFIG["train"])

    if args.steps:
        SETTINGS["max_steps"] = args.steps
    if args.batch_size:
        SETTINGS["batch_size"] = args.batch_size
    if args.device:
        CONFIG["device"] = args.device

    train(max_lines=args.max_lines)
