"""
Training loop for PoetryDuelGPT — mixed precision, cosine LR, checkpointing.

Modes:
  test  → 200 steps, batch=8   (smoke test)
  train → 5000 steps, batch=192 (full training)
"""

import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from model import PoetryDuelGPT
from dataset import PoetryDataset, tokenize_corpus, get_dataloaders

ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",

    # Model
    "n_embd": 576, "n_head": 6, "n_layer": 8, "block_size": 256, "dropout": 0.1,

    # Paths
    "corpus_path": "data/poetry_corpus.txt",
    "tokenizer_path": "tokenizer/poetry_bpe.model",
    "checkpoint_dir": "checkpoints",

    # Modes
    "test":  {"max_steps": 200,  "batch_size": 8,   "eval_interval": 100},
    "train": {"max_steps": 15000, "batch_size": 192, "eval_interval": 500},

    # Optimizer
    "learning_rate": 3e-4, "min_lr": 1e-5, "warmup_steps": 200,
    "weight_decay": 0.1, "grad_clip": 1.0,

    # Mixed precision
    "dtype": "bfloat16",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════════════════════
#  LR SCHEDULE
# ═══════════════════════════════════════════════════════════════

def get_lr(step, warmup, total, max_lr, min_lr):
    """Cosine with linear warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    loss, cnt = 0.0, 0
    for x, y in loader:
        if cnt >= n_batches: break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, l = model(x, y)
        loss += l.item(); cnt += 1
    model.train()
    return loss / cnt


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def save_ckpt(model, opt, step, loss, vocab, cfg, fname):
    d = ROOT / cfg["checkpoint_dir"]; d.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step, "loss": loss, "vocab_size": vocab,
        "model_config": {"n_embd": cfg["n_embd"], "n_head": cfg["n_head"],
                         "n_layer": cfg["n_layer"], "block_size": cfg["block_size"],
                         "dropout": cfg["dropout"]},
    }, d / fname)
    return d / fname


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(max_lines=None):
    cfg, dev = CONFIG, CONFIG["device"]
    mode = cfg["mode"]; S = cfg[mode]
    max_steps, batch_size = S["max_steps"], S["batch_size"]
    eval_interval = S["eval_interval"]

    # ── Banner ──
    print(f"\n{'='*60}\n🎭  PoetryDuelGPT — {mode.upper()} (Lục Bát)\n{'='*60}")
    print(f"   Device: {dev}  |  Steps: {max_steps:,}  |  Batch: {batch_size}  |  "
          f"Tok/step: {batch_size*cfg['block_size']:,}")
    print(f"   Peak LR: {cfg['learning_rate']}  |  Warmup: {cfg['warmup_steps']}  |  "
          f"Model: emb={cfg['n_embd']} head={cfg['n_head']} layer={cfg['n_layer']}")

    # ── Load tokenizer + data ──
    tok_path, corpus = ROOT / cfg["tokenizer_path"], ROOT / cfg["corpus_path"]
    # Check files exist
    for p, name in [(tok_path, "Tokenizer"), (corpus, "Corpus")]:
        if not p.exists():
            print(f"\n❌  {name} not found: {p}")
            print("   Run: python src/preprocess.py && python src/train_bpe.py")
            return

    print(f"\n📖  Tokenizer: {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))
    V = tok.get_vocab_size()
    print(f"    Vocab: {V:,}  |  Pad=0  |  End=3")

    print(f"\n📦  Corpus: {corpus}")
    with open(corpus, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_lines: lines = lines[:max_lines]
    print(f"    Lines: {len(lines):,}")
    data = tokenize_corpus(lines, tok)

    # ── DataLoaders ──
    train_loader, val_loader = get_dataloaders(data, cfg["block_size"], batch_size,
                                                val_fraction=0.05,
                                                num_workers=2 if dev == "cuda" else 0)

    # ── Model ──
    model = PoetryDuelGPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"],
                          cfg["block_size"], cfg["dropout"]).to(dev)

    # ── Optimizer (weight decay on weights, not biases/norms) ──
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg["learning_rate"], betas=(0.9, 0.95))

    # ── Training ──
    print(f"\n{'='*60}\n🚀  TRAINING START\n{'='*60}\n")
    model.train()
    step, best_val, loss_sum, loss_cnt = 0, float("inf"), 0.0, 0
    t0 = time.time()
    it = iter(train_loader)
    pbar = tqdm(total=eval_interval, desc=f"  Steps 0-{eval_interval}", unit="s", leave=False)

    while step < max_steps:
        # Next batch (new epoch if exhausted)
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(dev), y.to(dev)

        # LR schedule
        lr = get_lr(step, cfg["warmup_steps"], max_steps, cfg["learning_rate"], cfg["min_lr"])
        for pg in opt.param_groups: pg["lr"] = lr

        # Forward + backward
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        step += 1; loss_sum += loss.item(); loss_cnt += 1
        pbar.update(1); pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ── Eval (summary line, like diffusion epoch display) ──
        if step % eval_interval == 0:
            train_loss = loss_sum / loss_cnt
            val_loss = evaluate(model, val_loader, dev)
            trend = "📉" if val_loss < best_val else "➡️"
            best_val = min(best_val, val_loss)
            pbar.close()
            print(f"── Step {step:5d}/{max_steps} ({time.time()-t0:.0f}s) ── "
                  f"loss={train_loss:.4f} val={val_loss:.4f} {trend} lr={lr:.2e}")
            loss_sum = 0.0; loss_cnt = 0

            if step < max_steps:
                nxt = min(step + eval_interval, max_steps)
                pbar = tqdm(total=eval_interval, desc=f"  Steps {step}-{nxt}", unit="s", leave=False)

        # ── Save checkpoint ──
        if step % 1000 == 0:
            p = save_ckpt(model, opt, step, loss.item(), V, cfg, f"step_{step}.pt")
            print(f"   💾  {p.name}")

    pbar.close()
    final = save_ckpt(model, opt, step, loss.item(), V, cfg, "final.pt")
    print(f"\n{'='*60}\n✅  Done! {time.time()-t0:.0f}s | Best val: {best_val:.4f} | {final}\n{'='*60}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "train"], default="train")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_lines", type=int, default=None)
    args = p.parse_args()

    if args.mode:       CONFIG["mode"] = args.mode
    if args.steps:      CONFIG[CONFIG["mode"]]["max_steps"] = args.steps
    if args.batch_size: CONFIG[CONFIG["mode"]]["batch_size"] = args.batch_size
    if args.device:     CONFIG["device"] = args.device

    train(args.max_lines)
