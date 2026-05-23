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
from dataset import PoetryDataset, tokenize_corpus, get_dataloaders, CurriculumDataset

ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",

    # Model
    "n_embd": 512, "n_head": 8, "n_layer": 8, "block_size": 256, "dropout": 0.1,

    # Paths
    "corpus_path": "data/poetry_corpus.txt",
    "tokenizer_path": "tokenizer/poetry_bpe.model",
    "checkpoint_dir": "checkpoints",

    # Modes
    "test":  {"max_steps": 200,  "batch_size": 8,   "eval_interval": 100, "patience": 0},
    "train": {"max_steps": 10000, "batch_size": 128, "eval_interval": 200, "patience": 10},

    # Optimizer
    "learning_rate": 3e-4, "min_lr": 1e-5, "warmup_steps": 500,
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
    prefix = cfg.get("ckpt_prefix", "")
    full_name = f"{prefix}{fname}" if prefix else fname
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step, "loss": loss, "vocab_size": vocab,
        "model_config": {"n_embd": cfg["n_embd"], "n_head": cfg["n_head"],
                         "n_layer": cfg["n_layer"], "block_size": cfg["block_size"],
                         "dropout": cfg["dropout"]},
    }, d / full_name)
    return d / full_name


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(max_lines=None, resume_from=None, curriculum=False, curriculum_rate=0.25):
    cfg, dev = CONFIG, CONFIG["device"]
    mode = cfg["mode"]; S = cfg[mode]
    max_steps, batch_size = S["max_steps"], S["batch_size"]
    eval_interval = S["eval_interval"]
    patience = S.get("patience", 0)

    # ── Banner ──
    is_ft = resume_from is not None
    tag = "FINE-TUNE" if is_ft else mode.upper()
    print(f"\n{'='*60}\n🎭  PoetryDuelGPT — {tag}\n{'='*60}")
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
    if curriculum:
        # Curriculum learning: start with first curriculum_rate of sorted data,
        # progressively expand window during training.
        split = int(len(data) * 0.05)  # val always from last 5%
        train_data, val_data = data[:-split], data[-split:]

        train_ds = CurriculumDataset(train_data, cfg["block_size"], max_fraction=curriculum_rate)
        val_ds = PoetryDataset(val_data, cfg["block_size"])

        print(f"Train: {len(train_ds):,} samples ({curriculum_rate:.0%} of {len(train_data) - cfg['block_size']:,}) | Val: {len(val_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, num_workers=2 if dev == "cuda" else 0,
                                   drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=2 if dev == "cuda" else 0)
        print(f"   📈  Curriculum: starts at {curriculum_rate:.0%} → 100% over training")
    else:
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

    # ── Resume from checkpoint (for Stage 2 fine-tuning) ──
    start_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            resume_path = ROOT / resume_from
        print(f"\n📂  Resuming from: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=dev, weights_only=False)

        # Load model weights (with key remapping for old checkpoints)
        old_state = ckpt["model_state_dict"]
        new_state = {}
        for k, v in old_state.items():
            nk = k.replace("qkv_proj", "qkv").replace("out_proj", "out") \
                  .replace("causal_mask", "mask") \
                  .replace(".ffn.fc1.", ".ffn.net.0.") \
                  .replace(".ffn.fc2.", ".ffn.net.2.")
            nk = {"token_embedding.weight": "tok_emb.weight",
                  "position_embedding.weight": "pos_emb.weight",
                  "ln_final.weight": "ln_f.weight",
                  "ln_final.bias": "ln_f.bias",
                  "lm_head.weight": "head.weight"}.get(nk, nk)
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        old_step = ckpt.get("step", 0)

        # If resuming from a checkpoint trained further than max_steps
        # (e.g. Stage 2 fine-tuning from Stage 1), reset the step counter
        # so the loop actually runs and LR schedule restarts from warmup.
        if old_step >= max_steps:
            print(f"   🔄  Resetting step counter (old={old_step} ≥ max_steps={max_steps})")
            start_step = 0
        else:
            start_step = old_step

        # Lower LR for fine-tuning
        cfg["learning_rate"] = 1e-4
        cfg["warmup_steps"] = 100
        print(f"   Step: {start_step}  |  LR: {cfg['learning_rate']} (fine-tune)")

    # ── Training ──
    print(f"\n{'='*60}\n🚀  TRAINING START\n{'='*60}\n")
    model.train()
    step, best_val, loss_sum, loss_cnt = start_step, float("inf"), 0.0, 0
    loss = torch.tensor(0.0)  # safety net — always bound before save_ckpt
    plateau_count = 0  # evals without improvement
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

        # ── Scheduled Sampling ──
        # Replace a subset of teacher tokens with model's own argmax predictions,
        # forcing it to learn recovery from its mistakes. Decays from 100%→50%
        # teacher over training to reduce the train/inference mismatch.
        use_teacher_prob = max(0.5, 1.0 - step / max_steps * 0.5)
        if use_teacher_prob < 1.0:
            # 1. Get model's own predictions from teacher input (no_grad saves memory)
            with torch.no_grad():
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    logits, _ = model(x, y)
            model_tokens = logits.argmax(dim=-1)  # (B, T) — what model would generate
            del logits  # free ~786 MB (B*T*V in bf16)
            # Shift: prediction at position t becomes input at position t+1
            x_model = torch.cat([x[:, :1], model_tokens[:, :-1]], dim=1)
            # 2. Mix: with prob use_teacher_prob, keep teacher; else use model's own
            mask = torch.rand(x.shape, device=dev) < use_teacher_prob
            x_input = torch.where(mask, x, x_model)
        else:
            x_input = x  # early steps: pure teacher forcing

        # Forward + backward
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            _, loss = model(x_input, y)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        step += 1; loss_sum += loss.item(); loss_cnt += 1
        pbar.update(1); pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ── Eval ──
        if step % eval_interval == 0:
            train_loss = loss_sum / loss_cnt
            val_loss = evaluate(model, val_loader, dev)
            is_best = val_loss < best_val
            trend = "📉" if is_best else "➡️"
            best_val = min(best_val, val_loss)

            # Plateau tracking
            if is_best:
                plateau_count = 0
            else:
                plateau_count += 1

            pbar.close()
            status = f"loss={train_loss:.4f} val={val_loss:.4f} {trend} lr={lr:.2e}"
            if patience > 0:
                status += f"  [{plateau_count}/{patience}]"
            print(f"── Step {step:5d}/{max_steps} ({time.time()-t0:.0f}s) ── {status}")
            loss_sum = 0.0; loss_cnt = 0

            # Save best model
            if is_best:
                p = save_ckpt(model, opt, step, val_loss, V, cfg, "best.pt")
                print(f"   🏆  Best! val={val_loss:.4f} → {p.name}")

            # Early stop
            if patience > 0 and plateau_count >= patience:
                print(f"   ⏹️  Plateau ({patience} evals no improvement) — stopping early")
                break

            if step < max_steps:
                nxt = min(step + eval_interval, max_steps)
                pbar = tqdm(total=eval_interval, desc=f"  Steps {step}-{nxt}", unit="s", leave=False)

        # ── Curriculum expansion ──
        if curriculum and step % (eval_interval * 2) == 0:
            # Gradually grow from curriculum_rate → 1.0 over first 80% of training
            progress = min(1.0, step / (max_steps * 0.8))
            new_fraction = curriculum_rate + (1.0 - curriculum_rate) * progress
            old_n = len(train_ds)
            train_ds.expand(new_fraction)
            if len(train_ds) > old_n:
                print(f"   📈  Curriculum: {old_n:,} → {len(train_ds):,} samples ({new_fraction:.1%})")
                it = iter(train_loader)  # force DataLoader re-sampling with new range

        # ── Save periodic checkpoint ──
        if step % 5000 == 0:
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
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (for fine-tuning)")
    p.add_argument("--corpus", type=str, default=None, help="Override corpus file (e.g. data/corpus_luc_bat.txt)")
    p.add_argument("--name", type=str, default="", help="Prefix for checkpoint files (e.g. stage1_)")
    p.add_argument("--curriculum", action="store_true", help="Progressive difficulty: short→long poems")
    p.add_argument("--curriculum_rate", type=float, default=0.25, help="Starting fraction of curriculum data")
    args = p.parse_args()

    if args.mode:       CONFIG["mode"] = args.mode
    if args.steps:      CONFIG[CONFIG["mode"]]["max_steps"] = args.steps
    if args.batch_size: CONFIG[CONFIG["mode"]]["batch_size"] = args.batch_size
    if args.device:     CONFIG["device"] = args.device
    if args.corpus:     CONFIG["corpus_path"] = args.corpus
    if args.name:       CONFIG["ckpt_prefix"] = args.name

    train(args.max_lines, resume_from=args.resume, curriculum=args.curriculum, 
          curriculum_rate=args.curriculum_rate)
