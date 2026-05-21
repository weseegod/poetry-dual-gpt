"""
QLoRA fine-tuning of Qwen2.5-1.5B on Vietnamese poetry with control tokens.

Same training format as PoetryDuelGPT:
  <|start|> [LUC_BAT] [RHYME:ong] [TONE:BBBTTB] prompt <|reply|> response <|end|>

Uses Qwen's native tokenizer with our 335 special tokens added.
Two-stage: Stage 1 (all genres, 5K steps) → Stage 2 (Lục Bát, 2K steps).

Usage:
  python finetune/finetune.py              # Stage 1
  python finetune/finetune.py --stage 2    # Stage 2 from Stage 1 checkpoint
"""

import argparse, math, re, time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).parent.parent

# ── Third-party imports (available on Colab, optional locally) ──
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_HF = True
except ImportError:
    HAS_HF = False

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

MODEL_ID = "Qwen/Qwen2.5-1.5B"

STAGE_CONFIGS = {
    1: {"max_steps": 5000,  "corpus": "data/poetry_corpus.txt",              "lr": 2e-4, "patience": 5},
    2: {"max_steps": 2000,  "corpus": "data/corpus_luc_bat.txt",             "lr": 1e-4, "patience": 3},
}

LORA_CONFIG = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}

QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

TRAIN_CONFIG = {
    "batch_size": 4,         # per-device, with gradient accumulation
    "gradient_accumulation_steps": 8,  # effective batch = 4 × 8 = 32
    "block_size": 256,
    "warmup_steps": 100,
    "min_lr": 1e-6,
    "eval_interval": 200,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════════════════════
#  SPECIAL TOKENS (same 335 as train_bpe.py)
# ═══════════════════════════════════════════════════════════════

def collect_rhyme_groups(corpus_path):
    rhymes = set()
    with open(corpus_path) as f:
        for line in f:
            for m in re.finditer(r'\[RHYME:(\w+)\]', line):
                r = m.group(1)
                if all(c.isalpha() for c in r) and len(r) >= 1:
                    rhymes.add(r)
    return sorted(rhymes)

def collect_tone_patterns(corpus_path):
    tones = set()
    with open(corpus_path) as f:
        for line in f:
            for m in re.finditer(r'\[TONE:([BT]+)\]', line):
                tones.add(m.group(1))
    return sorted(tones)

def collect_doi_am_patterns(corpus_path):
    TRAC = set('áắấéếíóốớúứýạặậẹệịọộợụựỵảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹ'
               'ÁẮẤÉẾÍÓỐỚÚỨÝẠẶẬẸỆỊỌỘỢỤỰỴẢẲẨẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸ')
    def get_tone(syl):
        for ch in syl:
            if ch in TRAC: return 'T'
        return 'B'
    patterns = set()
    with open(corpus_path) as f:
        for line in f:
            if '[THAT_NGON]' in line and '<|reply|>' in line:
                parts = line.split('<|reply|>')
                if len(parts) >= 2:
                    resp = parts[1].split('<|end|>')[0].strip()
                    tones = ''.join(get_tone(s) for s in resp.split()[:7])
                    if len(tones) == 7:
                        patterns.add(tones)
    return sorted(patterns)

def build_special_tokens(corpus_path):
    core = ["<|pad|>", "<|start|>", "<|reply|>", "<|end|>",
            "[LUC_BAT]", "[TU_TUYET]", "[THAT_NGON_BAT_CU]", "[THAT_NGON]"]
    rhyme = [f"[RHYME:{r}]" for r in collect_rhyme_groups(corpus_path)]
    tone = [f"[TONE:{t}]" for t in collect_tone_patterns(corpus_path)]
    doi_am = [f"[DOIAM:{d}]" for d in collect_doi_am_patterns(corpus_path)]
    link = ["[LINK2:B]", "[LINK2:T]"]
    all_tokens = core + rhyme + tone + doi_am + link
    print(f"Special tokens: Core={len(core)} Rhyme={len(rhyme)} "
          f"Tone={len(tone)} ĐốiÂm={len(doi_am)} Link2={len(link)} → {len(all_tokens)} total")
    return all_tokens


# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════

class FlatTokenDataset(torch.utils.data.Dataset):
    """Flat token tensor → random windows (same as PoetryDataset)."""
    def __init__(self, data, block_size):
        assert len(data) > block_size
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:self.block_size], chunk[1:]


def tokenize_with_qwen(lines, tokenizer):
    """Encode text lines with Qwen tokenizer → flat LongTensor."""
    ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        if line.strip():
            ids.extend(tokenizer.encode(line, add_special_tokens=False))
    t = torch.tensor(ids, dtype=torch.long)
    print(f"  {len(lines):,} lines → {len(t):,} tokens")
    return t


def get_dataloaders(data, block_size, batch_size, val_frac=0.05):
    split = int(len(data) * val_frac)
    train_data, val_data = data[:-split], data[-split:]

    train_ds = FlatTokenDataset(train_data, block_size)
    val_ds = FlatTokenDataset(val_data, block_size)

    print(f"  Train: {len(train_ds):,} samples | Val: {len(val_ds):,} | Batch: {batch_size}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    loss, cnt = 0.0, 0
    for x, y in loader:
        if cnt >= n_batches: break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(x, labels=y)
        loss += out.loss.item(); cnt += 1
    model.train()
    return loss / cnt


def save_checkpoint(model, tokenizer, optimizer, step, loss, stage, path):
    """Save LoRA adapter weights + tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "stage": stage,
    }, path / "training_state.pt")
    return path


def train(stage=1, resume_from=None, max_lines=None):
    assert HAS_HF, "Install: pip install transformers peft bitsandbytes accelerate"

    SC = STAGE_CONFIGS[stage]
    TC = TRAIN_CONFIG
    dev = TC["device"]

    max_steps = SC["max_steps"]
    batch_size = TC["batch_size"]
    block_size = TC["block_size"]
    grad_accum = TC["gradient_accumulation_steps"]
    corpus_path = ROOT / SC["corpus"]

    print(f"\n{'='*60}\n🚀  Qwen2.5-1.5B QLoRA — Stage {stage}\n{'='*60}")
    print(f"   Steps: {max_steps:,} | Batch: {batch_size}×{grad_accum}={batch_size*grad_accum} | "
          f"Block: {block_size} | LR: {SC['lr']}")

    # ── Load model + tokenizer ──
    print(f"\n📦  Loading {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(**QLORA_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Add our 335 special tokens ──
    print(f"\n🔤  Adding special tokens...")
    special_tokens = build_special_tokens(str(corpus_path))
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Added {num_added} tokens | Vocab: {len(tokenizer):,}")

    # ── Apply LoRA ──
    print(f"\n🎯  Applying LoRA (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']})...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load data ──
    print(f"\n📖  Corpus: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_lines: lines = lines[:max_lines]

    # Pre-tokenize with Qwen tokenizer
    data = tokenize_with_qwen(lines, tokenizer)
    train_loader, val_loader = get_dataloaders(data, block_size, batch_size)

    # ── Optimizer ──
    opt = torch.optim.AdamW(model.parameters(), lr=SC["lr"], betas=(0.9, 0.95),
                             weight_decay=0.01)

    # ── Resume ──
    start_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        print(f"\n📂  Resuming from: {resume_path}")
        ckpt = torch.load(str(resume_path / "training_state.pt"), map_location=dev)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        # Resume LR schedule from checkpoint
        SC["max_steps"] = max(max_steps, start_step + 1000)
        max_steps = SC["max_steps"]
        print(f"   Step: {start_step} | Adjusted max_steps: {max_steps}")

    # ── Training ──
    print(f"\n{'='*60}\n🔥  TRAINING\n{'='*60}\n")
    model.train()
    step, best_val, plateau_count = start_step, float("inf"), 0
    loss_sum, loss_cnt = 0.0, 0
    t0 = time.time()
    it = iter(train_loader)
    pbar = tqdm(total=TC["eval_interval"], desc=f"  Steps 0-{TC['eval_interval']}", unit="s", leave=False)

    # Gradient accumulation counter
    micro_step = 0

    while step < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(dev), y.to(dev)

        # LR schedule
        lr = get_lr(step, TC["warmup_steps"], max_steps, SC["lr"], TC["min_lr"])
        for pg in opt.param_groups: pg["lr"] = lr

        # Forward
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            out = model(x, labels=y)
            loss_val = out.loss / grad_accum

        loss_val.backward()
        micro_step += 1

        # Gradient accumulation
        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC["grad_clip"])
            opt.step()
            opt.zero_grad()
            step += 1
            loss_sum += loss_val.item() * grad_accum
            loss_cnt += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_val.item()*grad_accum:.4f}"})

        # ── Eval ──
        if step > 0 and step % TC["eval_interval"] == 0 and micro_step % grad_accum == 0:
            train_loss = loss_sum / loss_cnt
            val_loss = evaluate(model, val_loader, dev)
            is_best = val_loss < best_val
            trend = "📉" if is_best else "➡️"
            best_val = min(best_val, val_loss)

            plateau_count = 0 if is_best else plateau_count + 1

            pbar.close()
            status = f"loss={train_loss:.4f} val={val_loss:.4f} {trend} lr={lr:.2e}"
            patience = SC.get("patience", 0)
            if patience > 0:
                status += f"  [{plateau_count}/{patience}]"
            print(f"── Step {step:5d}/{max_steps} ({time.time()-t0:.0f}s) ── {status}")
            loss_sum = 0.0; loss_cnt = 0

            if is_best:
                p = save_checkpoint(model, tokenizer, opt, step, val_loss, stage,
                                    ROOT / f"checkpoints/qwen_stage{stage}_best")
                print(f"   🏆  Best! val={val_loss:.4f} → {p}")

            if patience > 0 and plateau_count >= patience:
                print(f"   ⏹️  Plateau ({patience} evals) — stopping")
                break

            if step < max_steps:
                nxt = min(step + TC["eval_interval"], max_steps)
                pbar = tqdm(total=TC["eval_interval"], desc=f"  Steps {step}-{nxt}", unit="s", leave=False)

        # ── Periodic save ──
        if step > 0 and step % 2000 == 0:
            p = save_checkpoint(model, tokenizer, opt, step, float(loss_val) * grad_accum, stage,
                                ROOT / f"checkpoints/qwen_stage{stage}_step{step}")
            print(f"   💾  {p}")

    pbar.close()
    final_path = save_checkpoint(model, tokenizer, opt, step, best_val, stage,
                                  ROOT / f"checkpoints/qwen_stage{stage}_final")
    print(f"\n{'='*60}\n✅  Done! {time.time()-t0:.0f}s | Best val: {best_val:.4f} | {final_path}\n{'='*60}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Qwen2.5-1.5B QLoRA fine-tuning")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--resume", type=str, default=None, help="Resume from LoRA checkpoint dir")
    p.add_argument("--max_lines", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--corpus", type=str, default=None, help="Override corpus path")
    args = p.parse_args()

    TRAIN_CONFIG["device"] = args.device if torch.cuda.is_available() else "cpu"
    if args.corpus:
        STAGE_CONFIGS[args.stage]["corpus"] = args.corpus
    train(stage=args.stage, resume_from=args.resume, max_lines=args.max_lines)
