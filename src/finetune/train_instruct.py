"""
Instruction fine-tuning for Qwen2.5-1.5B-Instruct on Lục Bát poetry.
Custom training loop — clean display, manual checkpointing, no SFTTrainer.

Usage:
  python train_instruct.py
  python train_instruct.py --max-steps 100
  python train_instruct.py --resume checkpoints/instruct_best
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
#  PATH CONFIG
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(ROOT / "checkpoints")))

TRAIN_FILE = DATA_DIR / "instruct_train.jsonl"
VAL_FILE = DATA_DIR / "instruct_val.jsonl"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

TRAIN_CONFIG = {
    "max_steps": int(os.environ.get("TRAIN_MAX_STEPS", "5000")),
    "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", "4")),
    "gradient_accumulation_steps": int(os.environ.get("TRAIN_GRAD_ACCUM", "4")),
    "max_seq_length": int(os.environ.get("TRAIN_MAX_SEQ_LENGTH", "256")),
    "warmup_steps": 100,
    "learning_rate": 2e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "eval_interval": 200,
    "grad_clip": 1.0,
    "patience": 5,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
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


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def collate_batch(examples, tokenizer):
    """Pad a batch of pre-tokenized examples to max length in batch."""
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_len = min(max_len, TRAIN_CONFIG["max_seq_length"])
    
    input_ids = torch.full((len(examples), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    
    for i, ex in enumerate(examples):
        ids = ex["input_ids"][:max_len]
        am = ex["attention_mask"][:max_len]
        lbs = ex["labels"][:max_len]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(am)] = torch.tensor(am, dtype=torch.long)
        labels[i, :len(lbs)] = torch.tensor(lbs, dtype=torch.long)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def save_checkpoint(model, tokenizer, optimizer, step, loss, stage, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "loss": loss, "stage": stage,
    }, path / "training_state.pt")
    return path


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════

def train(resume_from=None, max_steps_override=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    TC = TRAIN_CONFIG
    dev = TC["device"]

    max_steps = max_steps_override or TC["max_steps"]
    batch_size = TC["batch_size"]
    grad_accum = TC["gradient_accumulation_steps"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n🚀  Qwen2.5-1.5B-Instruct — Lục Bát SFT\n{'='*60}")
    print(f"   Steps: {max_steps:,} | Batch: {batch_size}×{grad_accum}={batch_size*grad_accum}")
    print(f"   Seq: {TC['max_seq_length']} | LR: {TC['learning_rate']}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'}")

    # ── Load model + tokenizer ──
    print(f"\n📦  Loading {MODEL_ID}...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    bnb_config = BitsAndBytesConfig(**QLORA_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
        token=hf_token, cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=hf_token, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"   Vocab: {len(tokenizer):,} (native, no new tokens)")

    # ── LoRA ──
    print(f"\n🎯  QLoRA r={LORA_CONFIG['r']} alpha={LORA_CONFIG['lora_alpha']}")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ──
    print(f"\n📖  Loading data...")
    dataset = load_dataset("json", data_files={
        "train": str(TRAIN_FILE), "val": str(VAL_FILE),
    })
    print(f"   Train: {len(dataset['train']):,} | Val: {len(dataset['val']):,}")

    assistant_tmpl = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    
    def tokenize_and_mask(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, truncation=True, max_length=TC["max_seq_length"])
        ids = tokens["input_ids"]
        labels = ids.copy()
        at_len = len(assistant_tmpl)
        for j in range(len(ids) - at_len + 1):
            if ids[j:j+at_len] == assistant_tmpl:
                labels[:j+at_len] = [-100] * (j + at_len)
                break
        else:
            labels[:] = [-100] * len(labels)
        return {"input_ids": ids, "attention_mask": tokens["attention_mask"], "labels": labels}

    dataset = dataset.map(tokenize_and_mask, desc="Tokenizing", remove_columns=["messages"])
    dataset.set_format(type="python")

    # ── DataLoaders ──
    train_ds = dataset["train"]
    val_ds = dataset["val"]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer),
        num_workers=TC["num_workers"], pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer),
        num_workers=TC["num_workers"], pin_memory=True,
    )
    print(f"   Batches: {len(train_loader):,} train | {len(val_loader):,} val")

    # ── Optimizer ──
    opt = torch.optim.AdamW(model.parameters(), lr=TC["learning_rate"],
                             betas=(0.9, 0.95), weight_decay=TC["weight_decay"])

    # ── Resume ──
    start_step = 0
    if resume_from:
        from peft import PeftModel
        resume_path = Path(resume_from)
        if not resume_path.exists():
            resume_path = CHECKPOINT_DIR / resume_from
        print(f"\n📂  Resume: {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path)
        ckpt = torch.load(str(resume_path / "training_state.pt"), map_location=dev, weights_only=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        max_steps = max(max_steps, start_step + 1000)
        print(f"   Step: {start_step} | Max: {max_steps}")

    # ── Training ──
    print(f"\n{'='*60}\n🔥  TRAINING\n{'='*60}\n")
    model.train()
    step = start_step
    best_val = float("inf")
    plateau = 0
    loss_sum = 0.0
    loss_cnt = 0
    micro_step = 0
    t0 = time.time()
    it = iter(train_loader)

    pbar = tqdm(total=TC["eval_interval"], desc=f"  Steps 0-{TC['eval_interval']}", 
                unit="s", leave=False)

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x = batch["input_ids"].to(dev)
        y = batch["labels"].to(dev)
        attention = batch["attention_mask"].to(dev)

        lr = get_lr(step, TC["warmup_steps"], max_steps, TC["learning_rate"], TC["min_lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            out = model(input_ids=x, attention_mask=attention, labels=y)
            loss = out.loss / grad_accum

        loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC["grad_clip"])
            opt.step()
            opt.zero_grad()
            step += 1
            loss_sum += loss.item() * grad_accum
            loss_cnt += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item()*grad_accum:.4f}"})

        # ── Eval ──
        if step > 0 and step % TC["eval_interval"] == 0 and micro_step % grad_accum == 0:
            train_loss = loss_sum / loss_cnt
            val_loss = evaluate(model, val_loader, dev)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            plateau = 0 if is_best else plateau + 1

            pbar.close()
            trend = "📉 BEST" if is_best else "➡️"
            elapsed = time.time() - t0
            eta = (elapsed / step) * (max_steps - step) if step > 0 else 0
            print(f"── Step {step:5d}/{max_steps} | loss={train_loss:.4f} val={val_loss:.4f} "
                  f"{trend} | LR {lr:.2e} | ETA {eta/60:.0f}m")
            loss_sum = 0.0
            loss_cnt = 0

            if is_best:
                p = save_checkpoint(model, tokenizer, opt, step, val_loss, "instruct",
                                    CHECKPOINT_DIR / "instruct_best")
                print(f"   💾  {p.name}")

            if TC["patience"] > 0 and plateau >= TC["patience"]:
                print(f"   ⏹️  Plateau ({plateau} evals) — stopping")
                break

            if step < max_steps:
                nxt = min(step + TC["eval_interval"], max_steps)
                pbar = tqdm(total=TC["eval_interval"], desc=f"  Steps {step}-{nxt}",
                            unit="s", leave=False)

    pbar.close()

    # ── Save final ──
    final = save_checkpoint(model, tokenizer, opt, step, best_val, "instruct",
                            CHECKPOINT_DIR / "instruct_final")
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅  Done! {elapsed:.0f}s | Best val: {best_val:.4f} | {final}")
    print(f"{'='*60}")

    return str(final)


@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        if count >= n_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        attention = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(input_ids=x, attention_mask=attention, labels=y)
        total_loss += out.loss.item()
        count += 1
    model.train()
    return total_loss / count if count > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    TRAIN_CONFIG["device"] = args.device if torch.cuda.is_available() else "cpu"
    max_steps = args.max_steps or int(os.environ.get("TRAIN_MAX_STEPS", "0")) or None
    train(resume_from=args.resume, max_steps_override=max_steps)
