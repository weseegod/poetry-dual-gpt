"""
QLoRA fine-tune Qwen2.5-1.5B on Vietnamese poetry — containerized for Salad.

Designed to run headless inside a Docker container. At completion, uploads
checkpoints to GCS/S3, then exits (container stops, billing stops).

Usage:
  python train.py --stage 1
  python train.py --stage 2 --resume /tmp/checkpoints/qwen_stage1_best
"""

import argparse
import math
import os
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
#  PATH CONFIG (all paths relative to /app inside container)
# ═══════════════════════════════════════════════════════════════

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", str(APP_DIR / "corpus")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(APP_DIR / "checkpoints")))

STAGE_CORPUS = {
    1: CORPUS_DIR / "poetry_corpus.txt",
    2: CORPUS_DIR / "corpus_luc_bat.txt",
}

MODEL_ID = "Qwen/Qwen2.5-1.5B"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {"max_steps": 5000, "lr": 2e-4, "patience": 5},
    2: {"max_steps": 2000, "lr": 1e-4, "patience": 3},
}

LORA_CONFIG = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "modules_to_save": ["embed_tokens", "lm_head"],
    "task_type": "CAUSAL_LM",
}

QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

TRAIN_CONFIG = {
    "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", "4")),
    "gradient_accumulation_steps": int(os.environ.get("TRAIN_GRAD_ACCUM", "8")),
    "block_size": int(os.environ.get("TRAIN_BLOCK_SIZE", "256")),
    "warmup_steps": 100,
    "min_lr": 1e-6,
    "eval_interval": 200,
    "grad_clip": 1.0,
    "num_workers": int(os.environ.get("TRAIN_NUM_WORKERS", "4")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════════════════════
#  SPECIAL TOKENS (same 335 as the original train_bpe.py)
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
    ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        if line.strip():
            ids.extend(tokenizer.encode(line, add_special_tokens=False))
    t = torch.tensor(ids, dtype=torch.long)
    print(f"  {len(lines):,} lines → {len(t):,} tokens")
    return t


def get_dataloaders(data, block_size, batch_size, num_workers=4, val_frac=0.05):
    split = int(len(data) * val_frac)
    train_data, val_data = data[:-split], data[-split:]
    train_ds = FlatTokenDataset(train_data, block_size)
    val_ds = FlatTokenDataset(val_data, block_size)
    print(f"  Train: {len(train_ds):,} samples | Val: {len(val_ds):,} | Batch: {batch_size}")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
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
    print(f"💾  Saved checkpoint to {path}")
    return path


# ═══════════════════════════════════════════════════════════════
#  CLOUD UPLOAD
# ═══════════════════════════════════════════════════════════════

def upload_checkpoints_gcs(local_dir, bucket_uri, stage):
    """Upload checkpoint directory to Google Cloud Storage."""
    from google.cloud import storage
    client = storage.Client()
    bucket_name = bucket_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(bucket_uri.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)

    local_dir = Path(local_dir)
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            blob_path = f"{prefix}/stage{stage}/{fpath.relative_to(local_dir)}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(fpath))
            print(f"  ☁️  Uploaded: {blob_path}")

    print(f"✅  Stage {stage} uploaded to gs://{bucket_name}/{prefix}/")
    return f"gs://{bucket_name}/{prefix}/stage{stage}"


def upload_checkpoints_s3(local_dir, s3_uri, stage, endpoint_url=None):
    """Upload checkpoint directory to S3-compatible storage (AWS S3 or Cloudflare R2)."""
    import boto3
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
        print(f"   Using endpoint: {endpoint_url}")
    elif os.environ.get("AWS_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

    s3 = boto3.client("s3", **kwargs)
    bucket_name = s3_uri.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_uri.replace("s3://", "").split("/")[1:])

    local_dir = Path(local_dir)
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            key = f"{prefix}/stage{stage}/{fpath.relative_to(local_dir)}"
            s3.upload_file(str(fpath), bucket_name, key)
            print(f"  ☁️  Uploaded: s3://{bucket_name}/{key}")

    print(f"✅  Stage {stage} uploaded to s3://{bucket_name}/{prefix}/")
    return f"s3://{bucket_name}/{prefix}/stage{stage}"


def upload_checkpoints(local_dir, stage):
    """Upload to cloud storage — supports GCS, AWS S3, and Cloudflare R2."""
    gcs_bucket = os.environ.get("GCS_BUCKET")
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_endpoint = os.environ.get("S3_ENDPOINT")  # For R2: https://<id>.r2.cloudflarestorage.com

    if gcs_bucket:
        return upload_checkpoints_gcs(local_dir, gcs_bucket, stage)
    elif s3_bucket:
        return upload_checkpoints_s3(local_dir, s3_bucket, stage, endpoint_url=s3_endpoint)
    else:
        print("⚠️  No GCS_BUCKET or S3_BUCKET set — checkpoint saved locally only")
        print("   Set one of: GCS_BUCKET=gs://...  or  S3_BUCKET=s3://... + S3_ENDPOINT=https://...")
        return str(local_dir)


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(stage=1, resume_from=None, max_steps_override=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    SC = STAGE_CONFIGS[stage]
    TC = TRAIN_CONFIG
    dev = TC["device"]

    max_steps = max_steps_override if max_steps_override else SC["max_steps"]
    batch_size = TC["batch_size"]
    grad_accum = TC["gradient_accumulation_steps"]
    corpus_path = STAGE_CORPUS[stage]

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. "
                                f"Did you COPY it into the Docker image?")

    print(f"\n{'='*60}\n🚀  Qwen2.5-1.5B QLoRA — Stage {stage}\n{'='*60}")
    print(f"   Steps: {max_steps:,} | Batch: {batch_size}×{grad_accum}={batch_size*grad_accum} | "
          f"Block: {TC['block_size']} | LR: {SC['lr']}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'}")
    if dev == 'cuda':
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model + tokenizer ──
    print(f"\n📦  Loading {MODEL_ID}...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set! Set it to your HuggingFace token.")
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"   Cache: {cache_dir}")
    bnb_config = BitsAndBytesConfig(**QLORA_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Add 335 special tokens ──
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

    data = tokenize_with_qwen(lines, tokenizer)
    train_loader, val_loader = get_dataloaders(data, TC["block_size"], batch_size, num_workers=TC["num_workers"])

    # ── Optimizer ──
    opt = torch.optim.AdamW(model.parameters(), lr=SC["lr"], betas=(0.9, 0.95),
                             weight_decay=0.01)

    # ── Resume ──
    start_step = 0
    if resume_from:
        from peft import PeftModel
        resume_path = Path(resume_from)
        if not resume_path.exists():
            resume_path = CHECKPOINT_DIR / resume_from
        print(f"\n📂  Resuming from: {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path)
        ckpt = torch.load(str(resume_path / "training_state.pt"), map_location=dev, weights_only=False)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
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
    micro_step = 0

    while step < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(dev), y.to(dev)

        lr = get_lr(step, TC["warmup_steps"], max_steps, SC["lr"], TC["min_lr"])
        for pg in opt.param_groups: pg["lr"] = lr

        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            out = model(x, labels=y)
            loss_val = out.loss / grad_accum

        loss_val.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC["grad_clip"])
            opt.step()
            opt.zero_grad()
            step += 1
            loss_sum += loss_val.item() * grad_accum
            loss_cnt += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_val.item()*grad_accum:.4f}"})

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
            elapsed = time.time() - t0
            eta = (elapsed / max(1, step)) * (max_steps - step)
            print(f"── Step {step:5d}/{max_steps} ({elapsed:.0f}s, ETA: {eta:.0f}s) ── {status}")
            loss_sum = 0.0; loss_cnt = 0

            if is_best:
                p = save_checkpoint(model, tokenizer, opt, step, val_loss, stage,
                                    CHECKPOINT_DIR / f"qwen_stage{stage}_best")
                print(f"   🏆  Best! val={val_loss:.4f} → {p}")

            if patience > 0 and plateau_count >= patience:
                print(f"   ⏹️  Plateau ({patience} evals) — stopping")
                break

            if step < max_steps:
                nxt = min(step + TC["eval_interval"], max_steps)
                pbar = tqdm(total=TC["eval_interval"], desc=f"  Steps {step}-{nxt}", unit="s", leave=False)

        if step > 0 and step % 2000 == 0:
            save_checkpoint(model, tokenizer, opt, step, float(loss_val) * grad_accum, stage,
                            CHECKPOINT_DIR / f"qwen_stage{stage}_step{step}")

    pbar.close()

    # ── Final save + cloud upload ──
    final_path = save_checkpoint(model, tokenizer, opt, step, best_val, stage,
                                  CHECKPOINT_DIR / f"qwen_stage{stage}_final")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅  Stage {stage} complete! {elapsed:.0f}s | Best val: {best_val:.4f}")
    print(f"{'='*60}\n")

    # ── Upload checkpoints to cloud storage ──
    print("☁️  Uploading checkpoints...")
    cloud_path = upload_checkpoints(CHECKPOINT_DIR, stage)
    print(f"📍  Cloud path: {cloud_path}")

    return str(final_path)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5 QLoRA — containerized for Salad")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps (for smoke testing)")
    args = parser.parse_args()

    TRAIN_CONFIG["device"] = args.device if torch.cuda.is_available() else "cpu"

    # Env var override for max steps (works well with Docker -e flag)
    max_steps = args.max_steps or int(os.environ.get("TRAIN_MAX_STEPS", "0")) or None

    # Stage 1
    result = train(stage=args.stage, resume_from=args.resume, max_steps_override=max_steps)

    # Auto-chain: if stage 1 completed successfully and no explicit resume,
    # run stage 2 automatically (saves starting a second container)
    if args.stage == 1 and not args.resume:
        auto_chain = os.environ.get("AUTO_CHAIN_STAGES", "0")
        if auto_chain == "1":
            stage1_best = CHECKPOINT_DIR / "qwen_stage1_best"
            if stage1_best.exists():
                print("\n" + "="*60)
                print("🔗  AUTO-CHAIN: Stage 1 → Stage 2")
                print("="*60 + "\n")
                train(stage=2, resume_from=str(stage1_best), max_steps_override=max_steps)
