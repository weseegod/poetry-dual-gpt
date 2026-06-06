# 🚀 Step-by-Step: Train Qwen2.5-1.5B QLoRA on Salad

This guide takes you from zero to a trained Qwen QLoRA model on Salad GPU cloud.

---

## ⚡ Quick Overview

```
Your Laptop                    Cloud
─────────────                  ─────
1. Set up accounts       →    (one time)
2. Prepare corpus         →    local
3. Build Docker image     →    local
4. Push to GHCR           →    GitHub Container Registry
5. Deploy                 →    Salad RTX 3090 ($0.09/hr)
6. Monitor logs           →    Salad
7. Download checkpoint    →    Cloudflare R2 → your laptop
```

**Time:** ~3.5 hours total (2.5h training + 1h setup)  
**Cost:** ~$0.32 for full training (Stage 1 + Stage 2)

---

## Prerequisites

- [ ] Docker installed
- [ ] Python 3.10+ with `pip`
- [ ] Git
- [ ] GitHub account

---

## Step 1: Set Up Accounts (one time, ~30 min)

### 1a. HuggingFace

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **Read** token → copy it → this is `HF_TOKEN`
3. Go to [Qwen2.5-1.5B model page](https://huggingface.co/Qwen/Qwen2.5-1.5B)
4. Click **"Agree and access repository"** (required to download the model)

### 1b. Salad

1. Go to [portal.salad.com](https://portal.salad.com)
2. Create account, add $5 deposit minimum
3. Get your:
   - **API Key** (Settings → API Keys)
   - **Organization name** (from URL: `.../organizations/YOUR_ORG/...`)
   - **Project name** (from URL or create one in dashboard)

### 1c. Cloudflare R2 (free tier: 10GB)

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **R2**
2. Click **Create bucket** → name: `poetry-checkpoints` → Create
3. Go to **Manage R2 API Tokens** → **Create API Token**
   - Permission: **Object Read & Write**
   - Select your bucket → Create
   - Copy: **Access Key ID** and **Secret Access Key**
4. Go to **R2 Overview** → copy your **Account ID**
5. Your endpoint is: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

### 1d. GitHub Container Registry

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a **Classic token** with `write:packages` scope
3. Copy the token → this is your GHCR password

---

## Step 2: Configure Environment

```bash
cd /Users/thanhbm/Projects/poetry-dual-gpt

# Create .env from template
cp src/finetune/.env.example src/finetune/.env
```

Edit `src/finetune/.env` with your credentials:

```ini
# Salad
SALAD_API_KEY=salad_xxx
SALAD_ORG=your-org-name
SALAD_PROJECT=your-project-name
SALAD_IMAGE=ghcr.io/YOUR_GITHUB_USER/poetry-trainer-v5:latest

# HuggingFace
HF_TOKEN=hf_xxx

# Cloudflare R2
S3_BUCKET=s3://poetry-checkpoints/training-runs
S3_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
```

> Replace `YOUR_GITHUB_USER` with your actual GitHub username.

---

## Step 3: Prepare Corpus

Generate the training corpus from the authoritative dataset (`data/poems_dataset_clean.csv`, 125K poems).

```bash
cd /Users/thanhbm/Projects/poetry-dual-gpt

# Generate combined corpus (LB + TN) with full control tokens
./venv/bin/python src/preprocess.py \
    --csv data/poems_dataset_clean.csv \
    --output data/poetry_corpus.txt

# Create LB-only corpus for Stage 2 fine-tuning
grep '\[LUC_BAT\]' data/poetry_corpus.txt > data/corpus_luc_bat.txt

# Verify
wc -l data/poetry_corpus.txt data/corpus_luc_bat.txt
# Expected: ~869K combined, ~618K LB-only

# Spot-check control tokens
grep -c '\[RHYME:' data/corpus_luc_bat.txt   # → 618089
grep -c '\[TONE:' data/corpus_luc_bat.txt    # → 618089
grep -c '\[TRAMBONG:' data/corpus_luc_bat.txt # → 608288
```

> The corpus includes `[RHYME:*]`, `[TONE:*]`, and `[TRAMBONG:*]` control tokens.
> `[TRAMBONG:*]` tokens will be tokenized as plain text by Qwen (subword tokens),
> which is acceptable — the model learns to condition on them via attention.

---

## Step 4: Build, Test, and Push (CI-Powered)

### CI Pipeline (automated)

Push to `main` and GitHub Actions handles everything:

```
git push origin main
  │
  ▼
.github/workflows/ci.yml
  1. Build Docker image (CPU)
  2. Smoke test: verify corpus integrity, Python imports, CLI args
  3. Push image to GHCR (ghcr.io/USER/poetry-dual-gpt/poetry-trainer-v5:latest)
```

**Prerequisites**: None. GitHub Actions uses `secrets.GITHUB_TOKEN` to push to GHCR.
The workflow only triggers when files in `src/finetune/` or `data/` change.

### Local Development (before pushing)

While iterating on training code, test locally first:

```bash
# Quick dev test — builds locally, runs 100 steps on GPU
export HF_TOKEN=hf_xxx
bash src/finetune/dev.sh 100
```

### Smoke Test on Training Server

After CI passes, pull and test on your GPU server:

```bash
# Pull CI-built image, run 100-step GPU smoke test
export HF_TOKEN=hf_xxx
export GITHUB_USER=your-github-username
bash src/finetune/test.sh
# Configurable: POETRY_TRAINER_IMAGE, STAGE, STEPS, TEST_DIR
```

What it checks:
- GPU is accessible
- Model loads and tokenizes corpus
- Loss decreases over 100 steps
- No CUDA OOM errors
- Checkpoint saves correctly

### Full Training Run

```bash
# Full Stage 1 training (5000 steps, ~2.5h on RTX 3090)
export HF_TOKEN=hf_xxx
export GITHUB_USER=your-github-username
bash src/finetune/train.sh

# Both stages (auto-chain)
AUTO_CHAIN=1 bash src/finetune/train.sh

# With cloud checkpoint upload
export S3_BUCKET=s3://my-bucket/training
export S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
bash src/finetune/train.sh
```

---

## Step 5: (Optional) Deploy to Salad Cloud

```bash
cd /Users/thanhbm/Projects/poetry-dual-gpt

# Install Python deps for deploy script
pip install requests python-dotenv

# Deploy Stage 1 only (~2.5h, ~$0.23)
python src/finetune/deploy.py --stage 1

# OR: Deploy both stages in one container (~4.5h, ~$0.41)
# python src/finetune/deploy.py --stage 1 --auto-chain
```

What happens:
1. Salad schedules an RTX 3090 node
2. Container starts → `entrypoint.sh` runs
3. Qwen2.5-1.5B is downloaded from HuggingFace
4. Corpus is tokenized
5. QLoRA training begins (5K steps Stage 1, 2K steps Stage 2)
6. Checkpoints auto-uploaded to Cloudflare R2
7. Container exits → billing stops automatically

---

## Step 6: Monitor Training

```bash
# Check container status
python src/finetune/deploy.py --status

# Stream logs (Ctrl+C to stop)
python src/finetune/deploy.py --logs
```

Expected log output:
```
🚀  Qwen2.5-1.5B QLoRA — Stage 1
   Steps: 5,000 | Batch: 4×8=32 | Block: 256 | LR: 0.0002
   GPU: NVIDIA GeForce RTX 3090
   VRAM: 24.0 GB

📦  Loading Qwen/Qwen2.5-1.5B...
🔤  Adding special tokens...
   Added 335 tokens | Vocab: 152,000
🎯  Applying LoRA (r=16, alpha=32)...
   trainable params: 12,582,912 || all params: 1,556,636,160 || trainable%: 0.8084

── Step   200/5000 (18s, ETA: 450s) ── loss=3.2456 val=3.1234 📉 [0/5] 🏆 Best!
── Step   400/5000 (36s, ETA: 414s) ── loss=2.8912 val=2.9012 📉 [0/5] 🏆 Best!
...
── Step  5000/5000 (1800s, ETA: 0s) ── loss=2.3456 val=2.3567 📉

✅  Stage 1 complete! 1800s | Best val: 2.3456

☁️  Uploading checkpoints...
📍  Cloud path: s3://poetry-checkpoints/training-runs/stage1/
```

### What to watch for:

| Signal | Good | Bad |
|--------|------|-----|
| Val loss trending down | ✅ | Loss flat/increasing = LR too high |
| Plateau count | 0-2 | 3+ = might stop early |
| VRAM usage | < 24GB | CUDA OOM = reduce batch or block_size |

---

## Step 7: Download Trained Model

```bash
# Install AWS CLI for R2 access
pip install awscli

# Download Stage 1 checkpoint
aws s3 cp \
    --endpoint-url $S3_ENDPOINT \
    s3://poetry-checkpoints/training-runs/stage1/qwen_stage1_best/ \
    ./checkpoints/qwen_v5_stage1_best/ \
    --recursive

# Verify download
ls checkpoints/qwen_v5_stage1_best/
# Should see: adapter_config.json, adapter_model.safetensors, tokenizer files, training_state.pt
```

**Checkpoint structure:**
```
checkpoints/qwen_v5_stage1_best/
├── adapter_config.json       # LoRA config (r, alpha, target_modules)
├── adapter_model.safetensors # LoRA weights (~50MB)
├── tokenizer.json            # Qwen tokenizer + 335 special tokens
├── tokenizer_config.json
├── special_tokens_map.json
└── training_state.pt         # Optimizer state, step, loss
```

---

## Step 8: Test Inference Locally

```bash
cd /Users/thanhbm/Projects/poetry-dual-gpt

# Install inference deps
pip install transformers peft bitsandbytes torch

# Quick test script
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')

# Load LoRA adapter
model = PeftModel.from_pretrained(model, 'checkpoints/qwen_v5_stage1_best')

# Generate
prompt = '<|start|> [LUC_BAT] Thân em như chẽn lúa đòng\nPhất phơ dưới ngọn nắng hồng ban mai <|reply|>'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.75,
        top_p=0.92,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
# Extract just the response (after <|reply|>)
if '<|reply|>' in response:
    response = response.split('<|reply|>')[-1]
    response = response.replace('<|end|>', '').strip()
print('Response:', response)
"
```

---

## Step 9: Clean Up

```bash
# Stop the Salad container (stops billing)
python src/finetune/deploy.py --delete
```

> **⚠️ Important:** The container restarts with `restart_policy: never`, so it won't restart after training finishes. But always delete to be safe — Salad bills per second the container exists, even if idle.

---

## Cost Summary

| Item | Time | Cost |
|------|------|------|
| Docker build + push | ~15 min | Free |
| Salad Stage 1 (5K steps) | ~2.5h | $0.23 |
| Salad Stage 2 (2K steps, optional) | ~1.0h | $0.09 |
| R2 storage (checkpoint ~200MB) | — | ~$0.003/month |
| **Total (Stage 1 only)** | ~2.5h GPU | **$0.23** |
| **Total (Stage 1 + 2)** | ~3.5h GPU | **$0.32** |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker build` fails | Make sure `corpus/` has the txt files. `ls corpus/` |
| `docker push` 403 | `docker login ghcr.io` with PAT that has `write:packages` |
| Salad "scheduling" stuck | No 3090 nodes available. Wait ~5-15 min or edit `deploy.py` to use `rtx_4090` |
| Container exits immediately | Check logs: `python src/finetune/deploy.py --logs`. Usually missing `HF_TOKEN` or HF license not accepted |
| `CUDA out of memory` | Reduce `batch_size` to 2 or `block_size` to 128 in `train.py` `TRAIN_CONFIG` |
| R2 upload fails | Wrong endpoint URL. Must be `https://<ID>.r2.cloudflarestorage.com` (no bucket name in URL) |
| Can't download from HF | Accept license at https://huggingface.co/Qwen/Qwen2.5-1.5B |
| `deploy.py` can't find `.env` | `cp src/finetune/.env.example src/finetune/.env` then edit |

---

## Appendix: v5 Corpus Format Details

The corpus is generated by `src/preprocess.py` from `data/poems_dataset_clean.csv`.

### Output format

```
<|start|> [LUC_BAT] [RHYME:ong] [TONE:BBTBBT] [TRAMBONG:NH] câu_lục <|reply|> câu_bát <|end|>
<|start|> [THAT_NGON] [LINK2:B] [DOIAM:BTTBBTT] câu_7 <|reply|> câu_7 <|end|>
```

### Special tokens detected by train.py

`src/finetune/train.py`'s `build_special_tokens()` scans the corpus for:
- `[RHYME:\w+]` → rhyme group conditioning
- `[TONE:[BT]+]` → tone pattern conditioning
- `[DOIAM:[BT]+]` → đối âm for Thất Ngôn
- `[LINK2:B]`, `[LINK2:T]` → hardcoded

`[TRAMBONG:NH]` / `[TRAMBONG:HN]` are **not** added as special tokens — they appear
as regular text in the corpus and Qwen learns to condition on them via subword attention.

### Regenerating the corpus

```bash
./venv/bin/python src/preprocess.py \
    --csv data/poems_dataset_clean.csv \
    --output data/poetry_corpus.txt

grep '\[LUC_BAT\]' data/poetry_corpus.txt > data/corpus_luc_bat.txt
```
