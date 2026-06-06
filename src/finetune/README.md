# 🔧 Finetune — PoetryDuel-GPT v5 QLoRA

Containerized Qwen2.5-1.5B QLoRA fine-tuning for local GPU and Salad cloud **($0.09/hr)**.

---

## Architecture

```
┌───────────────────────────────────────────┐
│                 Your Laptop               │
│  ┌─────────┐      ┌──────────────────┐   │
│  │  .env   │─────▶│ finetune/deploy.py │   │
│  │ (creds) │      │ (REST API calls)  │   │
│  └─────────┘      └──────┬───────────┘   │
└──────────────────────────┼───────────────┘
                           │ HTTPS
┌──────────────────────────▼───────────────┐
│           Salad Cloud (headless)          │
│  ┌─────────────────────────────────────┐ │
│  │  Docker Container (RTX 3090)        │ │
│  │  ┌───────────────────────────────┐  │ │
│  │  │ train.py                      │  │ │
│  │  │  1. Download Qwen2.5-1.5B     │  │ │
│  │  │  2. Tokenize corpus           │  │ │
│  │  │  3. QLoRA fine-tune           │  │ │
│  │  │  4. Upload checkpoints → R2/  │  │ │
│  │  │     GCS/S3                    │  │ │
│  │  │  5. Exit (billing stops)      │  │ │
│  │  └───────────────────────────────┘  │ │
│  └─────────────────────────────────────┘ │
└───────────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               │   Cloudflare R2 /     │
               │   GCS / S3            │
               └───────────────────────┘
```

---

## Quick Start

### 1. Clone the `.env` file

```bash
cp .env.example .env
# Edit .env with your real credentials
```

### 2. Set up accounts (one time)

| Service | Where | What to get | Cost |
|---------|-------|-------------|------|
| **Salad** | [portal.salad.com](https://portal.salad.com) | API Key, Org name, Project name | Min $5 deposit |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Read token | Free |
| **HuggingFace** | [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Click "Agree and access repository" | Free |
| **Cloudflare R2** | [dash.cloudflare.com](https://dash.cloudflare.com) → R2 | Bucket, endpoint URL, access key + secret | $0.015/GB (free tier: 10GB) |
| **GitHub** | [github.com/settings/tokens](https://github.com/settings/tokens) | Classic token with `write:packages` | Free |

### 3. Configure `.env`

```ini
# Salad
SALAD_API_KEY=salad_xxx
SALAD_ORG=your-org
SALAD_PROJECT=your-project
SALAD_IMAGE=ghcr.io/YOUR_USER/poetry-trainer:latest

# HuggingFace
HF_TOKEN=hf_xxx

# Cloudflare R2 (recommended — cheapest, no egress fees)
S3_BUCKET=s3://poetry-checkpoints/training-runs
S3_ENDPOINT=https://abc123.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your-r2-access-key
AWS_SECRET_ACCESS_KEY=your-r2-secret-key
```

### 4. Build & push image

```bash
# Corpus is generated from data/poems_dataset_clean.csv
# See GUIDE.md Step 3 for full instructions

# Build
docker build -t ghcr.io/YOUR_USER/poetry-trainer:latest .

# Push
docker login ghcr.io -u YOUR_USER  # use GitHub PAT as password
docker push ghcr.io/YOUR_USER/poetry-trainer:latest
```

### 5. Deploy

```bash
# Stage 1 only — ~$0.23, ~2.5 hrs
python deploy.py --stage 1

# Both stages — ~$0.41, ~4.5 hrs
python deploy.py --stage 1 --auto-chain

# Monitor
python deploy.py --status
python deploy.py --logs

# Delete when done
python deploy.py --delete
```

---

## Files

| File | Purpose |
|------|---------|
| `.env.example` | Template — copy to `.env` and fill in credentials |
| `train.py` | Standalone QLoRA training script |
| `deploy.py` | Salad REST API client — loads `.env`, deploys, monitors, deletes |
| `Dockerfile` | Container definition — PyTorch + CUDA 12.4 + corpus + entrypoint |
| `entrypoint.sh` | Runtime setup (GCS creds, GPU info) then launches `train.py` |
| `requirements.txt` | Python deps (transformers, peft, bitsandbytes, boto3, etc.) |

---

## Cloudflare R2 Setup

R2 is the recommended storage — **$0.015/GB/month** with **zero egress fees**, compared
to $0.09/GB (GCS) or $0.09/GB (S3) for data transfer out.

### Step by step

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **R2**
2. Click **Create bucket** → name: `poetry-checkpoints` → Create
3. Go to **Manage R2 API Tokens** → **Create API Token**
   - Permission: **Object Read & Write**
   - Select your bucket → Create
   - Copy:
     - **Access Key ID** → `AWS_ACCESS_KEY_ID`
     - **Secret Access Key** → `AWS_SECRET_ACCESS_KEY`
4. Go back to **R2 Overview** → copy your **Account ID**
5. Your endpoint is: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

### `.env` config

```ini
S3_BUCKET=s3://poetry-checkpoints/training-runs
S3_ENDPOINT=https://abc123def456.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=abc123...
AWS_SECRET_ACCESS_KEY=xyz789...
```

Checkpoints appear at: `s3://poetry-checkpoints/training-runs/stage1/qwen_stage1_best/`

---

## Resuming from Cloud

```bash
# Download Stage 1 checkpoint from R2
aws s3 cp --endpoint-url $S3_ENDPOINT \
  s3://poetry-checkpoints/training-runs/stage1/qwen_stage1_best/ \
  ./checkpoints/qwen_stage1_best/ --recursive

# Deploy Stage 2
python deploy.py --stage 2 --resume /app/checkpoints/qwen_stage1_best
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for Qwen2.5-1.5B download |
| **R2 / S3** | | |
| `S3_BUCKET` | For R2/S3 | Bucket path (e.g. `s3://bucket/prefix`) |
| `S3_ENDPOINT` | For R2 | `https://<id>.r2.cloudflarestorage.com` |
| `AWS_ACCESS_KEY_ID` | For R2/S3 | Access key |
| `AWS_SECRET_ACCESS_KEY` | For R2/S3 | Secret key |
| `AWS_DEFAULT_REGION` | For S3 only | e.g. `us-east-1` |
| **GCS** | | |
| `GCS_BUCKET` | For GCS | Bucket path (e.g. `gs://bucket/prefix`) |
| `GCS_CREDENTIALS_JSON` | For GCS | Full service account JSON (single line) |
| **Training** | | |
| `AUTO_CHAIN_STAGES` | No | `1` to run Stage 1 → Stage 2 in one container |

---

## Cost Analysis

| | Colab L4 | Salad 3090 |
|---|----------|------------|
| **$/hr** | $0.17 | **$0.09** |
| **Stage 1 (5K steps)** | ~3.5 hrs / $0.60 | ~2.5 hrs / **$0.23** |
| **Stage 2 (2K steps)** | ~1.5 hrs / $0.26 | ~1.0 hrs / **$0.09** |
| **Both stages** | ~5 hrs / $0.85 | ~3.5 hrs / **$0.32** |

Plus: no disconnects, no idle costs, checkpoints auto-uploaded.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Container exits immediately | Check `SALAD_ORG` / `SALAD_PROJECT` match Salad dashboard |
| `CUDA out of memory` | 3090 has 24GB. If using larger model, reduce `LORA_CONFIG["r"]` to 8 |
| `403: access restricted` from HF | Didn't accept Qwen license — [click here](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| R2 upload fails `403` | Wrong endpoint URL. Must be `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| No checkpoint uploaded | Check `S3_BUCKET` / `S3_ENDPOINT` are set. Read container logs: `python deploy.py --logs` |
| Salad stuck in "scheduling" | No 3090 nodes available. Wait or add `"gtx_4090"` as GPU fallback |
| `.env` not loading | `deploy.py` won't override existing env vars. `unset` conflicting vars first |
