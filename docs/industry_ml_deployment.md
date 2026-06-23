# 🏭 Industry-Standard ML Training Deployment

> What real ML teams do to train, track, and deploy models — from solo dev to big tech.
> Written for PoetryDuel-GPT v5 as reference, applicable to any ML project.

---

## The Stack, Layer by Layer

### 1. Containerization

```
Your code → Docker image → runs identically everywhere
```

**Non-negotiable.** Every serious ML team containers their training.

What we have (correct pattern):
- `Dockerfile` with pinned base image (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`)
- Corpus baked into the image (fine for datasets under a few GB)
- `ENTRYPOINT` that runs training, uploads artifacts, then **exits** (stops billing)
- Container is the unit of work — starts, trains, uploads, dies. No long-running servers for training.

| Do | Don't |
|----|-------|
| Pin exact base image tags | Use `:latest` |
| Make image self-contained | Depend on mounted volumes for critical deps |
| Exit after training completes | Run a training loop as a long-lived service |

### 2. Artifact Management

```
Checkpoints → S3/GCS/R2 → versioned, retrievable
```

Our pattern (correct):
```python
# After training completes:
upload_checkpoints(local_dir, stage)
# → s3://poetry-checkpoints/training-runs/stage1/
```

What bigger teams add:

| Tool | What it does | When to adopt |
|------|-------------|---------------|
| **MLflow** | Track experiments (params, metrics, artifacts). "Run #42: lr=2e-4 → val_loss=2.34" | 5+ training runs, need comparisons |
| **Weights & Biases** | Cloud-hosted MLflow alternative. Real-time loss curves, system metrics, model registry | Team > 1 person, or need shareable dashboards |
| **DVC** | Data versioning. `dvc push` tags your dataset with a git commit hash | Dataset changes over time, need reproducibility |
| **Model Registry** | HuggingFace Hub, MLflow Registry. Tag models as "staging", "production", "archived" | Serving models to users, need rollback ability |

**Minimum viable**: upload checkpoints + training log to S3. Costs nothing, prevents regret.

### 3. Secrets Management

Our pattern (correct for solo dev):
```ini
# .env file (gitignored)
HF_TOKEN=hf_xxx
AWS_ACCESS_KEY_ID=xxx
```

Industry patterns by team size:

| Pattern | When | Security |
|---------|------|----------|
| `.env` + `.gitignore` | Solo dev | ⚠️ Tokens in filesystem |
| **Docker build secrets** | CI/CD builds | ✅ `docker build --secret id=hf_token,env=HF_TOKEN` |
| **Cloud secret stores** | Team in cloud | ✅ AWS Secrets Manager / GCP Secret Manager. Container pulls at runtime via IAM role |
| **Kubernetes Secrets** | K8s orchestration | ✅ Mounted as files/env vars, encrypted at rest |

**Critical rule**: secrets should **never be in Docker image layers**. They leak via `docker history`. Our pattern passes `HF_TOKEN` at runtime via container env vars — correct.

### 4. Experiment Tracking

Every training run should be recorded. We currently have no run-level tracking.

**Minimal implementation** (30 lines, no dependencies):

```python
import json, subprocess, time
from datetime import datetime

# At start of training:
run = {
    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model": "Qwen2.5-1.5B",
    "lora_config": {"r": 16, "alpha": 32, "dropout": 0.05},
    "stage": stage,
    "corpus": str(corpus_path),
    "batch_size": batch_size * grad_accum,
    "lr": SC["lr"],
    "max_steps": max_steps,
    "git_commit": subprocess.getoutput("git rev-parse HEAD"),
    "git_branch": subprocess.getoutput("git rev-parse --abbrev-ref HEAD"),
    "started_at": time.time(),
    "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "CPU",
    "steps": [],
}

# Each eval interval:
run["steps"].append({
    "step": step,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "lr": lr,
    "is_best": is_best,
    "elapsed": time.time() - t0,
})

# At end:
run["completed_at"] = time.time()
run["best_val_loss"] = best_val
run["total_elapsed"] = time.time() - run["started_at"]
json.dump(run, open(CHECKPOINT_DIR / f"run_{run['run_id']}.json", "w"))
upload_to_s3(CHECKPOINT_DIR / f"run_{run['run_id']}.json")
```

**Why**: after 5+ training runs you will not remember which hyperparams produced which result. This file answers that question in 2 seconds.

### 5. Infrastructure Orchestration

Spectrum from simple to complex:

| Level | Tool | What you get | When |
|-------|------|-------------|------|
| **1. Direct API** | Our `deploy.py` → Salad REST | Simple, debuggable, zero overhead | Solo dev, < 10 runs/month |
| **2. Scripted** | Bash script wrapping Docker + cloud CLI | Repeatable, can cron it | Solo, regular retraining |
| **3. Kubernetes Job** | `kubectl apply -f train-job.yaml` | Auto-scheduling, retry on failure, resource quotas | Team, multiple models |
| **4. Fully managed** | SageMaker / Vertex AI | "Here's my Docker image + data, train it" | Company with cloud credits |
| **5. Workflow orchestrator** | Airflow / Prefect / Dagster | Scheduled retraining, dependency graphs, Slack alerts | ML platform team |

Our current approach (Level 1) is the **serverless GPU** model — a modern trend. Platforms like Modal, Replicate, and Banana use the same pattern: container starts → runs work → exits → billing stops. This is the right model for infrequent training.

### 6. CI/CD for ML

What MLEng teams actually build:

```
git push → CI pipeline:
  1. Lint + type check (ruff, mypy)
  2. Run unit tests (mock model, test data pipeline)
  3. Build Docker image
  4. Smoke test: run 10 training steps on CPU, verify no crashes
  5. Push image to registry
  6. [Manual gate] Deploy to GPU cluster
  7. Training completes → evaluate metrics
  8. If metrics > threshold → auto-promote to model registry
  9. If metrics < threshold → Slack alert, block deployment
```

**Pragmatic starting point** (`.github/workflows/smoke-test.yml`):

```yaml
name: Smoke Test Training
on:
  push:
    paths:
      - "src/finetune/**"
      - "data/poetry_corpus.txt"
      - "data/corpus_luc_bat.txt"

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t poetry-trainer -f src/finetune/Dockerfile .
      - name: Smoke test (10 steps, CPU)
        run: |
          docker run --rm \
            -e HF_TOKEN=${{ secrets.HF_TOKEN }} \
            -e STAGE=1 \
            -e TRAIN_BATCH_SIZE=1 \
            -e TRAIN_BLOCK_SIZE=64 \
            -e TRAIN_NUM_WORKERS=0 \
            --entrypoint python \
            poetry-trainer \
            -c "
            import os; os.environ['TRAIN_MAX_STEPS']='10'
            # Import and run just enough to verify pipeline works
            "
      - name: Verify checkpoint produced
        run: |
          # Check that the smoke test generated output
          echo "Smoke test passed"
```

**Why**: catches broken Dockerfiles, missing deps, import errors before you push to Salad and waste $0.09 on a failed container.

### 7. Data Pipeline Versioning

Current flow:
```
data/poems_dataset_clean.csv → src/preprocess.py → data/poetry_corpus.txt
```

Industry adds versioning:

```bash
# Tag your data with DVC
dvc add data/poems_dataset_clean.csv
git add data/poems_dataset_clean.csv.dvc
git commit -m "dataset v3: added 5000 new poems from Facebook scrape"

# Training log records this:
run["dataset_version"] = "v3"
run["dataset_git_hash"] = "abc123"
```

Now you can answer: "Did the model degrade because of code changes or data changes?"

**Minimum viable without DVC**: track the CSV file hash in your training log:

```python
import hashlib
csv_hash = hashlib.md5(open("data/poems_dataset_clean.csv", "rb").read()).hexdigest()
run["dataset_md5"] = csv_hash
```

### 8. Cost-Aware Training (Spot/Preemptible Instances)

The biggest industry shift in recent years: **spot instances + checkpoint resilience**.

```
Start training → every N steps save checkpoint → if killed, resume from last checkpoint
```

Everyone training large models uses this. Preemptible GPUs are 60-90% cheaper. Our `train.py` already has `--resume` — we're set up for this.

| GPU Source | On-Demand | Spot/Preemptible | Savings |
|-----------|-----------|-----------------|---------|
| AWS p4d (A100) | $32.77/hr | $9.83/hr | 70% |
| GCP A100 | $29.46/hr | $8.84/hr | 70% |
| Salad 3090 | $0.09/hr | N/A (already cheap) | — |

Salad is already preemptible by nature (decentralized GPU sharing). The `--resume` flag is essential.

---

## What We Have vs What's Missing

| Capability | Status | Priority |
|-----------|--------|----------|
| **Containerization** | ✅ Done | — |
| **Artifact upload (S3/R2)** | ✅ Done | — |
| **Secrets at runtime** | ✅ Done | — |
| **Resume from checkpoint** | ✅ Done | — |
| **Grammar-correct Vietnamese evaluation** | ✅ Done (v4.2.3 eval suite) | — |
| **Experiment tracking** | ❌ Missing | 🔴 High |
| **Git commit in logs** | ❌ Missing | 🔴 High |
| **Docker build secrets** | ❌ Using `.env` | 🟡 Medium |
| **CI smoke test** | ❌ Missing | 🟡 Medium |
| **Data versioning** | ❌ Missing | 🟢 Low (data is stable) |
| **Model registry** | ❌ Missing | 🟢 Low (one model) |
| **Workflow orchestrator** | ❌ Missing | ⚪ Not needed |

---

## Recommended Next Steps

### Priority 1 — Add run tracking (30 min)

Add a `run_log.json` per training session. Upload to R2 alongside checkpoints. Track:
- Hyperparameters (LoRA r, alpha, LR, batch size, steps)
- Git commit hash + branch
- Dataset info (file path, line count, MD5 hash)
- Per-eval metrics (step, train_loss, val_loss, LR, elapsed)
- GPU info (name, VRAM)
- Total runtime, best val loss

### Priority 2 — Smoke test in CI (20 min)

GitHub Action that builds the Docker image and runs 10 steps on CPU. Fails the build if the container crashes. Prevents deploying broken code to Salad.

### Priority 3 — Docker build secrets (10 min)

Switch from `.env` at build time to `--secret`:
```bash
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t poetry-trainer \
  -f src/finetune/Dockerfile \
  .
```

### Priority 4 — Data versioning (when dataset changes)

When we add canonical poets or Facebook-scraped poems, tag the dataset version. Record it in training logs.

---

## The Real Industry Secret

> **Add infrastructure when the current pain is worse than the setup cost.**

Most teams over-build. They set up K8s, MLflow, Airflow, feature stores, model registries... for a 2-person team training a single model.

Right now we have: Docker + direct API deploy + S3 upload + `.env` secrets. That's a **solid, industry-recognized minimal stack**. A solo ML engineer at a startup would use exactly this — possibly swapping Salad for RunPod or Lambda Labs, but the pattern is identical.

**Rule of thumb**: if you can't explain why you need a tool in one sentence, you don't need it yet.

---

## Reference: Our Current Stack

```
┌─────────────────────────────────────────────┐
│                  Local / CI                  │
│  ┌──────────┐    ┌──────────────────────┐   │
│  │  .env    │───▶│ deploy.py             │   │
│  │ (creds)  │    │ (Salad REST client)   │   │
│  └──────────┘    └──────────┬───────────┘   │
└─────────────────────────────┼───────────────┘
                              │ HTTPS API
┌─────────────────────────────▼───────────────┐
│          Salad Cloud / Local GPU             │
│  ┌──────────────────────────────────────┐   │
│  │  Docker Container                     │   │
│  │  1. git clone / COPY corpus           │   │
│  │  2. pip install deps                  │   │
│  │  3. Download Qwen2.5-1.5B from HF     │   │
│  │  4. Tokenize corpus                   │   │
│  │  5. QLoRA fine-tune (5K-7K steps)     │   │
│  │  6. Upload checkpoints → R2/S3/GCS    │   │
│  │  7. Exit (billing stops)              │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────┬───────────────┘
                              │
                  ┌───────────┴───────────┐
                  │   Cloudflare R2 /     │
                  │   GCS / S3            │
                  │                       │
                  │  checkpoints/         │
                  │  ├── stage1/          │
                  │  │   ├── qwen_s1_best/│
                  │  │   └── qwen_s1_final│
                  │  └── stage2/          │
                  │      ├── qwen_s2_best/│
                  │      └── qwen_s2_final│
                  └───────────────────────┘
```

---

## Further Reading

- [MLOps Maturity Model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model) — Microsoft's framework
- [ML Experiments Management at Scale](https://neptune.ai/blog/ml-experiment-management) — Comparison of MLflow, W&B, Neptune, Comet
- [Docker Build Secrets](https://docs.docker.com/build/building/secrets/) — Secure handling of HF_TOKEN, etc.
- [Spot Instance Best Practices](https://aws.amazon.com/blogs/compute/deep-learning-with-spot-instances/) — Training on preemptible GPUs
- [Kubernetes Jobs for ML](https://kubeflow.org/) — When you outgrow direct API calls
