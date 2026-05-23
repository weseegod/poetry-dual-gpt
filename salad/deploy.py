#!/usr/bin/env python3
"""
Deploy PoetryDuel-GPT training to Salad Cloud.

Usage:
  # Deploy Stage 1 only
  python deploy.py --stage 1

  # Deploy both stages (Stage 2 auto-chains after Stage 1 completes)
  python deploy.py --stage 1 --auto-chain

  # Deploy Stage 2 from a previous Stage 1 checkpoint
  python deploy.py --stage 2 --resume gs://my-bucket/checkpoints/stage1/qwen_stage1_best

  # Check status / get logs
  python deploy.py --status
  python deploy.py --logs

Prerequisites:
  export SALAD_API_KEY="..."
  export HF_TOKEN="..."
  export GCS_BUCKET="gs://your-bucket/poetry-checkpoints"  # or S3_BUCKET
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# ── Load .env file ──
ENV_FILE = Path(__file__).resolve().parent / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("\"'")
            if key and key not in os.environ:  # Don't override existing env vars
                os.environ[key] = val

SALAD_API = "https://api.salad.com/api/public/v1"
ORG_NAME = os.environ.get("SALAD_ORG", "")
PROJ_NAME = os.environ.get("SALAD_PROJECT", "")
API_KEY = os.environ.get("SALAD_API_KEY", "")
IMAGE = os.environ.get("SALAD_IMAGE", "ghcr.io/YOUR_USER/poetry-trainer:latest")

HEADERS = {
    "Salad-Api-Key": API_KEY,
    "Content-Type": "application/json",
}


def _url(path):
    # Salad V1 URL pattern
    base = f"{SALAD_API}/organizations/{ORG_NAME}/projects/{PROJ_NAME}"
    return f"{base}/{path.lstrip('/')}"


def get_container_group(name="poetry-train"):
    """Get existing container group by name."""
    resp = requests.get(_url("containergroups"), headers=HEADERS)
    resp.raise_for_status()
    groups = resp.json().get("items", [])
    for g in groups:
        if g.get("name") == name:
            return g
    return None


def deploy(stage=1, auto_chain=False, resume=None):
    """Deploy a Salad container group for training."""
    print(f"\n{'='*60}")
    print(f"🍃  Salad Deployment — PoetryDuel-GPT Stage {stage}")
    print(f"{'='*60}\n")

    if not all([ORG_NAME, PROJ_NAME, API_KEY]):
        print("❌  Missing env vars: SALAD_ORG, SALAD_PROJECT, SALAD_API_KEY")
        sys.exit(1)

    # ── Check for existing group ──
    existing = get_container_group()
    if existing and existing.get("current_state") not in ("stopped", "deleted"):
        print(f"⚠️  Container group '{existing['name']}' is {existing['current_state']}")
        ans = input("   Delete and recreate? [y/N]: ")
        if ans.lower() == "y":
            requests.delete(_url(f"containergroups/{existing['id']}"), headers=HEADERS)
            print("   Deleted old group — waiting 10s for cleanup...")
            time.sleep(10)
        else:
            print("   Keeping existing group. Exiting.")
            return

    # ── Build command ──
    cmd = ["--stage", str(stage)]
    if resume:
        cmd.extend(["--resume", resume])

    # ── Environment (pass to container) ──
    env_vars = {}

    # HuggingFace
    for var in ["HF_TOKEN"]:
        if os.environ.get(var):
            env_vars[var] = os.environ[var]

    # Cloud storage — GCS
    for var in ["GCS_BUCKET", "GCS_CREDENTIALS_JSON"]:
        if os.environ.get(var):
            env_vars[var] = os.environ[var]

    # Cloud storage — S3 / Cloudflare R2
    for var in ["S3_BUCKET", "S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION"]:
        if os.environ.get(var):
            env_vars[var] = os.environ[var]

    if auto_chain:
        env_vars["AUTO_CHAIN_STAGES"] = "1"

    # Detect which storage is configured
    gcs_bucket = os.environ.get("GCS_BUCKET", "")
    s3_bucket = os.environ.get("S3_BUCKET", "")

    # ── Create container group ──
    body = {
        "name": "poetry-train",
        "container": {
            "image": IMAGE,
            "command": cmd,
            "resources": {
                "cpu": 4,
                "memory": "16Gi",
                "gpu_classes": ["gtx_3090"],  # $0.09/hr, 24GB VRAM
            },
            "environment_variables": env_vars,
            "logging": {"new_relic": None},  # stdout logs
        },
        "replicas": 1,
        "autostart_policy": True,
        "restart_policy": "never",  # Don't restart after training completes
    }

    print(f"🚀  Deploying container group...")
    resp = requests.post(_url("containergroups"), headers=HEADERS, json=body)
    if resp.status_code != 200:
        print(f"❌  Failed: {resp.status_code}\n{resp.text}")
        sys.exit(1)

    cg = resp.json()
    cg_id = cg.get("id", "?")
    print(f"✅  Created: {cg_id}")
    print(f"   Image:  {IMAGE}")
    print(f"   GPU:    gtx_3090 (~$0.09/hr)")
    print(f"   Stage:  {stage}  {'[auto-chain → S2]' if auto_chain else ''}")
    print(f"   Upload: {gcs_bucket or s3_bucket or '⚠️  NONE — checkpoints will be lost!'}")
    print()
    print(f"⏳  Waiting for instance to start (pulls image, downloads Qwen1.5B)...")

    # ── Wait for running ──
    for attempt in range(60):  # up to 10 min
        time.sleep(10)
        cg_resp = requests.get(_url(f"containergroups/{cg_id}"), headers=HEADERS)
        if cg_resp.status_code != 200:
            continue
        cg = cg_resp.json()
        state = cg.get("current_state", "?")
        instances = cg.get("instances", [])
        if instances:
            inst_state = instances[0].get("state", "?")
            print(f"   [{attempt*10}s] group={state}  instance={inst_state}")
            if inst_state == "running":
                break
    else:
        print("⚠️  Timeout waiting for running. Check Salad dashboard.")

    print(f"\n📋  Training has started. Monitor with:")
    print(f"      python deploy.py --logs")
    print(f"      python deploy.py --status")
    print(f"\n💡  The container will auto-upload checkpoints and exit when done.")
    print(f"    Billing stops automatically on exit (restart_policy=never).")


def get_logs():
    """Fetch recent logs from the training container."""
    cg = get_container_group()
    if not cg:
        print("❌  No container group 'poetry-train' found.")
        return

    resp = requests.get(_url(f"containergroups/{cg['id']}/logs"), headers=HEADERS)
    if resp.status_code != 200:
        print(f"❌  Failed: {resp.status_code}\n{resp.text}")
        return

    logs = resp.json()
    for entry in logs.get("items", [])[-50:]:
        print(entry.get("message", ""))


def check_status():
    """Check the current status of the training container."""
    cg = get_container_group()
    if not cg:
        print("❌  No container group 'poetry-train' found.")
        return

    state = cg.get("current_state", "?")
    instances = cg.get("instances", [])
    inst_state = instances[0].get("state", "?") if instances else "?"
    machine_type = instances[0].get("machine_type", "?") if instances else "?"

    print(f"\n🍃  Poetry-Train Status")
    print(f"   Group:     {state}")
    print(f"   Instance:  {inst_state}")
    print(f"   Machine:   {machine_type}")

    # Estimate cost
    created = cg.get("create_time", "")
    if created:
        # crude estimate
        print(f"   Created:   {created}")

    if state == "running":
        print(f"\n   💰  Billing active (~$0.09/hr)")
    elif state == "stopped":
        print(f"   ✅  Training complete — billing stopped")


def delete():
    """Delete the container group to free resources."""
    cg = get_container_group()
    if not cg:
        print("❌  No container group 'poetry-train' found.")
        return

    resp = requests.delete(_url(f"containergroups/{cg['id']}"), headers=HEADERS)
    if resp.status_code in (200, 204):
        print(f"🗑️  Deleted: {cg['id']}")
    else:
        print(f"❌  Failed: {resp.status_code}\n{resp.text}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salad deployment for PoetryDuel-GPT")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--auto-chain", action="store_true",
                        help="Auto-run Stage 2 after Stage 1 completes (single container)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from cloud checkpoint (gs:// or s3:// path)")
    parser.add_argument("--status", action="store_true", help="Check training status")
    parser.add_argument("--logs", action="store_true", help="Fetch training logs")
    parser.add_argument("--delete", action="store_true", help="Delete the container group")
    parser.add_argument("--image", type=str, default=None,
                        help="Override the container image")
    args = parser.parse_args()

    if args.image:
        IMAGE = args.image

    if args.status:
        check_status()
    elif args.logs:
        get_logs()
    elif args.delete:
        delete()
    else:
        deploy(stage=args.stage, auto_chain=args.auto_chain, resume=args.resume)
