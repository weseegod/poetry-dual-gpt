#!/usr/bin/env python3
"""
Salad REST API client — deploy, monitor, and delete GPU containers.

Loads credentials from .env, creates Salad container groups, streams logs,
and cleans up when done.

Usage:
  python deploy.py --stage 1
  python deploy.py --stage 2 --resume /app/checkpoints/qwen_stage1_best
  python deploy.py --stage 1 --auto-chain
  python deploy.py --status
  python deploy.py --logs
  python deploy.py --delete
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

SALAD_API_KEY = os.environ.get("SALAD_API_KEY")
SALAD_ORG = os.environ.get("SALAD_ORG")
SALAD_PROJECT = os.environ.get("SALAD_PROJECT")
SALAD_IMAGE = os.environ.get("SALAD_IMAGE")

BASE_URL = f"https://api.salad.com/api/public/v1/organizations/{SALAD_ORG}/projects/{SALAD_PROJECT}"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Salad-Api-Key": SALAD_API_KEY,
}

GPU_CLASS = "rtx_3090"  # $0.09/hr

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _check_config():
    missing = []
    for var in ["SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT", "SALAD_IMAGE"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"❌  Missing env vars: {', '.join(missing)}")
        print("   Copy .env.example → .env and fill in credentials")
        sys.exit(1)


def _api(method, path, **kwargs):
    url = f"{BASE_URL}/{path.lstrip('/')}"
    resp = requests.request(method, url, headers=HEADERS, **kwargs)
    if resp.status_code >= 400:
        print(f"❌  API error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    return resp.json() if resp.text else {}


# ═══════════════════════════════════════════════════════════════
#  CONTAINER GROUP MANAGEMENT
# ═══════════════════════════════════════════════════════════════

CONTAINER_GROUP_NAME = "poetry-trainer-v5"


def _build_env_vars(stage, resume_from=None, auto_chain=False):
    """Build environment dict for container."""
    env = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "STAGE": str(stage),
    }
    if auto_chain:
        env["AUTO_CHAIN_STAGES"] = "1"
    if resume_from:
        env["RESUME_FROM"] = resume_from

    # Optional cloud storage
    for key in ["S3_BUCKET", "S3_ENDPOINT", "AWS_ACCESS_KEY_ID",
                 "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
                 "GCS_BUCKET", "GCS_CREDENTIALS_JSON"]:
        if os.environ.get(key):
            env[key] = os.environ[key]

    return env


def deploy(stage=1, resume_from=None, auto_chain=False):
    """Create and deploy a container group on Salad."""
    _check_config()

    env_vars = _build_env_vars(stage, resume_from, auto_chain)

    payload = {
        "name": CONTAINER_GROUP_NAME,
        "container": {
            "image": SALAD_IMAGE,
            "resources": {
                "cpu": 4,
                "memory": 16384,  # 16GB
                "gpu_class": GPU_CLASS,
                "gpu_count": 1,
            },
            "environment_variables": env_vars,
            "restart_policy": "never",  # stop billing when training finishes
        },
        "replicas": 1,
    }

    print(f"🚀  Deploying container group '{CONTAINER_GROUP_NAME}'...")
    print(f"   Image: {SALAD_IMAGE}")
    print(f"   GPU: {GPU_CLASS} | Stage: {stage}")
    if auto_chain:
        print(f"   Auto-chain: Stage 1 → Stage 2")
    if resume_from:
        print(f"   Resume from: {resume_from}")

    # Delete existing group if present
    try:
        _api("DELETE", f"/containergroups/{CONTAINER_GROUP_NAME}")
        print("   Deleted existing container group")
        time.sleep(2)
    except Exception:
        pass

    result = _api("POST", "/containergroups", json=payload)
    print(f"✅  Container group created: {result.get('name', CONTAINER_GROUP_NAME)}")
    print(f"   Monitor: python deploy.py --status")
    print(f"   Logs:    python deploy.py --logs")
    return result


def status():
    """Get container group status."""
    _check_config()
    try:
        result = _api("GET", f"/containergroups/{CONTAINER_GROUP_NAME}")
        state = result.get("state", "unknown")
        instances = result.get("instances", [])
        print(f"📊  Container Group: {CONTAINER_GROUP_NAME}")
        print(f"   State: {state}")
        for i, inst in enumerate(instances):
            print(f"   Instance {i}: state={inst.get('state')}, "
                  f"machine={inst.get('machine_state')}")
        return result
    except Exception as e:
        print(f"❌  Error: {e}")
        return None


def logs():
    """Stream container logs."""
    _check_config()
    print("📜  Streaming logs (Ctrl+C to stop)...\n")
    try:
        while True:
            result = _api("GET", f"/containergroups/{CONTAINER_GROUP_NAME}/logs")
            entries = result.get("logs", []) if isinstance(result, dict) else []
            for entry in entries:
                print(entry.get("message", entry))
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped log streaming")


def delete():
    """Delete container group (stops billing)."""
    _check_config()
    try:
        result = _api("DELETE", f"/containergroups/{CONTAINER_GROUP_NAME}")
        print(f"🗑️  Container group '{CONTAINER_GROUP_NAME}' deleted")
        print("   Billing stopped.")
        return result
    except Exception as e:
        print(f"❌  Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salad GPU deployment for PoetryDuel-GPT v5")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto-chain", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    if args.delete:
        delete()
    elif args.status:
        status()
    elif args.logs:
        logs()
    else:
        deploy(stage=args.stage, resume_from=args.resume, auto_chain=args.auto_chain)
