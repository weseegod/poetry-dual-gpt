#!/usr/bin/env python3
"""Upload existing checkpoints to Cloudflare R2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import boto3

# Load .env from src/finetune/
load_dotenv(Path(__file__).resolve().parent.parent / "src" / "finetune" / ".env")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# --- Validate ---
missing = []
for name, val in [
    ("S3_BUCKET", S3_BUCKET),
    ("S3_ENDPOINT", S3_ENDPOINT),
    ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
    ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
]:
    if not val:
        missing.append(name)
if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}")
    print("   Fill in src/finetune/.env")
    sys.exit(1)

bucket_name = S3_BUCKET.replace("s3://", "").split("/")[0]
prefix = "/".join(S3_BUCKET.replace("s3://", "").split("/")[1:])

print(f"☁️  Cloudflare R2")
print(f"   Bucket:  {bucket_name}")
print(f"   Prefix:  {prefix}")
print(f"   Endpoint: {S3_ENDPOINT}")
print()

# --- Upload ---
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

for subdir in sorted(CHECKPOINT_DIR.iterdir()):
    if not subdir.is_dir():
        continue
    print(f"📤  {subdir.name}/")
    for fpath in subdir.rglob("*"):
        if not fpath.is_file():
            continue
        key = f"{prefix}/{subdir.name}/{fpath.relative_to(subdir)}"
        s3.upload_file(str(fpath), bucket_name, key)
        print(f"    ✅  s3://{bucket_name}/{key}")

print(f"\n🎉  Done — {bucket_name}/{prefix}/")
