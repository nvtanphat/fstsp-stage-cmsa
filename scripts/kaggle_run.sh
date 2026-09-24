#!/usr/bin/env bash
set -euo pipefail
USERNAME="${1:?Usage: scripts/kaggle_run.sh <username> [slug]}"
SLUG="${2:-fstsp-stage-cmsa-reimplementation}"
python scripts/prepare_kaggle_kernel.py --username "$USERNAME" --slug "$SLUG" --private
KERNEL="$USERNAME/$SLUG"
kaggle kernels push -p dist/kaggle_kernel -t 1200
kaggle kernels status "$KERNEL"
echo "After completion: kaggle kernels output $KERNEL -p artifacts/kaggle_download -o"
