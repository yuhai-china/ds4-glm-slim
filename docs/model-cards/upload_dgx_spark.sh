#!/bin/bash
# Upload the two DGX Spark GGUF releases to Hugging Face under the `autotrust` org as PRIVATE repos.
#   autotrust/GLM-5.3-Flash-GGUF-DGX-Spark   <- /root/glm-5.3-gguf/glm-5.3-flash-e256-q2   (79 GiB)
#   autotrust/GLM-5.3-GGUF-DGX-Spark         <- /root/glm-5.3-gguf/GLM-5.3-SLIM-E192-GGUF  (150 GiB)
# Uses `hf upload-large-folder` (resumable, multi-part, parallel); re-run the script to resume.
# Usage: ./upload_dgx_spark.sh [flash|glm|all]   (default: all)
set -euo pipefail
ORG=autotrust
FLASH_REPO=$ORG/GLM-5.3-Flash-GGUF-DGX-Spark
GLM_REPO=$ORG/GLM-5.3-GGUF-DGX-Spark
FLASH_DIR=/root/glm-5.3-gguf/glm-5.3-flash-e256-q2
GLM_DIR=/root/glm-5.3-gguf/GLM-5.3-SLIM-E192-GGUF
WORKERS=${WORKERS:-4}
WHAT=${1:-all}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run 'hf auth login' first"; exit 1; }
hf auth whoami 2>/dev/null | grep -q "$ORG" || { echo "the logged-in account is not a member of org '$ORG'"; exit 1; }

verify(){  # DIR
  log "verifying checksum in $1"
  (cd "$1" && for f in *.sha256; do sha256sum -c "$f"; done)
}

upload(){  # REPO DIR
  local repo=$1 dir=$2
  log "creating private repo $repo (no-op if it exists)"
  hf repo create "$repo" --repo-type model --private 2>&1 | tail -1 || true
  # make sure it is private even if it already existed
  python3 - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi(); repo = sys.argv[1]
api.update_repo_settings(repo_id=repo, private=True)
print(f"{repo}: private = {api.model_info(repo).private}")
PY
  log "uploading $dir -> $repo"
  hf upload-large-folder "$repo" "$dir" --repo-type model --private --num-workers "$WORKERS"
  log "done: https://huggingface.co/$repo (private)"
}

case "$WHAT" in
  flash) verify "$FLASH_DIR"; upload "$FLASH_REPO" "$FLASH_DIR" ;;
  glm)   verify "$GLM_DIR";   upload "$GLM_REPO"   "$GLM_DIR" ;;
  all)   verify "$FLASH_DIR"; verify "$GLM_DIR"
         upload "$FLASH_REPO" "$FLASH_DIR"
         upload "$GLM_REPO"   "$GLM_DIR" ;;
  *) echo "usage: $0 [flash|glm|all]"; exit 1 ;;
esac
log "ALL DONE"
