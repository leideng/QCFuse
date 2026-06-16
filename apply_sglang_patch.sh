#!/usr/bin/env bash
# Apply QCFuse SGLang runtime patches onto the installed sglang package.
#
# Usage:
#   # With uv project venv (recommended):
#   uv run bash apply_sglang_patch.sh
#
#   # Or with an activated venv / conda env:
#   bash apply_sglang_patch.sh
#
#   # Or point at a specific Python:
#   PYTHON=/path/to/python bash apply_sglang_patch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QCFUSE_ROOT="$SCRIPT_DIR"
PATCH_SRC="${QCFUSE_ROOT}/srt"
EXPECTED_SGLANG_VERSION="0.5.4"

if [[ ! -d "$PATCH_SRC" ]]; then
  echo "error: QCFuse srt/ not found at: $PATCH_SRC" >&2
  exit 1
fi

# Prefer uv's project interpreter when available.
if [[ -z "${PYTHON:-}" ]]; then
  if command -v uv >/dev/null 2>&1 && [[ -f "${QCFUSE_ROOT}/pyproject.toml" ]]; then
    PYTHON="uv run python"
  else
    PYTHON="python"
  fi
fi

echo "Using Python: $PYTHON"

SGLANG_VERSION="$($PYTHON -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))")"
SGLANG_SRT="$($PYTHON -c "
import sglang.srt
path = getattr(sglang.srt, '__file__', None)
if path:
    import os
    print(os.path.dirname(path))
else:
    print(sglang.srt.__path__[0])
")"

echo "Installed sglang version: $SGLANG_VERSION"
echo "Target sglang.srt path:    $SGLANG_SRT"

if [[ "$SGLANG_VERSION" != "$EXPECTED_SGLANG_VERSION" ]]; then
  echo "warning: expected sglang $EXPECTED_SGLANG_VERSION, found $SGLANG_VERSION" >&2
  echo "warning: QCFuse srt/ is based on v0.5.4; mismatches may cause runtime errors." >&2
fi

BACKUP_DIR="${SGLANG_SRT}.qcfuse_backup"
if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Creating one-time backup at: $BACKUP_DIR"
  cp -a "$SGLANG_SRT" "$BACKUP_DIR"
else
  echo "Backup already exists at: $BACKUP_DIR (skipping)"
fi

echo "Applying QCFuse srt/ overlay..."
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${PATCH_SRC}/" "${SGLANG_SRT}/"
else
  rm -rf "${SGLANG_SRT:?}"/*
  cp -a "${PATCH_SRC}/." "$SGLANG_SRT/"
fi

echo "Verifying patched modules..."
$PYTHON -c "
from sglang.srt.utils.digest_index_manager import DigestIndexManager, DIGEST_INDEX_VERSION
from sglang.srt.utils.kv_ssd_manager import PACKED_KV_FORMAT, QUERY_CACHE_FORMAT
print('Patch applied successfully.')
print(f'  digest_index version: {DIGEST_INDEX_VERSION}')
print(f'  packed kv format:     {PACKED_KV_FORMAT}')
print(f'  query cache format:   {QUERY_CACHE_FORMAT}')
"

echo
echo "Done. You can now run:"
echo "  uv run bash run_qcfuse.sh"
echo
echo "Note: re-running 'uv sync' may reinstall vanilla sglang and undo this patch."
echo "      Re-run this script after 'uv sync' if needed."
