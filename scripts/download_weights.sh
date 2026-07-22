#!/usr/bin/env bash
# Download a merged MemQ model using public, read-only B2 URLs only.
#
#   scripts/download_weights.sh                      # default version
#   MEMQ_MODEL_VERSION=v14 scripts/download_weights.sh
#   MEMQ_MODEL_VERSION=v9  scripts/download_weights.sh models/my-dir
#
# Versions:
#   v9  - WebQSP+CWQ only (the model the seminar report evaluates)
#   v14 - joint WebQSP+CWQ+GrailQA, full operator support (count, extrema,
#         comparison, literals, type-anchored questions)
#
# Checksums live in scripts/checksums/<version>.sha256 and double as the file
# list, so adding a version means adding one file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# v14 is the current model; set MEMQ_MODEL_VERSION=v9 to fetch the one the
# seminar report's WebQSP/CWQ numbers were measured on.
VERSION="${MEMQ_MODEL_VERSION:-v14}"
BASE_URL="${MEMQ_MODEL_BASE_URL:-https://f003.backblazeb2.com/file/memq-finetunings/$VERSION/merged}"
LICENSE_URL="https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/LICENSE?download=true"
# The release name follows the Meta Llama 3 redistribution requirement.
DESTINATION="${1:-models/Llama-3-MemQ-$VERSION}"

CHECKSUMS="$ROOT/scripts/checksums/$VERSION.sha256"
if [[ ! -s "$CHECKSUMS" ]]; then
  echo "Unknown model version '$VERSION' (no $CHECKSUMS)." >&2
  echo "Available versions:" >&2
  ls -1 "$ROOT/scripts/checksums/" 2>/dev/null | sed 's/\.sha256$//' >&2
  exit 2
fi

if command -v curl >/dev/null 2>&1; then
  downloader() { curl --fail --location --continue-at - --retry 5 --retry-delay 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  downloader() { wget --continue --tries=5 -O "$2" "$1"; }
else
  echo "Install curl or wget before running this script." >&2
  exit 1
fi

mkdir -p "$DESTINATION"
cp "$CHECKSUMS" "$DESTINATION/SHA256SUMS"
cd "$DESTINATION"

echo "MemQ $VERSION -> $(pwd)"
while read -r expected file; do
  [[ -n "${file:-}" ]] || continue
  if [[ -f "$file" ]] && [[ "$(sha256sum "$file" | awk '{print $1}')" == "$expected" ]]; then
    echo "Already verified: $file"
    continue
  fi
  echo "Downloading: $file"
  downloader "$BASE_URL/$file" "$file"
done < SHA256SUMS

sha256sum --check --strict SHA256SUMS

# A copy of the base-model license and its required attribution travel with the
# downloaded derivative model. Both are public URLs/content; no token is used.
downloader "$LICENSE_URL" LICENSE-META-LLAMA-3
cat > NOTICE <<'EOF'
Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
EOF

echo "Model is ready at $(pwd)"
