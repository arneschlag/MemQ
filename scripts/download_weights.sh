#!/usr/bin/env bash
# Download the merged MemQ v9 model using public, read-only B2 URLs only.
set -euo pipefail

BASE_URL="https://f003.backblazeb2.com/file/memq-finetunings/v9/merged"
LICENSE_URL="https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/LICENSE?download=true"
# The release name follows the Meta Llama 3 redistribution requirement.
DESTINATION="${1:-models/Llama-3-MemQ-v9}"

if command -v curl >/dev/null 2>&1; then
  downloader() { curl --fail --location --continue-at - --retry 5 --retry-delay 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  downloader() { wget --continue --tries=5 -O "$2" "$1"; }
else
  echo "Install curl or wget before running this script." >&2
  exit 1
fi

mkdir -p "$DESTINATION"
cd "$DESTINATION"

declare -A SHA256=(
  [Modelfile]=785862227631976533b7ea9f31e141fcfd5772e0acf4f30613de1a64e288adfd
  [chat_template.jinja]=ba03a121d097859c7b5b9cd03af99aafe95275210d2876f642ad9929a150f122
  [config.json]=d75ee28b7f24fb4d4c65a461b3bcb9369662842b748d79a6c1eea16caa77fdcf
  [generation_config.json]=117e10970aa9c0e80898c9212371b29121e71c3c7345bc6d2930b1763e20ad94
  [model-00001-of-00004.safetensors]=248b161f38cdcc210d39679fe843d59d5aa2a7160e58c1e5a068bbded6d82019
  [model-00002-of-00004.safetensors]=e4e5f20e47dc72f7eddf22ddc19ccda3273dbf8994eb185a23b9c75c54fbe391
  [model-00003-of-00004.safetensors]=7323828dfa26af6ebdbc163945c03a2557b7624cf7b5a12ee689c36c215fc46f
  [model-00004-of-00004.safetensors]=12a9fe22535ad73c44938191bea4a200b7d9add8ffaec0162bad7a419318b6d5
  [model.safetensors.index.json]=560e89ca3d220e7d9b76fbc0749c2b8e67f7fcfae6d34f3a67ea88fcdc2ab6ae
  [special_tokens_map.json]=994823bd9d0de3b2f59f09f6502a65a228954ca8e5b711534988deb083c37449
  [tokenizer.json]=8c1dcab308e7cf5970ea38815e0a62887d705c5b436f869ca27a5dcdd40c36a6
  [tokenizer_config.json]=0b6caa80b3e57fec440d26673b3b801a1609c5c058402b104872b7c52f69fc22
)

for file in "${!SHA256[@]}"; do
  if [[ -f "$file" ]] && [[ "$(sha256sum "$file" | awk '{print $1}')" == "${SHA256[$file]}" ]]; then
    echo "Already verified: $file"
    continue
  fi
  echo "Downloading: $file"
  downloader "$BASE_URL/$file" "$file"
done

for file in "${!SHA256[@]}"; do
  printf '%s  %s\n' "${SHA256[$file]}" "$file"
done | sha256sum --check --strict

# A copy of the base-model license and its required attribution travel with the
# downloaded derivative model. Both are public URLs/content; no token is used.
downloader "$LICENSE_URL" LICENSE-META-LLAMA-3
cat > NOTICE <<'EOF'
Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
EOF

echo "Model is ready at $(pwd)"
