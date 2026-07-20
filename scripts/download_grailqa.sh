#!/usr/bin/env bash
# Download the labelled GrailQA v1.0 release from its official distribution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/data/grailqa"
ARCHIVE="$TARGET/GrailQA_v1.0.zip"
URL="https://dl.orangedox.com/WyaCpL/?dl=1"

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 1; }
mkdir -p "$TARGET"
if [[ ! -s "$TARGET/grailqa_v1.0_dev.json" ]]; then
  echo "Downloading official GrailQA v1.0 release (CC BY-SA 4.0)"
  curl --fail --location --retry 3 -o "$ARCHIVE" "$URL"
  temp="$(mktemp -d)"
  trap 'rm -rf "$temp"' EXIT
  unzip -q "$ARCHIVE" -d "$temp"
  cp "$temp/GrailQA_v1.0"/*.json "$TARGET/"
fi
echo "GrailQA ready: $TARGET/grailqa_v1.0_dev.json"
