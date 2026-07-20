#!/usr/bin/env bash
# Download public data/artifacts needed for the documented MemQ reproduction.
set -euo pipefail

BASE_URL="https://f003.backblazeb2.com/file/memq-finetunings/reproduction/v1"
DOWNLOAD_RAW=0
if [[ "${1:-}" == "--raw" ]]; then
  DOWNLOAD_RAW=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--raw]" >&2
  exit 2
fi

if command -v curl >/dev/null 2>&1; then
  downloader() { curl --fail --location --retry 5 --retry-delay 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  downloader() { wget --tries=5 -O "$2" "$1"; }
else
  echo "Install curl or wget before running this script." >&2
  exit 1
fi

download() {
  local remote="$1" local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  if [[ -s "$local_path" ]]; then
    echo "Already present: $local_path"
    return
  fi
  echo "Downloading: $remote"
  downloader "$BASE_URL/$remote" "$local_path"
}

# Enough to reproduce the public, database-free lookup pass from the supplied
# v9 plans.  The raw datasets are optional because their original licenses and
# terms remain applicable.
download artifacts/output/key_explain.json output/key_explain.json
download artifacts/output/All_cached_mid_names.json output/All_cached_mid_names.json
download artifacts/output/webqsp_test_prompt.json output/webqsp_test_prompt.json
download artifacts/output/cwq_test_prompt.json output/cwq_test_prompt.json
download artifacts/output/webqsp_test_plan_v10.json output/webqsp_test_plan_v10.json
download artifacts/output/cwq_test_plan_v10.json output/cwq_test_plan_v10.json
download artifacts/output/webqsp_metrics_v9_dirfb.json output/webqsp_metrics_v9_dirfb.json
download artifacts/output/cwq_metrics_v9_dirfb.json output/cwq_metrics_v9_dirfb.json

if [[ "$DOWNLOAD_RAW" == 1 ]]; then
  download datasets/webqsp/WebQSP.train.json data/webqsp/WebQSP.train.json
  download datasets/webqsp/WebQSP.test.json data/webqsp/WebQSP.test.json
  download datasets/webqsp/WebQSP.train.partial.json data/webqsp/WebQSP.train.partial.json
  download datasets/webqsp/WebQSP.test.partial.json data/webqsp/WebQSP.test.partial.json
  download datasets/cwq/ComplexWebQuestions_train.json data/cwq/ComplexWebQuestions_train.json
  download datasets/cwq/ComplexWebQuestions_dev.json data/cwq/ComplexWebQuestions_dev.json
  download datasets/cwq/ComplexWebQuestions_test.json data/cwq/ComplexWebQuestions_test.json
fi

echo "Reproduction artifacts are ready."
