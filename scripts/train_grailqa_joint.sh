#!/usr/bin/env bash
# Provision the joint v14 LoRA fine-tune (WebQSP + CWQ + GrailQA) on the ROCm
# training host. Run this ON the training server as root; it is idempotent.
#
#   scripts/train_grailqa_joint.sh setup    # container + LlamaFactory + config
#   scripts/train_grailqa_joint.sh cron     # install the GPU time-sharing cron
#   scripts/train_grailqa_joint.sh start    # launch/resume now
#
# Input: output/memq_finetune_data_v14.json (built by scripts/build_grailqa_joint.sh),
# uploaded to $DATA_DIR on the server.
#
# --- Hard-won ROCm facts (gfx1201 / Radeon AI PRO R9700, ROCm 7.2.3) ----------
#   * flash_attn MUST be sdpa. LlamaFactory's `disabled` (eager) attention
#     triggers an amdgpu page fault (GCVM_L2_PROTECTION_FAULT, TCP client) in
#     the first backward pass. The visible symptom is a *downstream* message,
#     "rocBLAS error: Could not initialize Tensile host: No devices found",
#     because the HIP context is already dead by then - check `dmesg -T` for the
#     real cause instead of trusting that message.
#   * TORCH_BLAS_PREFER_HIPBLASLT=0 - hipBLASLt cannot handle the narrow
#     LoRA-rank GEMMs (n=16) on RDNA4 and corrupts the context.
#   * Do NOT set PYTORCH_HIP_ALLOC_CONF=expandable_segments (page fault during
#     shard load) and do NOT set HSA_OVERRIDE_GFX_VERSION (irrelevant here).
#   * peft must be pinned to 0.18.1, and `pip install -e .` must never replace
#     the ROCm torch build with a CUDA one.
#
# --- GPU time-sharing --------------------------------------------------------
# The GPU is shared with the energy-arena chronos forecast workers, which own
# 07:00-13:59 local (SUBMIT_HOURS_UTC=5-11 in their compose env; the daily
# submission deadline is 12:00). Training owns 14:00-07:00. `pause` kills the
# trainer rather than idling it, because only a dead process releases the ROCm
# context - verified: 15.3 GB -> 74 MB VRAM. LlamaFactory resumes from the
# newest checkpoint, so a pause costs at most save_steps steps.
set -euo pipefail

CONTAINER="${MEMQ_CONTAINER:-llamafactory}"
IMAGE="${MEMQ_IMAGE:-chronos-rocm:latest}"
DATA_DIR="${MEMQ_DATA_DIR:-/mnt/data/appdata_slow/llamafactory/data}"
HF_DIR="${MEMQ_HF_DIR:-/mnt/data/appdata_slow/llamafactory/hf}"
# Checkpoints go to the NVMe SSD, never to the slow ZFS data pool.
SSD_DIR="${MEMQ_SSD_DIR:-/mnt/cache/memq-v14/models}"
CTL="${MEMQ_CTL:-/mnt/data/appdata_slow/llamafactory/memq-v14.sh}"
GPU_LOCK_VOLUME="${MEMQ_GPU_LOCK_VOLUME:-energy-arena-chronos_gpu-lock}"

case "${1:-}" in

setup)
  mkdir -p "$SSD_DIR"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "creating container $CONTAINER"
    docker run -d --name "$CONTAINER" --restart unless-stopped \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --shm-size 8g --ipc host \
      -v "$DATA_DIR:/root/data" \
      -v "$SSD_DIR:/root/models_ssd" \
      -v "$HF_DIR:/root/.cache/huggingface" \
      -v "$GPU_LOCK_VOLUME:/locks" \
      -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      -e DISABLE_VERSION_CHECK=1 -e AMD_SERIALIZE_KERNEL=3 \
      -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
      "$IMAGE" sleep infinity
  else
    echo "container $CONTAINER already exists"
    docker start "$CONTAINER" >/dev/null 2>&1 || true
  fi
  docker exec "$CONTAINER" bash /root/data/setup_llamafactory_v14.sh
  ;;

cron)
  # Idempotent: drop any previous MemQ block, then append the current one.
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v 'memq-v14.sh' > "$tmp" || true
  cat >> "$tmp" <<EOF

# --- MemQ v14 joint training (shares the AMD GPU with energy-arena chronos) ---
# chronos: 05-11 UTC = 07:00-13:59 local. MemQ: 14:00-07:00 local.
# start is idempotent and doubles as crash recovery (resumes from checkpoint).
*/15 14-23,0-5 * * * $CTL start
55 6 * * * $CTL pause
*/10 7-13 * * * $CTL pause
EOF
  crontab "$tmp"
  rm -f "$tmp"
  crontab -l | tail -8
  ;;

start)  "$CTL" start ;;
pause)  "$CTL" pause ;;
status) "$CTL" status ;;

*)
  echo "usage: $0 {setup|cron|start|pause|status}" >&2
  exit 2 ;;
esac
