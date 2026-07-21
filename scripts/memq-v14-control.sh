#!/bin/bash
# MemQ v14 (joint WebQSP+CWQ+GrailQA) LoRA training control.
#
# The GPU is shared with the energy-arena chronos forecast workers. Those run
# only 05-11 UTC (07:00-13:59 CEST, SUBMIT_HOURS_UTC in their compose env); the
# rest of the day belongs to this training run.
#
#   memq-v14.sh start   launch or resume training (no-op if already running)
#   memq-v14.sh pause   kill the trainer and verify the VRAM is actually free
#   memq-v14.sh status  show run state, last log lines and VRAM
#
# `pause` terminates the process rather than idling it: only a dead process
# releases the ROCm context, so chronos gets a completely free GPU. LlamaFactory
# resumes from the newest checkpoint in output_dir on the next `start`
# (parser.py get_last_checkpoint), so at most save_steps steps are repeated.
set -u

CONTAINER=llamafactory
PATTERN='llamafactory-cli train'
CONFIG=/root/data/memq_lora_train_v14.yaml
LOG=/root/models_ssd/train_v14.log
HOSTLOG=/mnt/cache/memq-v14/models/train_v14.log
CTLLOG=/mnt/cache/memq-v14/memq-v14-control.log

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$CTLLOG"; }

running() { docker exec "$CONTAINER" pgrep -f "$PATTERN" >/dev/null 2>&1; }

vram() {
  docker exec "$CONTAINER" rocm-smi --showmeminfo vram 2>/dev/null \
    | grep -i "used memory" | head -1 | tr -s ' '
}

case "${1:-status}" in

start)
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "start: container $CONTAINER is not running -> abort"; exit 1
  fi
  # Runs every 15 min inside the training window, so a running trainer is the
  # normal case and must stay quiet; it doubles as crash recovery.
  if running; then exit 0; fi
  # Never take the GPU while a chronos forecast holds the shared lock.
  if docker exec "$CONTAINER" flock -n /locks/gpu.lock true 2>/dev/null; then
    log "start: chronos GPU lock is free"
  else
    log "start: chronos holds the GPU lock -> retry at the next cron tick"; exit 0
  fi
  log "start: launching v14 training (resumes from newest checkpoint if present)"
  docker exec -d -w /root/LlamaFactory \
    -e DISABLE_VERSION_CHECK=1 \
    -e AMD_SERIALIZE_KERNEL=3 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_HOME=/root/.cache/huggingface \
    "$CONTAINER" \
    bash -c "llamafactory-cli train $CONFIG >> $LOG 2>&1"
  sleep 20
  if running; then log "start: OK, trainer is up"; else
    log "start: FAILED to come up, last log lines:"; tail -20 "$HOSTLOG" | tee -a "$CTLLOG"; exit 1
  fi
  ;;

pause)
  # Silent no-op: this also runs as a guard every 10 min during the chronos
  # window, so it must not spam the log when there is nothing to stop.
  if ! running; then exit 0; fi
  log "pause: VRAM before: $(vram)"
  docker exec "$CONTAINER" pkill -TERM -f "$PATTERN"
  for _ in $(seq 1 60); do running || break; sleep 2; done
  if running; then
    log "pause: still alive after 120s -> SIGKILL"
    docker exec "$CONTAINER" pkill -9 -f "$PATTERN"
    sleep 5
  fi
  # Any surviving process with a ROCm context would keep VRAM allocated.
  for _ in $(seq 1 15); do
    docker exec "$CONTAINER" pgrep -f "$PATTERN" >/dev/null 2>&1 || break
    sleep 2
  done
  docker exec "$CONTAINER" pkill -9 -f "torchrun|llamafactory" >/dev/null 2>&1
  sleep 3
  log "pause: VRAM after:  $(vram)"
  log "pause: remaining GPU processes: $(docker exec "$CONTAINER" rocm-smi --showpids 2>/dev/null | grep -ci 'PID' || echo 0) header-lines"
  log "pause: done, GPU released for chronos"
  ;;

status)
  if running; then echo "state: RUNNING"; else echo "state: stopped"; fi
  echo "vram : $(vram)"
  echo "--- last log ---"
  tail -8 "$HOSTLOG" 2>/dev/null || echo "(no log yet)"
  ;;

*)
  echo "usage: $0 {start|pause|status}" >&2; exit 2 ;;
esac
