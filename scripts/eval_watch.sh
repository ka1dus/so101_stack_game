#!/bin/bash
# Phase 6: evaluate every new ACT checkpoint from the GPU PC in sim, on the Mac (detached; safe to re-run).
# Polls the PC every 10 min, copies new checkpoints/<step>/pretrained_model to data/checkpoints/act_v1/<step>/,
# runs sim_eval.py (60 episodes, nominal visuals, fixed seed) and appends the summary to notes/phase6_eval_log.md.
#   nohup scripts/eval_watch.sh > data/eval/eval_watch.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PC="${PC_HOST:-so101-gpu}"   # Tailscale name (works away from home); LAN: 192.168.0.107
K=(-i "$HOME/.ssh/so101_pc_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)   # keep-alives: a dead link (sleep, network change) fails in ~1 min instead of hanging
RUN='C:\Users\User\so101_runs\act_v1\checkpoints'
LOCAL=data/checkpoints/act_v1; mkdir -p "$LOCAL" data/eval
LOG=notes/phase6_eval_log.md; [ -f "$LOG" ] || printf "# Phase 6: ACT v1 checkpoint evaluations (sim, 60 episodes each, seed 777, nominal visuals)\n\n" > "$LOG"
N=${EVAL_N:-60}
while true; do
  steps=$(ssh "${K[@]}" user@"$PC" "dir /b /ad $RUN" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{6}$' | sort)
  for s in $steps; do
    [ -f "$LOCAL/$s/.evaluated" ] && continue
    [ "$s" = "last" ] && continue
    mkdir -p "$LOCAL/$s"
    if [ ! -f "$LOCAL/$s/pretrained_model/model.safetensors" ]; then
      echo "$(date '+%F %T') copying checkpoint $s"
      scp -q -r "${K[@]}" "user@$PC:C:/Users/User/so101_runs/act_v1/checkpoints/$s/pretrained_model" "$LOCAL/$s/" || { rm -rf "$LOCAL/$s/pretrained_model"; continue; }
    fi
    echo "$(date '+%F %T') evaluating checkpoint $s"
    if MUJOCO_GL=cgl HF_HUB_OFFLINE=1 ~/.venvs/so101_eval/bin/python scripts/sim_eval.py --policy "$LOCAL/$s/pretrained_model" \
        --n "$N" --videos 3 --out "data/eval/act_v1_$s" > "data/eval/act_v1_$s.log" 2>&1; then
      { echo "## step $s ($(date '+%F %H:%M'))"; echo '```'; cat "data/eval/act_v1_$s/summary.txt"; echo '```'; echo; } >> "$LOG"
      touch "$LOCAL/$s/.evaluated"
    else
      echo "$(date '+%F %T') eval of $s failed; see data/eval/act_v1_$s.log"
      { echo "## step $s: eval FAILED ($(date '+%F %H:%M')), see data/eval/act_v1_$s.log"; echo; } >> "$LOG"
      touch "$LOCAL/$s/.evaluated"
    fi
  done
  if ssh "${K[@]}" user@"$PC" "findstr /c:\"ACT training finished\" C:\\Users\\User\\so101_stack_game\\logs\\pipeline.log" >/dev/null 2>&1; then
    remaining=$(for s in $(ssh "${K[@]}" user@"$PC" "dir /b /ad $RUN" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{6}$'); do [ -f "$LOCAL/$s/.evaluated" ] || echo "$s"; done)
    [ -z "$remaining" ] && { echo "$(date '+%F %T') training finished and all checkpoints evaluated"; exit 0; }
  fi
  sleep 600
done
