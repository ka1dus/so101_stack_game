#!/bin/bash
# Pause the Mac's sim evaluations when the battery gets low, resume when it has recharged (user request 2026-10-22).
# SIGSTOP freezes the sim_eval.py processes (and their ffmpeg writers) in place; SIGCONT continues them exactly where
# they were (the sim doesn't depend on wall-clock time). Exits when no sim_eval.py is left.
#   nohup scripts/battery_guard.sh > data/eval/battery_guard.log 2>&1 &
LOW=${LOW:-10}; HIGH=${HIGH:-80}; paused=0
pct() { pmset -g batt | grep -oE '[0-9]+%' | head -1 | tr -d '%'; }
pids() { pgrep -f "sim_eval.py|ffmpeg -loglevel error -y -f rawvideo" | tr '\n' ' '; }
echo "$(date '+%F %T') guard started: pause below ${LOW}%, resume at ${HIGH}% (battery $(pct)%)"
while true; do
  p=$(pct); procs=$(pids)
  [ -z "$procs" ] && { echo "$(date '+%F %T') no evaluations left; guard exits (battery ${p}%)"; exit 0; }
  if [ "$paused" = 0 ] && [ "$p" -lt "$LOW" ]; then
    kill -STOP $procs; paused=1; echo "$(date '+%F %T') battery ${p}% < ${LOW}%: PAUSED $procs"
  elif [ "$paused" = 1 ] && [ "$p" -ge "$HIGH" ]; then
    kill -CONT $procs; paused=0; echo "$(date '+%F %T') battery ${p}% >= ${HIGH}%: RESUMED $procs"
  fi
  sleep 60
done
