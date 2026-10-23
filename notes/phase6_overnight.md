# Overnight run 2026-10-22: dataset build + ACT training on the Windows GPU PC

## Morning summary (read this first)
- **Done:** the PC is set up from scratch (native Windows, no admin env). All 3,000 episodes were copied, and the
  **ACT dataset was built and verified** (2,000 episodes, 699,945 frames, images aligned with states).
  **ACT training is running** (100k steps, ETA ~11:15). The Mac evaluates each checkpoint in sim:
  results in `notes/phase6_eval_log.md`.
- **Went wrong and fixed:** the first export silently dropped video frames (LeRobot's streaming encoder drops
  frames when overloaded), which crashed training at step 4,220. That was lucky: otherwise it would have trained
  on misaligned images. Re-exported with a no-drop setting and a built-in frame check; pixel-verified.
- **First checkpoint (10k steps, ~0.1 epoch):** weak, as expected this early. Mostly missed grasps; it reaches the
  boxes and sometimes completes a full placement. Later checkpoints arrive every ~40 min.
- **Waiting on you:** (1) Tailscale sign-in for the PC and the Mac, plus one firewall command (section Tailscale);
  (2) `hf auth login` on the PC if you want the datasets on the Hub; (3) commit/push of tonight's code (not done).
- The VLA dataset export starts automatically after training (it was moved there to protect the PC's RAM).


Read this first in the morning. Latest status is at the bottom (**Log**).

## The PC (checked over SSH)
- `desktop-33pvcl5`, user `user`, LAN 192.168.0.107, Windows 11 Home
- **AMD Ryzen 5 5600X (6 cores), 16 GB RAM, NVIDIA RTX 3060 Ti 8 GB** (driver 616.64), 872 GB free.
  CLAUDE.md said "RTX 3080" and "WSL with a working lerobot env": **neither is true**. No WSL is installed.
- Everything runs in **native Windows** (no admin needed for the env):
  - Miniforge `C:\Users\User\miniforge3` (official NumFOCUS-signed installer), env `lerobot`:
    Python 3.12.15, git 2.56, ffmpeg 7.1.1, torch 2.11.0+cu128, torchvision 0.26.0, av 15.1.0, torchcodec 0.11.1 (CPU)
  - LeRobot `C:\Users\User\lerobot` at **ca69a206** (same as the Mac), `pip install -e ".[training]"`
  - project files `C:\Users\User\so101_stack_game` (scripts, sim, inputs/calibration, data/raw: 3,000 episodes)
- SSH from the Mac: key `~/.ssh/so101_pc_ed25519` (dedicated, for this PC only).

## Data
- Generation finished on the Mac: **2,000 ACT + 1,000 VLA** raw episodes (~5 % of attempts discarded,
  mostly `hit_backdrop`). All copied to the PC and counted (500 × 4 ACT, 250 × 4 VLA).
- Export speed: the LeRobot default (PNG frames + AV1) took 20 s/episode. `--streaming` (direct encoding, this
  LeRobot commit) takes 3.7 s/episode with identical read-back. 4 parallel parts, then `scripts/merge_lerobot.py`
  (LeRobot `merge_datasets`, tested on the Mac: order kept, frames identical to raw).
- Datasets (local, **not pushed**: no Hugging Face login on the PC):
  `C:\Users\User\so101_datasets\so101_stack_sim_act` (2,000 episodes) and `...\so101_stack_sim_vla` (3,000 episodes)

## Training (ACT)
- Smoke test on Windows: 30 steps OK, checkpoint written, 3.7 GB VRAM, 0.39 s/step (0.34 with AMP) while 4
  exports loaded the CPU.
- Run: `lerobot-train --policy.type=act --policy.use_amp=true --batch_size=8 --steps=100000 --save_freq=10000`
  → `C:\Users\User\so101_runs\act_v1`. LeRobot defaults otherwise (ACT chunk 100, ResNet-18, lr 1e-5).
  The time depends on the step time measured after the exports end (see Log).

## Self-driving pipeline on the PC
`scripts/pc_pipeline.ps1`, started detached (WMI, survives SSH disconnects):
wait for the ACT exports → merge ACT → start training → VLA exports (below-normal priority) → merge VLA → wait for training.
Logs in `C:\Users\User\so101_stack_game\logs\`: `pipeline.log` (stages), `export_*.log`, `merge_*.log`,
`train_act.log` / `train_act.err`.

## Tailscale (for remote access from work)
- Installed on the PC (official signed MSI, 1.104.2), service automatic, **unattended mode** (stays connected
  without anyone logged in at the PC), hostname `so101-gpu`. **Logged out: needs your sign-in.**
- **You, in the morning, before leaving:**
  1. Sign the PC in: open https://login.tailscale.com/a/a75fb720112ac (any device; your Tailscale account,
     create one if needed). If the link has expired: on the PC, click the Tailscale tray icon → Log in.
  2. On this Mac: install Tailscale (App Store or https://tailscale.com/download/mac), sign in with the
     **same** account, allow the VPN configuration when macOS asks.
  3. Let SSH in over Tailscale. The current firewall rule only allows Private networks, and the Tailscale
     adapter may not count as one. PowerShell **as Administrator** on the PC:
     ```powershell
     New-NetFirewallRule -Name OpenSSH-Tailscale -DisplayName "OpenSSH via Tailscale" -Direction Inbound -Protocol TCP -LocalPort 22 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
     ```
     (I didn't change firewall rules myself.)
  4. Tell me when it's done. I'll connect via `so101-gpu` (Tailscale name) instead of 192.168.0.107.
- Keep the PC on and awake (you set sleep to Never on AC).

## Not done / needs you
- Hugging Face push: `hf auth login` on the PC, then push both datasets (commands to follow).
- CLAUDE.md corrections (GPU model, no WSL): to record when you're back.

## Log
EOF
cd ~/so101_stack_game && echo "- $(date '+%H:%M') pipeline running on the PC: ACT exports ~110/500 per part at 01:58; training starts automatically after the merge." >> notes/phase6_overnight.md; echo ok- 02:05 `scripts/sim_eval.py` written and tested end to end on the Mac (eval venv `~/.venvs/so101_eval`; ~20 s/episode,
  so ~35 min per 100). Judge checked against the scripted expert: 10/12 agree. One "disagreement" is the expert's own
  backdrop flag (now a separate safety flag in the eval). The other is a **resting-stack creep in sim**: a top box slid
  10 mm in 12 s with nothing touching it. Rare (0/40 fresh states > 5 mm). The eval skips such states; noted in CLAUDE.md.
- 02:10 ACT exports ~198/500 per part (~6/min each); merge + training start expected ~03:00–03:10.
- 02:54 ACT export done: 4 × 500 exit 0; **merged 2,000 episodes / 699,945 frames**
  (`C:\Users\User\so101_datasets\so101_stack_sim_act`). ACT training started (pid 25224).
- 03:01 training: step 1k, **0.246 s/step** (GPU 95 %, 5.5 GB of 8 GB, 75 °C), loss 1.58 → ETA ~09:45 for 100k.
  VLA export running at below-normal priority (~37/750 per part; ETA ~04:45, then merge).
- Mac: `scripts/eval_watch.sh` (nohup) evaluates every new checkpoint (60 sim episodes, seed 777) and appends
  to `notes/phase6_eval_log.md`. The Mac stays awake until ~13:50 (`caffeinate -t 43200`), plugged in.
- 03:39 **Training crashed at step 4,220**: `IndexError: Invalid frame index=31365 ... must be less than 31365` (torchcodec).
  Root cause, found by checking every episode against its video file: the **streaming export had dropped frames**.
  LeRobot's streaming encoder drops frames when its 30-frame queue is full (`video_utils.py:1028-1061`, built for
  live recording), and 4 exports saturating the CPU filled it. 283 of 500 part-0 episodes were short: images
  misaligned with states, so **the first dataset was unusable**. The crash was lucky; misalignment alone would have
  trained silently. My Mac tests (one process) never dropped, so they didn't catch it.
- 03:43 Fix and restart: `--queue 600` (≥ the longest episode, 480 frames, so nothing can drop) plus a built-in
  check (`verify_videos`: every video file's frames = its episodes' lengths; export exits 2 and the merge refuses
  otherwise). The check was tested on a deliberately corrupted file (3 frames removed → flagged). Deleted the corrupted
  datasets and run (old logs in `logs\run1_corrupt`), restarted the pipeline: ACT export → verified merge → training.
  The VLA export now runs **after** training, because 4 exports with large queues + training could exhaust 16 GB RAM.
  New ETA: training start ~05:00, finish ~11:45.
- 04:56 Re-export done: **all 4 parts pass the video check, 0 dropped-frame warnings**; merged 2,000 episodes /
  699,945 frames, merged dataset also passes. Pixel check against the raw recordings (episodes 0, 1234, 1999; both
  cameras; start, middle, end): dataset frame k matches raw frame k (mean diff 0.8–1.9 out of 255) and not k±1
  (up to 22 while the arm moves). **Images are aligned with states.**
- 04:56 ACT training restarted (`act_v1`): GPU 98 %, 5.5 GB, ~0.26–0.28 s/step → ETA ~12:00.
  First checkpoint (10k) ~05:40; the Mac evaluates it automatically (`notes/phase6_eval_log.md`).
- 06:05 **Checkpoint 10k eval (60 sim episodes): 2/60 success (3 %)**: 45 missed grasp, 7 drop, 5 misplaced,
  1 knocked; 7 wrong box; 0 backdrop touches. It reaches the supply but often doesn't close on a box, then goes to
  the stack empty. Expected at ~0.1 epoch. Later checkpoints are logged automatically in `notes/phase6_eval_log.md`.
  Watchers still running: training watchdog (my session) and `eval_watch.sh` (Mac, nohup).
- 09:40 **Tailscale working**: PC `so101-gpu` (100.116.230.30), Mac 100.72.94.106; SSH by name works (direct LAN
  path at home, relayed elsewhere). The first PC sign-in hadn't stuck (my login command timed out); redone with a
  waiting login. The extra firewall rule isn't needed (the Tailscale network is Private on the PC). The eval watcher
  and helpers now use `so101-gpu`.
- 10:38 After the commute: the Mac's sleep killed a checkpoint copy mid-transfer (scp hung at 15 MB). Added SSH keep-alives to `eval_watch.sh` (dead link fails in ~1 min) and restarted it; 80k copied in 35 s over Tailscale (direct path). Training at 90k, ETA ~11:15.
- 11:13 **ACT training finished**: 100k steps in 6 h 17 min (1.14 epochs), final loss 0.048. The PC pipeline
  started the VLA export (4 parts, verified) automatically.
- 11:16 Re-planning screen on the 100k checkpoint (`scripts/screen_replanning.sh`, 60 episodes each, seed 777,
  all videos saved): trained default (re-plan every 100 steps), every 25, every 10, temporal ensembling 0.01.
- 11:45 Checkpoints 80k: 45 %, 90k: 38 % (default inference): plateau ~40 %.
- 11:45 **Missed-grasp review** (100k, default inference, episodes 2-5; frame sheets of both cameras):
  1. aim error ~1–2 cm (half a box width), not millimetres: descends into the gap between boxes or straddles two
     packed boxes. A wider opening alone won't fix it.
  2. the descent is planned from high above and runs open loop (100-step chunks); the wrist camera shows the
     misalignment near the box, but the policy doesn't re-plan then → the motivation for k10/temporal ensembling.
  3. no retry: after closing on air it carries nothing to the stack and "places" (no failed grasps in the demos).
  4. sparse supplies (1–3 boxes left in tall-stack states): the wrist view is nearly all black mat; the box must be
     found from the front camera (~50 px wide at 640×480; ResNet-18's last feature map has a ~32 px cell ≈ 1.5–2 cm).
- 12:05 **Re-planning screen (100k, 60 states, seed 777):** every 100 steps (default) 25/60 = 41.7 %; every 25
  steps 22/60 = 36.7 %; every 10 steps stopped at 1/17 (jerky switching between plans without averaging);
  temporal ensembling 0.01: 16/28 = 57 % at that point (still running). Temporal ensembling chosen.
- 12:05 200-episode run, temporal ensembling, **new** states (seeds 1001–1004, 4 × 50), so the setting isn't judged
  on the states used to choose it. ~200 s/episode per process → ETA ~14:30–15:00. Mac kept awake to ~18:00.
- 12:35 Battery guard (`scripts/battery_guard.sh`, nohup): freezes all sim evaluations below 10 % battery and resumes at ≥ 80 % (user request). Pause/resume tested on one run. Log: `data/eval/battery_guard.log`. Pausing pushes the 200-episode ETA back by the charging time.
- 13:25 Screen result, temporal ensembling (60 states, seed 777): **34/60 = 56.7 %** vs 41.7 % default (22 missed grasps vs 29); 0-3: 43 %, 4+: 64 %; 0 backdrop touches. 200-episode run at 56/116 (48 %) so far; battery steady at 26 % (charging).
- 14:10 **200-episode result (temporal ensembling, new states): 98/200 = 49.0 %**. Phase 6 report: `notes/phase6.md`.
