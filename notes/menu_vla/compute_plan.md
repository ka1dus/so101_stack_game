# Compute plan and cost estimate (menu VLA)

Prices checked on 2026-10-30 at <https://www.runpod.io/pricing> (they change; re-check
before booking). Taxes not included.

## What runs where

| Job | Machine | Cost |
|---|---|---|
| Data generation (~30–60k episodes) | botmind (16 workers) + siix-pc (16 workers) + home pc | $0 (already owned) |
| Pilots, sim evaluation, targeted fine-tunes | botmind and siix-pc, in parallel | $0 |
| Main SmolVLA training (one run) | RunPod, 1 × A100 80 GB PCIe, Community Cloud, $1.19/h (or siix-pc if the speed test allows) | see below |

**siix-pc** (added 2026-11-02): Windows, NVIDIA RTX 4000 Ada 20 GB (130 W, 360 GB/s),
Intel Core Ultra 7 265K (20 cores), 64 GB RAM, 932 GB disk (~380 GB free). Expected training
speed about the same as botmind or a bit slower (half the memory bandwidth; full-rate
FP32-accumulate tensor maths, 20 GB allows larger batches); **not measured**. Before use: set up
the `lerobot` env like botmind, SSH/Tailscale, confirm headless MuJoCo rendering, and a
~500-step SmolVLA speed test on both machines.
| Dataset/checkpoint storage near the GPU | RunPod network volume, 150 GB, $0.07/GB/month | ~$10.50/month |

**No Linux CPU machine is needed.** botmind already generates ~2 s/episode on the box
tasks (expect ~3–4 s with objects), so 60k episodes take about 2–3 days, run in
parallel with other work. Renting CPUs would save at most ~1–2 days. The rentable
options are also poor fits: Hetzner's 48-core CCX63 costs €1.37/h, is listed as
unavailable, and has no GPU for MuJoCo rendering.

## GPU hours

Reference speed: the LeRobot docs give ~4–5 h per 20k SmolVLA steps at batch 64 on one
A100, i.e. about 70–90 samples/s (<https://huggingface.co/docs/lerobot/smolvla>).
The pilot on botmind measures our own speed before any booking.

| Run | Steps (batch 64) | Where | A100 hours |
|---|---|---|---|
| Main run | 60k | RunPod A100 | 13–15 |
| Setup, data copy, failed start buffer | — | RunPod A100 | 2–5 |
| Targeted fine-tune 1 | 20k | botmind or siix-pc (~10–18 h) | 0 |
| Targeted fine-tune 2 | 20k | botmind or siix-pc (~10–18 h) | 0 |
| **Total rented** | | | **~15–20** |

## Cost scenarios

| Scenario | GPU | Storage | USB hub | Total | Whole project |
|---|---|---|---|---|---|
| botmind only | $0 | $0 | ~$40 | **~$40** | ~5–7 weeks |
| botmind + siix-pc, no cloud | $0 | $0 | ~$40 | **~$40** | ~4–5 weeks |
| **Recommended**: both + A100 Community for the main run, 20 h | 20 × $1.19 = $23.80 | $10.50 | ~$40 | **~$75** | ~3–4 weeks |
| Safe: both + A100 Secure Cloud, 2× overrun (40 h) | 40 × $1.79 = $71.60 | $10.50 | ~$40 | **~$122** | ~3–4 weeks |

**Ask for $50 prepaid RunPod credit plus ~$40 for the hub** (was $150 before the second
PC). The prepaid credit is a hard cap: RunPod stops pods when the credit runs out, so
spending can't exceed it. The hub is bought only if a different Mac port doesn't fix
the camera resets.

Not included: Claude Code usage for parallel agents (depends on your plan) and the
optional language-model API for the planner (cents per task).

## Ways the cost stays low

1. Pilots and evaluation on botmind; only full-size runs go to the cloud.
2. Community Cloud (33 % cheaper than Secure); checkpoints every 2k steps so a lost
   machine costs minutes, not hours.
3. The pod stops itself at the end of the training script (`runpodctl stop pod $RUNPOD_POD_ID`).
4. 320×240 data (proven in the stacking game): smaller uploads, faster data loading.
5. Placement stays computed, so the VLA needs less precision → fewer steps.
6. Targeted fine-tunes (only failing actions) instead of retraining from scratch.
7. SmolVLA's default trains the action expert with the vision encoder frozen (to confirm in the installed config).

## How to rent (RunPod)

1. Sign up at <https://www.runpod.io> with the company card; **Billing → add $50 credit**
   (prepaid; no auto-top-up).
2. **Settings → SSH Public Keys**: paste the Mac's public key (`cat ~/.ssh/so101_pc_ed25519.pub`
   or a new key). Never put keys in the repo.
3. **Storage → New Network Volume**: 150 GB, in a datacenter that lists A100 80 GB availability.
4. **Pods → Deploy**: GPU *A100 80GB PCIe*, *Community Cloud*, template *RunPod PyTorch*
   (CUDA 12.x), attach the network volume at `/workspace`, container disk 50 GB,
   enable SSH (public IP / exposed TCP port).
5. Claude gives the exact setup commands: install LeRobot at the pinned commit + ffmpeg,
   copy the dataset (rsync over SSH from botmind or `runpodctl send`), start training in
   `tmux` with `--save_freq=2000`, end the script with `runpodctl stop pod $RUNPOD_POD_ID`.
6. After the run: copy the checkpoints to botmind, then **terminate** the pod (a stopped pod
   still bills its disk). Delete the network volume at the end of the project.

Alternative: Vast.ai (marketplace, usually cheaper, interruptible instances "50 %+ cheaper",
less predictable hosts). Check live prices in its console; use only with frequent checkpoints.
