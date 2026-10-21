# Phase 5: finishing the large generation on the Windows GPU machine

Status when the Mac run was stopped (2026-10-21): **326 ACT episodes done** on the Mac
(`data/raw/act_main_w0..3`, about 690 MB). Still to do: **1,674 ACT + 1,000 VLA**.

Where it runs:
- **Generation runs in native Windows** (PowerShell), not WSL. Offscreen OpenGL inside WSL is unreliable
  (CLAUDE.md working rule 3). On Windows, MuJoCo renders through the NVIDIA OpenGL driver.
- **Export runs in WSL** (`lerobot` env), which reads the Windows files under `/mnt/c/...`.

Speed: the job is **CPU-bound**. Physics and IK take about 60 % of the time and image processing about 30 %;
rendering is only ~5 %, so the RTX 3080 barely changes the speed. What matters is the number of CPU cores:
run about one worker per physical core, minus one. Rough time = 2,674 episodes × ~22 s ÷ workers
(7 workers ≈ 2.3 h, 11 ≈ 1.5 h). The real speed shows in the logs after a few minutes.

---

## 0. On the Mac (done, or still needed)
- The Mac run is stopped. Nothing else needs to run there.
- The code must be on GitHub: the commit that adds `scripts/run_generation.py` and this file has to be pushed first.

## 1. Install the tools (Windows, PowerShell, once)
```powershell
winget install Git.Git
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```
Close and reopen PowerShell, then check:
```powershell
git --version; py -3.12 --version; ffmpeg -version
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
nvidia-smi
```
Note `NumberOfCores` (physical cores); it decides `--workers` in step 5.

## 2. Get the project
```powershell
cd $HOME
git clone https://github.com/ka1dus/so101_stack_game.git
cd so101_stack_game
```
The repo is private: Git Credential Manager opens a browser to sign in to GitHub. If the folder already
exists, run `git pull` inside it instead.

## 3. Sim environment (same pins as the Mac)
```powershell
py -3.12 -m venv .venv-sim
.\.venv-sim\Scripts\python -m pip install -r scripts\requirements-sim.txt
```
If a pinned package fails to install on Windows, stop and send me the error. Don't change versions yourself.

## 4. Check before the long run (~5 min)
Run these in an **interactive desktop session**. Rendering needs a logged-in desktop, not an SSH or
headless session. Don't use `$env:MUJOCO_GL`; leave it unset.
```powershell
.\.venv-sim\Scripts\python scripts\expert.py --n 20
```
Expected: `success 20/20` (19/20 is acceptable: physics can differ slightly between platforms).
```powershell
.\.venv-sim\Scripts\python scripts\generate.py --n 2 --run smoke_win
.\.venv-sim\Scripts\python scripts\inspect_episodes.py smoke_win
```
Open `notes\img\phase5_smoke_win_frames.png`. The frames must **not be black**, and should look like
`notes\img\phase5_act_pilot_frames.png` (black arm, yellow/black tape areas, printed boxes, both cameras).
If they match, delete the test run:
```powershell
Remove-Item -Recurse data\raw\smoke_win
```
If the frames are black or the script errors, stop and send me the output.

## 5. Large run
Keep Windows awake while it runs: Settings → System → Power & battery → Screen and sleep → "Never" when plugged in.
Replace `7` with your physical core count minus 1:
```powershell
.\.venv-sim\Scripts\python scripts\run_generation.py --workers 7 --act 1674 --vla 1000 --prefix win --seed-base 3000
```
Leave the PowerShell window open. Progress (in a second PowerShell window, in the project folder):
```powershell
Get-ChildItem data\raw\logs\win_w*.log | ForEach-Object { Get-Content $_ -Tail 1 }
```
Each line shows `ep N/total`, failures so far, and seconds per episode. Failed attempts (mostly
`hit_backdrop`, ~5 %) are thrown away automatically; that's expected.

**Pause or stop:** press Ctrl+C in the run window (or restart the PC). Finished episodes stay on disk.
**Resume:** run **exactly the same command** again (same `--workers`, `--act`, `--vla`, `--prefix`,
`--seed-base`). Each run continues from its last finished episode; half-finished episodes are discarded.
A different `--workers` value would split the totals differently, so don't change it mid-way.

When all workers have finished, the window prints `exit codes: [0, 0, ...]`.

## 6. Check the result
```powershell
Get-ChildItem data\raw\logs\win_w*.log | ForEach-Object { Select-String -Path $_ -Pattern "^done" }
.\.venv-sim\Scripts\python scripts\inspect_episodes.py act_win_w0
.\.venv-sim\Scripts\python scripts\inspect_episodes.py vla_win_w0
```
Send me the printed summaries (or commit the `notes\img\phase5_*_win_*` figures and the text output).

## 7. Add the Mac's 326 episodes (optional but saves ~15 min of compute)
When the Mac is awake, copy its four folders `~/so101_stack_game/data/raw/act_main_w0` … `act_main_w3`
(~690 MB) into `so101_stack_game\data\raw\` on Windows, using a USB drive, OneDrive, or AirDrop to a phone and
then the PC. They are plain folders: copy them as they are.
If you skip this, also generate those 326 on Windows:
```powershell
.\.venv-sim\Scripts\python scripts\run_generation.py --workers 7 --act 326 --vla 0 --prefix win2 --seed-base 4000
```

## 8. Export to LeRobot datasets (WSL Ubuntu)
```bash
conda activate lerobot
cd ~/lerobot && git log --oneline -1      # the Mac test used ca69a206; tell me if it differs
python -c "import av; print(av.__version__)"
```
Copy the raw data into the Linux filesystem first (reading thousands of videos from /mnt/c is slow).
Replace `<WindowsUser>` with your Windows user name:
```bash
mkdir -p ~/so101_stack_game_data
cp -r /mnt/c/Users/<WindowsUser>/so101_stack_game/data/raw ~/so101_stack_game_data/
cp -r /mnt/c/Users/<WindowsUser>/so101_stack_game/scripts ~/so101_stack_game_data/
cd ~/so101_stack_game_data
```
Try 3 episodes first, locally, no upload:
```bash
python scripts/export_lerobot.py --raw raw/act_* --dataset act --repo-id Maxctl352/so101_stack_sim_act_test --root /tmp/export_test --limit 3
```
If that prints `wrote 3 episodes`, delete `/tmp/export_test`, log in once, and export both datasets:
```bash
rm -rf /tmp/export_test
hf auth login
python scripts/export_lerobot.py --raw raw/act_* --dataset act --repo-id Maxctl352/so101_stack_sim_act --push
python scripts/export_lerobot.py --raw raw/act_* raw/vla_* --dataset vla --repo-id Maxctl352/so101_stack_sim_vla --push
```
Notes:
- The export re-encodes every video in LeRobot's own format. It wasn't timed for the full set; run the
  first export and watch the episode counter. It can be left running overnight.
- If an export stops partway, delete its local copy before re-running, because `create` refuses an existing folder:
  `rm -rf ~/.cache/huggingface/lerobot/Maxctl352/so101_stack_sim_act` (or `_vla`).
- Datasets are pushed as **private**.
- Per-episode game metadata (picked type, stack composition, pairing) is written to
  `meta/so101_stack_episodes.json` inside each dataset, for the Phase 6 per-bucket evaluation.

Then Phase 6: ACT training in the same WSL env.
