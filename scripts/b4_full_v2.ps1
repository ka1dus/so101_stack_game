# botmind: full v2 run (user approved 2026-10-23). Self-driving; each stage logs to logs\full_v2.log and the run
# stops at the first failed stage. Re-running resumes generation (existing ep_* are kept) but redoes later stages.
#   1. generate 3,000 v2 episodes (game-rule states, kind mix 45/40/15), 16 workers
#   2. export at 320x240 in 8 parallel parts (streaming, queue 600, video check), verified merge
#   3. ACT training 300k steps (AMP, batch 8), checkpoint every 25k
#   4. evaluate the final checkpoint on the v1 200-state set (seeds 1001-1004) and the game-rule set (3001-3004)
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; $runs = "C:\Users\botmi\so101_runs"
cd $root
$sim = "$root\.venv-sim\Scripts\python.exe"; $lr = "$root\.venv-lerobot\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\full_v2.log" }
function Stop-Run($m) { Log "FAILED: $m"; exit 1 }

# ---- 1. generation
Log "stage 1: generating 3,000 episodes"
& $sim scripts\run_generation.py --workers 16 --act 3000 --vla 0 --prefix full --seed-base 30000 *> "$root\logs\full_v2_generate.log"
$n = (Get-ChildItem -Directory "$root\data\raw\act_full_w*\ep_*").Count
if ($n -lt 3000) { Stop-Run "only $n episodes generated" }
Log "stage 1 done: $n episodes"

# ---- 2. export (8 parts at 320x240) + verified merge
Log "stage 2: exporting at 320x240"
$raw = Get-ChildItem -Directory "$root\data\raw\act_full_w*" | Sort-Object Name | ForEach-Object { $_.FullName }
$procs = @()
foreach ($k in 0..7) {
  $grp = @($raw | Where-Object { [array]::IndexOf($raw, $_) % 8 -eq $k })
  $out = "$ds\v2_half_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $a = @("scripts\export_lerobot.py", "--raw") + $grp + @("--dataset", "act", "--repo-id", "local/v2_half_part$k", "--root", $out,
       "--streaming", "--queue", "600", "--resize", "320x240")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\full_v2_export_$k.log" -RedirectStandardError "$root\logs\full_v2_export_$k.err"
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($k in 0..7) { if (-not (Select-String -Quiet -Path "$root\logs\full_v2_export_$k.log" -Pattern "video check: every video")) { Stop-Run "export part $k failed its video check" } }
Remove-Item -Recurse -Force "$ds\so101_stack_sim_act_v2_half" -ErrorAction SilentlyContinue
$parts = 0..7 | ForEach-Object { "$ds\v2_half_part$_" }
& $lr scripts\merge_lerobot.py --parts $parts --repo-id Maxctl352/so101_stack_sim_act_v2 --out "$ds\so101_stack_sim_act_v2_half" *> "$root\logs\full_v2_merge.log"
if ($LASTEXITCODE -ne 0) { Stop-Run "merge failed" }
Log ("stage 2 done: " + ((Get-Content "$root\logs\full_v2_merge.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / "))

# ---- 3. training
Log "stage 3: training 300k steps"
Remove-Item -Recurse -Force "$runs\act_v2_half" -ErrorAction SilentlyContinue
& "$root\.venv-lerobot\Scripts\lerobot-train.exe" --dataset.repo_id=Maxctl352/so101_stack_sim_act_v2 --dataset.root="$ds\so101_stack_sim_act_v2_half" `
  --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_v2_half" `
  --job_name=act_v2_half --batch_size=8 --steps=300000 --save_freq=25000 --log_freq=500 --num_workers=4 --wandb.enable=false *> "$root\logs\full_v2_train.log"
if (-not (Test-Path "$runs\act_v2_half\checkpoints\300000\pretrained_model\model.safetensors")) { Stop-Run "training did not reach 300k (see logs\full_v2_train.log)" }
Log "stage 3 done"

# ---- 4. evaluation on both 200-state sets
Log "stage 4: evaluating on both test sets"
$ck = "$runs\act_v2_half\checkpoints\300000\pretrained_model"
$procs = @()
foreach ($set in @(@{n = "v1"; s = 1001}, @{n = "game"; s = 3001})) {
  foreach ($k in 0..3) {
    $seed = $set.s + $k; $out = "data\eval\v2half_300k_$($set.n)_s$seed"
    $a = @("scripts\sim_eval.py", "--policy", $ck, "--n", "50", "--seed", "$seed", "--states", $set.n, "--videos", "2",
           "--temporal-ensemble", "0.01", "--resize", "320x240", "--out", $out)
    $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\full_v2_eval_$($set.n)_$seed.log" -RedirectStandardError "$root\logs\full_v2_eval_$($set.n)_$seed.err"
  }
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($n in "v1", "game") {
  $dirs = Get-ChildItem -Directory "data\eval\v2half_300k_${n}_s*" | ForEach-Object { $_.FullName }
  & $lr scripts\eval_aggregate.py --title "act_v2_half 300k, TE 0.01, $n test set" --out "data\eval\v2half_300k_$n.txt" --resize 320x240 --states $n --temporal-ensemble 0.01 $dirs *> "$root\logs\full_v2_aggregate_$n.log"
  Log ("stage 4 $n set: " + ((Get-Content "data\eval\v2half_300k_$n.txt" -ErrorAction SilentlyContinue | Select-String "^success") -join " "))
}
Log "done"
