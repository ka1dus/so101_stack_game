# botmind: targeted v3 round (user approved 2026-10-23): tall crooked stacks with large/medium boxes, using the fixed
# expert (backdrop-safe moves, outward rise near the base, release by the lowest corner). Log: logs\v3.log
#   1. 1,000 episodes: heights 5-7, hard stacks, rule picks no3/no35, 70 % crooked >= 10 mm (game-rule supplies)
#   2. export 320x240 in 8 parts; verified merge with the 8 v2 parts -> so101_stack_sim_act_v3_half (4,000 episodes)
#   3. fine-tune from v2 300k for 60k steps (LeRobot recomputes normalisation stats from the merged dataset)
#   4. evaluate: same targeted seeds as the confirmation (4001-4006, 5001-5002) + game set 3001-3004, then 3005-3010
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$sim = "$root\.venv-sim\Scripts\python.exe"; $lr = "$root\.venv-lerobot\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\v3.log" }
function Stop-Run($m) { Log "FAILED: $m"; exit 1 }

Log "stage 1: generating 1,000 targeted episodes"
& $sim scripts\run_generation.py --workers 16 --act 1000 --vla 0 --prefix tall --seed-base 40000 `
  --height-min 5 --hard-p 1.0 --want-types no3,no35 --min-crooked-mm 10 --crooked-share 0.7 *> "$root\logs\v3_generate.log"
$n = (Get-ChildItem -Directory "$root\data\raw\act_tall_w*\ep_*").Count
if ($n -lt 1000) { Stop-Run "only $n episodes" }
Log "stage 1 done: $n episodes"

Log "stage 2: export + merge"
$raw = Get-ChildItem -Directory "$root\data\raw\act_tall_w*" | Sort-Object Name | ForEach-Object { $_.FullName }
$procs = @()
foreach ($k in 0..7) {
  $grp = @($raw | Where-Object { [array]::IndexOf($raw, $_) % 8 -eq $k }); $out = "$ds\v3_half_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $a = @("scripts\export_lerobot.py", "--raw") + $grp + @("--dataset", "act", "--repo-id", "local/v3_half_part$k", "--root", $out, "--streaming", "--queue", "600", "--resize", "320x240")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\v3_export_$k.log" -RedirectStandardError "$root\logs\v3_export_$k.err"
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($k in 0..7) { if (-not (Select-String -Quiet -Path "$root\logs\v3_export_$k.log" -Pattern "video check: every video")) { Stop-Run "export part $k failed" } }
Remove-Item -Recurse -Force "$ds\so101_stack_sim_act_v3_half" -ErrorAction SilentlyContinue
$parts = @(0..7 | ForEach-Object { "$ds\v2_half_part$_" }) + @(0..7 | ForEach-Object { "$ds\v3_half_part$_" })
& $lr scripts\merge_lerobot.py --parts $parts --repo-id Maxctl352/so101_stack_sim_act_v3 --out "$ds\so101_stack_sim_act_v3_half" *> "$root\logs\v3_merge.log"
if ($LASTEXITCODE -ne 0) { Stop-Run "merge failed" }
Log ("stage 2 done: " + ((Get-Content "$root\logs\v3_merge.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / "))

Log "stage 3: fine-tuning 60k steps from v2 300k"
Remove-Item -Recurse -Force "$runs\act_v3_ft" -ErrorAction SilentlyContinue
& "$root\.venv-lerobot\Scripts\lerobot-train.exe" --policy.path="$runs\act_v2_half\checkpoints\300000\pretrained_model" `
  --dataset.repo_id=Maxctl352/so101_stack_sim_act_v3 --dataset.root="$ds\so101_stack_sim_act_v3_half" `
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_v3_ft" --job_name=act_v3_ft `
  --batch_size=8 --steps=60000 --save_freq=20000 --log_freq=500 --num_workers=4 --wandb.enable=false *> "$root\logs\v3_train.log"
if (-not (Test-Path "$runs\act_v3_ft\checkpoints\060000\pretrained_model\model.safetensors")) { Stop-Run "fine-tuning did not reach 60k (logs\v3_train.log)" }
Log "stage 3 done"

Log "stage 4: evaluation"
$ck = "$runs\act_v3_ft\checkpoints\060000\pretrained_model"
$common = @("--policy", $ck, "--states", "game", "--videos", "0", "--temporal-ensemble", "0.01", "--resize", "320x240")
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
$wave1 = @()
foreach ($s in 4001..4006) { $wave1 += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $wave1 += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
foreach ($s in 3001..3004) { $wave1 += ,@("game_s$s", @("--n", "50", "--seed", "$s")) }
$wave2 = @(); foreach ($s in 3005..3010) { $wave2 += ,@("A_game_s$s", @("--n", "50", "--seed", "$s")) }
foreach ($wave in @($wave1, $wave2)) {
  $procs = @()
  foreach ($j in $wave) {
    $out = "data\eval\v3_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
    $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
      -RedirectStandardOutput "$root\logs\v3_eval_$($j[0]).log" -RedirectStandardError "$root\logs\v3_eval_$($j[0]).err"
    Start-Sleep 15                                  # staggered: 20 simultaneous starts ran out of GPU memory before
  }
  $procs | ForEach-Object { $_.WaitForExit() }
  Log "evaluation wave done"
}
$dirs = @(Get-ChildItem -Directory "data\eval\v3_*" | Where-Object { Test-Path "$($_.FullName)\episodes.json" } | ForEach-Object { $_.FullName })
& $lr scripts\confirm_report.py --out "data\eval\confirmation_v3.md" $dirs *> "$root\logs\v3_report.log"
$miss = @(Get-ChildItem -Directory "data\eval\v3_*" | Where-Object { -not (Test-Path "$($_.FullName)\episodes.json") } | ForEach-Object { $_.Name })
if ($miss.Count) { Log ("runs without results: " + ($miss -join ", ")) }
Log "done (data\eval\confirmation_v3.md)"
