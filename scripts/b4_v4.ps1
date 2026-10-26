# botmind: ACT v4 = v3 fine-tuned on episodes rendered with the trimmed backdrop (near end at ~16 cm, 2026-10-26;
# user approved the run the same day). Log: logs\v4_steps.log
#   1. 1,000 game-distribution episodes (as v2) + 500 targeted (heights 4-7, rule picks no3/no35, 70 % crooked >= 10 mm:
#      the bucket the trim hurt most), trimmed-board scene
#   2. export 320x240 in 8 parts; merge with the 8 v2 + 8 v3 parts -> so101_stack_sim_act_v4_half (5,500 episodes)
#   3. fine-tune from v3 60k for 40k steps
#   4. evaluate on the 460 states of v3 wave 1 (seeds 4001-4006, 5001-5002, 3001-3004): pairs with data\eval\bdtrim_*
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$sim = "$root\.venv-sim\Scripts\python.exe"; $lr = "$root\.venv-lerobot\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\v4_steps.log" }
function Stop-Run($m) { Log "FAILED: $m"; exit 1 }

Log "stage 1: generating 1,000 game + 500 targeted episodes (trimmed backdrop)"
& $sim scripts\run_generation.py --workers 16 --act 1000 --vla 0 --prefix bdgame --seed-base 50000 *> "$root\logs\v4_generate_game.log"
& $sim scripts\run_generation.py --workers 16 --act 500 --vla 0 --prefix bdtall --seed-base 51000 `
  --height-min 4 --hard-p 1.0 --want-types no3,no35 --min-crooked-mm 10 --crooked-share 0.7 *> "$root\logs\v4_generate_tall.log"
$n = (Get-ChildItem -Directory "$root\data\raw\act_bdgame_w*\ep_*", "$root\data\raw\act_bdtall_w*\ep_*").Count
if ($n -lt 1500) { Stop-Run "only $n episodes" }
Log "stage 1 done: $n episodes"

Log "stage 2: export + merge"
$raw = @(Get-ChildItem -Directory "$root\data\raw\act_bdgame_w*", "$root\data\raw\act_bdtall_w*" | Sort-Object Name | ForEach-Object { $_.FullName })
$procs = @()
foreach ($k in 0..7) {
  $grp = @($raw | Where-Object { [array]::IndexOf($raw, $_) % 8 -eq $k }); $out = "$ds\v4_half_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $a = @("scripts\export_lerobot.py", "--raw") + $grp + @("--dataset", "act", "--repo-id", "local/v4_half_part$k", "--root", $out, "--streaming", "--queue", "600", "--resize", "320x240")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\v4_export_$k.log" -RedirectStandardError "$root\logs\v4_export_$k.err"
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($k in 0..7) { if (-not (Select-String -Quiet -Path "$root\logs\v4_export_$k.log" -Pattern "video check: every video")) { Stop-Run "export part $k failed" } }
Remove-Item -Recurse -Force "$ds\so101_stack_sim_act_v4_half" -ErrorAction SilentlyContinue
$parts = @(0..7 | ForEach-Object { "$ds\v2_half_part$_" }) + @(0..7 | ForEach-Object { "$ds\v3_half_part$_" }) + @(0..7 | ForEach-Object { "$ds\v4_half_part$_" })
& $lr scripts\merge_lerobot.py --parts $parts --repo-id Maxctl352/so101_stack_sim_act_v4 --out "$ds\so101_stack_sim_act_v4_half" *> "$root\logs\v4_merge.log"
if ($LASTEXITCODE -ne 0) { Stop-Run "merge failed" }
Log ("stage 2 done: " + ((Get-Content "$root\logs\v4_merge.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / "))

Log "stage 3: fine-tuning 40k steps from v3 60k"
Remove-Item -Recurse -Force "$runs\act_v4_ft" -ErrorAction SilentlyContinue
& "$root\.venv-lerobot\Scripts\lerobot-train.exe" --policy.path="$runs\act_v3_ft\checkpoints\060000\pretrained_model" `
  --dataset.repo_id=Maxctl352/so101_stack_sim_act_v4 --dataset.root="$ds\so101_stack_sim_act_v4_half" `
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_v4_ft" --job_name=act_v4_ft `
  --batch_size=8 --steps=40000 --save_freq=20000 --log_freq=500 --num_workers=4 --wandb.enable=false *> "$root\logs\v4_train.log"
if (-not (Test-Path "$runs\act_v4_ft\checkpoints\040000\pretrained_model\model.safetensors")) { Stop-Run "fine-tuning did not reach 40k (logs\v4_train.log)" }
Log "stage 3 done"

Log "stage 4: evaluation"
$ck = "$runs\act_v4_ft\checkpoints\040000\pretrained_model"
$common = @("--policy", $ck, "--states", "game", "--videos", "0", "--temporal-ensemble", "0.01", "--resize", "320x240")
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
$jobs = @()
foreach ($s in 4001..4006) { $jobs += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $jobs += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
foreach ($s in 3001..3004) { $jobs += ,@("game_s$s", @("--n", "50", "--seed", "$s")) }
$procs = @()
foreach ($j in $jobs) {
  $out = "data\eval\v4_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
    -RedirectStandardOutput "$root\logs\v4_eval_$($j[0]).log" -RedirectStandardError "$root\logs\v4_eval_$($j[0]).err"
  Start-Sleep 15                                    # staggered: 20 simultaneous starts ran out of GPU memory before
}
$procs | ForEach-Object { $_.WaitForExit() }
$dirs = @(Get-ChildItem -Directory "data\eval\v4_*" | Where-Object { Test-Path "$($_.FullName)\episodes.json" } | ForEach-Object { $_.FullName })
& $lr scripts\confirm_report.py --out "data\eval\confirmation_v4.md" $dirs *> "$root\logs\v4_report.log"
$miss = @(Get-ChildItem -Directory "data\eval\v4_*" | Where-Object { -not (Test-Path "$($_.FullName)\episodes.json") } | ForEach-Object { $_.Name })
if ($miss.Count) { Log ("runs without results: " + ($miss -join ", ")) }
Log "done (data\eval\confirmation_v4.md)"
