# botmind: ACT v3 (60k fine-tune) re-evaluated with the trimmed backdrop (near end moved from 10 to ~16 cm along the
# right edge, 2026-10-26). Same seeds as the v3 evaluation wave 1, so the results pair with data\eval\v3_*:
#   crooked height 4-7 per type (4001-4006), large onto small (5001-5002), game set (3001-3004). Log: logs\bdtrim_steps.log
$root = "C:\Users\botmi\so101_stack_game"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$lr = "$root\.venv-lerobot\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\bdtrim_steps.log" }
$ck = "$runs\act_v3_ft\checkpoints\060000\pretrained_model"
$common = @("--policy", $ck, "--states", "game", "--videos", "0", "--temporal-ensemble", "0.01", "--resize", "320x240")
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
$jobs = @()
foreach ($s in 4001..4006) { $jobs += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $jobs += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
foreach ($s in 3001..3004) { $jobs += ,@("game_s$s", @("--n", "50", "--seed", "$s")) }
Log "started"
$procs = @()
foreach ($j in $jobs) {
  $out = "data\eval\bdtrim_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
    -RedirectStandardOutput "$root\logs\bdtrim_$($j[0]).log" -RedirectStandardError "$root\logs\bdtrim_$($j[0]).err"
  Start-Sleep 15                                    # staggered: 20 simultaneous starts ran out of GPU memory before
}
$procs | ForEach-Object { $_.WaitForExit() }
$dirs = @(Get-ChildItem -Directory "data\eval\bdtrim_*" | Where-Object { Test-Path "$($_.FullName)\episodes.json" } | ForEach-Object { $_.FullName })
& $lr scripts\confirm_report.py --out "data\eval\confirmation_bdtrim.md" $dirs *> "$root\logs\bdtrim_report.log"
$miss = @(Get-ChildItem -Directory "data\eval\bdtrim_*" | Where-Object { -not (Test-Path "$($_.FullName)\episodes.json") } | ForEach-Object { $_.Name })
if ($miss.Count) { Log ("runs without results: " + ($miss -join ", ")) }
Log "done (data\eval\confirmation_bdtrim.md)"
