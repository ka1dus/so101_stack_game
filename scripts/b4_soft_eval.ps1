# botmind: evaluate one checkpoint with the fitted servos on the 460 states of the v4 comparison (seeds 4001-4006,
# 5001-5002, 3001-3004; pairs with data\eval\v4_* and data\eval\servo_*). GPU, staggered starts.
#   powershell -File scripts\b4_soft_eval.ps1 <checkpoint pretrained_model dir> <label> [parallel jobs, default 12] [servo kp/kv scale, default 1]
param([string]$Ck, [string]$Label, [int]$Par = 12, [string]$Scale = "1.0")
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$lr = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\soft_steps.log" }
Log "eval $Label start ($Ck, servo scale $Scale)"
$common = @("--policy", $Ck, "--states", "game", "--videos", "0", "--temporal-ensemble", "0.01", "--resize", "320x240", "--servo", "sim\assets\servo_fit.yaml", "--servo-scale", $Scale)
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
$jobs = @()
foreach ($s in 4001..4006) { $jobs += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $jobs += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
foreach ($s in 3001..3004) { $jobs += ,@("game_s$s", @("--n", "50", "--seed", "$s")) }
$procs = @()
foreach ($j in $jobs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $Par) { Start-Sleep 20 }
  $out = "data\eval\$($Label)_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
    -RedirectStandardOutput "$root\logs\$($Label)_eval_$($j[0]).log" -RedirectStandardError "$root\logs\$($Label)_eval_$($j[0]).err"
  Start-Sleep 15
}
$procs | ForEach-Object { $_.WaitForExit() }
$dirs = @(Get-ChildItem -Directory "data\eval\$($Label)_*" | Where-Object { Test-Path "$($_.FullName)\episodes.json" } | ForEach-Object { $_.FullName })
& $lr scripts\confirm_report.py --out "data\eval\confirmation_$Label.md" $dirs *> "$root\logs\$($Label)_report.log"
Log "eval $Label done ($($dirs.Count)/12 runs; data\eval\confirmation_$Label.md)"
