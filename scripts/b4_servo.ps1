# botmind: ACT v4 in sim with the real-arm servo fit (sim/assets/servo_fit.yaml, Phase 8 §8), CPU inference only
# (user approved botmind, CPU only, 2026-10-27). Same 460 states as the v4 evaluation (stage 4 of b4_v4.ps1), so
# results pair with data\eval\v4_*. Log: logs\servo_steps.log
$root = "C:\Users\botmi\so101_stack_game"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$lr = "$root\.venv-lerobot\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"; $env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:CUDA_VISIBLE_DEVICES = ""
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\servo_steps.log" }
$ck = "$runs\act_v4_ft\checkpoints\040000\pretrained_model"
$common = @("--policy", $ck, "--states", "game", "--videos", "2", "--temporal-ensemble", "0.01", "--resize", "320x240",
            "--device", "cpu", "--servo", "sim\assets\servo_fit.yaml")
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
$jobs = @()
foreach ($s in 4001..4006) { $jobs += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $jobs += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
foreach ($s in 3001..3004) { $jobs += ,@("game_s$s", @("--n", "50", "--seed", "$s")) }
Log "start: $($jobs.Count) eval jobs, CPU"
$procs = @()
foreach ($j in $jobs) {
  $out = "data\eval\servo_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
    -RedirectStandardOutput "$root\logs\servo_eval_$($j[0]).log" -RedirectStandardError "$root\logs\servo_eval_$($j[0]).err"
  Start-Sleep 5
}
$procs | ForEach-Object { $_.WaitForExit() }
$miss = @(Get-ChildItem -Directory "data\eval\servo_*" | Where-Object { -not (Test-Path "$($_.FullName)\episodes.json") } | ForEach-Object { $_.Name })
if ($miss.Count) { Log ("runs without results: " + ($miss -join ", ")) }
Log "done"
