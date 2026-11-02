# botmind: sim twin of the 28 Oct real stage-2 trials (and the stage-1 batch for reference): does wcal60 pick the
# expected box in sim? 4 runs per trial, CPU, 6 in parallel. Log logs\stage2_twin.log; per trial logs\twin_<session>_t<n>.log
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
$pol = "C:\Users\botmi\so101_runs\act_wcal\checkpoints\060000\pretrained_model"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\stage2_twin.log" }
$jobs = @()
foreach ($t in 1..7) { $jobs += ,@("20261028_201746", $t) }
foreach ($t in 1..5) { $jobs += ,@("20261028_203311", $t) }
foreach ($t in 1..10) { $jobs += ,@("20261028_195923", $t) }
Log "start: $($jobs.Count) trials x 4 runs"
$procs = @()
foreach ($j in $jobs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge 6) { Start-Sleep 10 }
  $name = "twin_$($j[0])_t$($j[1])"
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList @("scripts\phase8_sim_twin.py", "data\real_trials\$($j[0])", "$($j[1])", "--policy", $pol, "--runs", "4", "--device", "cpu") `
    -RedirectStandardOutput "$root\logs\$name.log" -RedirectStandardError "$root\logs\$name.err"
}
$procs | Wait-Process
Log "done"
