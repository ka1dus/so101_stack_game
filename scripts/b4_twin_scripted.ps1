# botmind: policy pick + scripted place on the sim twin of the 29 Oct stage-2 trials, 4 runs each, CPU, 5 in parallel.
# Log logs\tsp_summary.log; per trial logs\tsp_<session>_t<n>.log
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\tsp_summary.log" }
$jobs = @("20261029_111738:1", "20261029_114535:1", "20261029_114535:2", "20261029_114535:3")
foreach ($t in 1..6) { $jobs += "20261029_121913:$t" }
foreach ($t in 1..10) { $jobs += "20261029_125445:$t" }
Log "start: $($jobs.Count) trials x 4 runs"
$procs = @()
foreach ($j in $jobs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge 5) { Start-Sleep 10 }
  $name = "tsp_" + $j.Replace(":", "_t")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList @("scripts\phase8_twin_scripted.py", "cpu", $j, "--runs", "4") `
    -RedirectStandardOutput "$root\logs\$name.log" -RedirectStandardError "$root\logs\$name.err"
}
$procs | Wait-Process
Log "done"
