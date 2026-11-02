# botmind: stage-3 setups (policy pick + scripted place, crooked 3-4 box stacks), 4 runs each, CPU, 4 in parallel.
# Log logs\s3_summary.log; per setup logs\s3_<n>.log
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\s3_summary.log" }
$specs = @("no3 no35 no35 > no3", "no35 no35 no3 > no35", "no3 no35 no10 > no3", "no35 no10 no35 > no3", "no3 no3 no35 > no35",
           "no3 no35 no35 no35 > no3", "no35 no3 no35 no35 > no3", "no3 no35 no35 no10 > no35")
Log "start: $($specs.Count) setups x 4 runs"
$procs = @(); $i = 0
foreach ($sp in $specs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge 4) { Start-Sleep 10 }
  $i++
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList @("scripts\phase8_stage3_sim.py", "cpu", "`"$sp`"", "--runs", "4") `
    -RedirectStandardOutput "$root\logs\s3_$i.log" -RedirectStandardError "$root\logs\s3_$i.err"
}
$procs | Wait-Process
Log "done"
