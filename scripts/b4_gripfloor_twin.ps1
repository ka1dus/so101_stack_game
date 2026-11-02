# botmind: sim twin of the 22 trials of b4_stage2_twin.ps1 with a gripper-opening rule (skill/grip_floor.py).
# Usage: b4_gripfloor_twin.ps1 <tag>, tag = floor<N> (--grip-floor N) or cap<N> (--grip-cap N). 4 runs per trial,
# CPU, 6 in parallel. Log logs\gf_<tag>_summary.log; per trial logs\gf_<tag>_<session>_t<n>.log
param([string]$tag, [string]$set = "")
if ($tag -match '^floor(\d+)rel(\d+)$') { $extra = @("--grip-floor", $Matches[1], "--release-floor", $Matches[2]) }
elseif ($tag -match '^floor(\d+)$') { $extra = @("--grip-floor", $Matches[1]) } elseif ($tag -match '^cap(\d+)$') { $extra = @("--grip-cap", $Matches[1]) } else { throw "bad tag $tag" }
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
$pol = "C:\Users\botmi\so101_runs\act_wcal\checkpoints\060000\pretrained_model"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\gf$($set)_${tag}_summary.log" }
$jobs = @()
if ($set -eq "0429") {           # the 29 Oct --grip-floor 28 trials
  $jobs += ,@("20261029_111738", 1); foreach ($t in 1..3) { $jobs += ,@("20261029_114535", $t) }; foreach ($t in 1..6) { $jobs += ,@("20261029_121913", $t) }
} else {
  foreach ($t in 1..7) { $jobs += ,@("20261028_201746", $t) }
  foreach ($t in 1..5) { $jobs += ,@("20261028_203311", $t) }
  foreach ($t in 1..10) { $jobs += ,@("20261028_195923", $t) }
}
Log "start $tag ($extra): $($jobs.Count) trials x 4 runs"
$procs = @()
foreach ($j in $jobs) {
  while (@($procs | Where-Object { -not $_.HasExited }).Count -ge 6) { Start-Sleep 10 }
  $name = "gf$($set)_$($tag)_$($j[0])_t$($j[1])"
  $args_ = @("scripts\phase8_sim_twin.py", "data\real_trials\$($j[0])", "$($j[1])", "--policy", $pol, "--runs", "4", "--device", "cpu") + $extra
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList $args_ `
    -RedirectStandardOutput "$root\logs\$name.log" -RedirectStandardError "$root\logs\$name.err"
}
$procs | Wait-Process
Log "done"
