# botmind: export the v2 pilot raw episodes to two LeRobot datasets (full 640x480 and half 320x240), 4 parts each
# in parallel, then verified merges. Log: logs\pilot_export.log
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\pilot_export.log" }
$runs = Get-ChildItem -Directory "$root\data\raw\act_pilot_w*" | Sort-Object Name | ForEach-Object { $_.FullName }
$groups = @(0..3 | ForEach-Object { $k = $_; ,@($runs | Where-Object { [array]::IndexOf($runs, $_) % 4 -eq $k }) })
$procs = @()
foreach ($res in @("full", "half")) {
  foreach ($k in 0..3) {
    $out = "$ds\pilot_${res}_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
    $args_ = @("scripts\export_lerobot.py", "--raw") + $groups[$k] + @("--dataset", "act", "--repo-id", "local/pilot_${res}_part$k", "--root", $out, "--streaming", "--queue", "600")
    if ($res -eq "half") { $args_ += @("--resize", "320x240") }
    $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList $args_ -RedirectStandardOutput "$root\logs\pilot_export_${res}_$k.log" -RedirectStandardError "$root\logs\pilot_export_${res}_$k.err"
  }
}
Log "started $($procs.Count) export processes"
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($res in @("full", "half")) {
  $bad = 0
  foreach ($k in 0..3) { if (-not (Select-String -Quiet -Path "$root\logs\pilot_export_${res}_$k.log" -Pattern "video check: every video")) { $bad++ } }
  if ($bad) { Log "$res export: $bad parts failed the video check; not merging"; continue }
  $parts = 0..3 | ForEach-Object { "$ds\pilot_${res}_part$_" }
  Remove-Item -Recurse -Force "$ds\pilot_$res" -ErrorAction SilentlyContinue
  & $py scripts\merge_lerobot.py --parts $parts --repo-id "local/pilot_$res" --out "$ds\pilot_$res" *> "$root\logs\pilot_merge_$res.log"
  Log ("$res merge exit ${LASTEXITCODE}: " + (Get-Content "$root\logs\pilot_merge_$res.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / ")
}
Log "done"
