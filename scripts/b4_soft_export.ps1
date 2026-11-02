# botmind: export + merge the soft-servo episodes (botmind act_softtall_w*, act_softgame_w*; the PC's act_pcsoft_w*
# copied in) -> so101_stack_sim_act_soft_half (320x240). Log: logs\soft_steps.log
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; cd $root
$lr = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\soft_steps.log" }
function Stop-Run($m) { Log "FAILED: $m"; exit 1 }
$raw = @(Get-ChildItem -Directory "$root\data\raw\act_softtall_w*", "$root\data\raw\act_softgame_w*", "$root\data\raw\act_pcsoft_w*" | Sort-Object Name | ForEach-Object { $_.FullName })
$n = (Get-ChildItem -Directory ($raw | ForEach-Object { "$_\ep_*" })).Count
Log "export: $n episodes in $($raw.Count) folders"
$procs = @()
foreach ($k in 0..7) {
  $grp = @($raw | Where-Object { [array]::IndexOf($raw, $_) % 8 -eq $k }); $out = "$ds\soft_half_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $a = @("scripts\export_lerobot.py", "--raw") + $grp + @("--dataset", "act", "--repo-id", "local/soft_half_part$k", "--root", $out, "--streaming", "--queue", "600", "--resize", "320x240")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\soft_export_$k.log" -RedirectStandardError "$root\logs\soft_export_$k.err"
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($k in 0..7) { if (-not (Select-String -Quiet -Path "$root\logs\soft_export_$k.log" -Pattern "video check: every video")) { Stop-Run "export part $k failed" } }
Remove-Item -Recurse -Force "$ds\so101_stack_sim_act_soft_half" -ErrorAction SilentlyContinue
$parts = @(0..7 | ForEach-Object { "$ds\soft_half_part$_" })
& $lr scripts\merge_lerobot.py --parts $parts --repo-id Maxctl352/so101_stack_sim_act_soft --out "$ds\so101_stack_sim_act_soft_half" *> "$root\logs\soft_merge.log"
if ($LASTEXITCODE -ne 0) { Stop-Run "merge failed" }
Log ("export done: " + ((Get-Content "$root\logs\soft_merge.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / "))
