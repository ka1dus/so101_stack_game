# botmind: everything after the calibrated-wrist generation, chained here so it does not depend on the Mac being awake:
# export + merge (act_wctall_w*, act_wcgame_w*, act_pcwcal_w*) -> so101_stack_sim_act_wcal_half (320x240); fine-tune B80
# (soft-servo policy) 60k steps on it (checkpoints 30k, 60k); evaluate both (and B80 as a baseline) with the fitted servos on the 460 states.
# Log: logs\wcal_steps.log
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$lr = "$root\.venv-lerobot\Scripts\python.exe"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\wcal_steps.log" }
function Stop-Run($m) { Log "FAILED: $m"; exit 1 }
$raw = @(Get-ChildItem -Directory "$root\data\raw\act_wctall_w*", "$root\data\raw\act_wcgame_w*", "$root\data\raw\act_pcwcal_w*" | Sort-Object Name | ForEach-Object { $_.FullName })
$n = (Get-ChildItem -Directory ($raw | ForEach-Object { "$_\ep_*" })).Count
Log "export: $n episodes in $($raw.Count) folders"
$procs = @()
foreach ($k in 0..7) {
  $grp = @($raw | Where-Object { [array]::IndexOf($raw, $_) % 8 -eq $k }); $out = "$ds\wcal_half_part$k"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $a = @("scripts\export_lerobot.py", "--raw") + $grp + @("--dataset", "act", "--repo-id", "local/wcal_half_part$k", "--root", $out, "--streaming", "--queue", "600", "--resize", "320x240")
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $lr -ArgumentList $a -RedirectStandardOutput "$root\logs\wcal_export_$k.log" -RedirectStandardError "$root\logs\wcal_export_$k.err"
}
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($k in 0..7) { if (-not (Select-String -Quiet -Path "$root\logs\wcal_export_$k.log" -Pattern "video check: every video")) { Stop-Run "export part $k failed" } }
Remove-Item -Recurse -Force "$ds\so101_stack_sim_act_wcal_half" -ErrorAction SilentlyContinue
$parts = @(0..7 | ForEach-Object { "$ds\wcal_half_part$_" })
& $lr scripts\merge_lerobot.py --parts $parts --repo-id Maxctl352/so101_stack_sim_act_wcal --out "$ds\so101_stack_sim_act_wcal_half" *> "$root\logs\wcal_merge.log"
if ($LASTEXITCODE -ne 0) { Stop-Run "merge failed" }
Log ("export done: " + ((Get-Content "$root\logs\wcal_merge.log" | Select-String "merged:|video check of the merged" | ForEach-Object { $_.Line }) -join " / "))
Log "train: 60k steps from B80"
Remove-Item -Recurse -Force "$runs\act_wcal" -ErrorAction SilentlyContinue
& "$root\.venv-lerobot\Scripts\lerobot-train.exe" --policy.path="$runs\act_softB_080000\pretrained_model" `
  --dataset.repo_id=Maxctl352/so101_stack_sim_act_wcal --dataset.root="$ds\so101_stack_sim_act_wcal_half" `
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_wcal" --job_name=act_wcal `
  --batch_size=8 --steps=60000 --save_freq=30000 --log_freq=500 --num_workers=4 --wandb.enable=false *> "$root\logs\wcal_train.log"
if (-not (Test-Path "$runs\act_wcal\checkpoints\060000\pretrained_model\model.safetensors")) { Stop-Run "training did not reach 60k" }
Log "train done"
# the 30k and 60k checkpoints, plus B80 itself on the rebuilt scene (baseline: same scene, old training data)
$e = @()
foreach ($j in @(@("$runs\act_wcal\checkpoints\030000\pretrained_model", "wcal30"), @("$runs\act_wcal\checkpoints\060000\pretrained_model", "wcal60"),
                 @("$runs\act_softB_080000\pretrained_model", "b80wcal"))) {
  $e += Start-Process -PassThru -NoNewWindow powershell -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", "scripts\b4_soft_eval.ps1", $j[0], $j[1], "4")
  Start-Sleep 60
}
$e | ForEach-Object { $_.WaitForExit() }
Log "chain done"
