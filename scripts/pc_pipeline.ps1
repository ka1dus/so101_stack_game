# Overnight pipeline on the Windows GPU PC (Phase 5 export -> Phase 6 ACT training). Started detached by
# run_detached.ps1 (inherits the lerobot env PATH). Each stage logs to logs\<stage>.log; progress in logs\pipeline.log.
$root = "$env:USERPROFILE\so101_stack_game"; $ds = "$env:USERPROFILE\so101_datasets"; $runs = "$env:USERPROFILE\so101_runs"
cd $root
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\pipeline.log" }
function WaitExports($names) {
  while ($true) {
    $done = 0
    foreach ($n in $names) {
      $t = Get-Content -Raw "$root\logs\$n.log" -ErrorAction SilentlyContinue
      if ($t -match "EXIT (\d+)") { if ($Matches[1] -ne "0") { Log "$n failed (exit $($Matches[1])); stopping"; exit 1 }; $done++ }
    }
    if ($done -eq $names.Count) { return }
    Start-Sleep 60
  }
}
Log "pipeline started; starting 4 ACT exports (streaming, queue 600 = no dropped frames, verified)"
foreach ($w in 0..3) {
  & "$root\scripts\run_detached.ps1" -Name "export_act_w$w" -Cmd "python scripts\export_lerobot.py --raw data\raw\act_main_w$w --dataset act --repo-id Maxctl352/so101_stack_sim_act_part$w --root %USERPROFILE%\so101_datasets\act_part$w --streaming --queue 600" | Out-Null
}
WaitExports @("export_act_w0", "export_act_w1", "export_act_w2", "export_act_w3")
Log "merging ACT parts"
python scripts\merge_lerobot.py --parts $ds\act_part0 $ds\act_part1 $ds\act_part2 $ds\act_part3 --repo-id Maxctl352/so101_stack_sim_act --out $ds\so101_stack_sim_act *> "$root\logs\merge_act.log"
if ($LASTEXITCODE -ne 0) { Log "ACT merge failed; stopping"; exit 1 }
Log ("ACT merge done: " + (Get-Content "$root\logs\merge_act.log" | Select-String "merged:" | Select-Object -Last 1))
Log "starting ACT training (AMP, 100k steps, checkpoint every 10k) -> $runs\act_v1"
$train = Start-Process -PassThru -NoNewWindow -FilePath "lerobot-train" -RedirectStandardOutput "$root\logs\train_act.log" -RedirectStandardError "$root\logs\train_act.err" -ArgumentList @(
  "--dataset.repo_id=Maxctl352/so101_stack_sim_act", "--dataset.root=$ds\so101_stack_sim_act", "--policy.type=act",
  "--policy.device=cuda", "--policy.use_amp=true", "--policy.push_to_hub=false", "--output_dir=$runs\act_v1",
  "--job_name=act_so101_stack_v1", "--batch_size=8", "--steps=100000", "--save_freq=10000", "--log_freq=200",
  "--num_workers=4", "--wandb.enable=false")
Log "training pid $($train.Id); the VLA export runs after training (RAM)"
$train.WaitForExit()
Log "ACT training finished (exit $($train.ExitCode)); starting 4 VLA exports"
foreach ($w in 0..3) {
  & "$root\scripts\run_detached.ps1" -Name "export_vla_w$w" -Cmd "python scripts\export_lerobot.py --raw data\raw\act_main_w$w data\raw\vla_main_w$w --dataset vla --repo-id Maxctl352/so101_stack_sim_vla_part$w --root %USERPROFILE%\so101_datasets\vla_part$w --streaming --queue 600" | Out-Null
}
WaitExports @("export_vla_w0", "export_vla_w1", "export_vla_w2", "export_vla_w3")
Log "merging VLA parts"
python scripts\merge_lerobot.py --parts $ds\vla_part0 $ds\vla_part1 $ds\vla_part2 $ds\vla_part3 --repo-id Maxctl352/so101_stack_sim_vla --out $ds\so101_stack_sim_vla *> "$root\logs\merge_vla.log"
if ($LASTEXITCODE -ne 0) { Log "VLA merge failed" } else { Log ("VLA merge done: " + (Get-Content "$root\logs\merge_vla.log" | Select-String "merged:" | Select-Object -Last 1)) }
Log "pipeline done"
