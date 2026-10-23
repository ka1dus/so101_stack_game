# Wait for a training run's 30k checkpoint, then evaluate 10k/20k/30k in sim (60 states, seed 777, temporal ensembling).
# Results: data\eval\<run>_<step>_te\summary.txt; progress in logs\eval_<run>.log
param([string]$Root, [string]$Py, [string]$RunDir, [string]$Run, [string]$Resize = "")
cd $Root
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$Root\logs\eval_$Run.log" }
Log "waiting for $RunDir\checkpoints\030000"
while (-not (Test-Path "$RunDir\checkpoints\030000\pretrained_model\model.safetensors")) { Start-Sleep 60 }
Start-Sleep 60                                      # let the trainer finish writing and exit
$procs = @()
foreach ($s in "010000", "020000", "030000") {
  $a = @("scripts\sim_eval.py", "--policy", "$RunDir\checkpoints\$s\pretrained_model", "--n", "60", "--seed", "777", "--videos", "3",
         "--temporal-ensemble", "0.01", "--out", "data\eval\${Run}_${s}_te")
  if ($Resize) { $a += @("--resize", $Resize) }
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $Py -ArgumentList $a -RedirectStandardOutput "$Root\logs\eval_${Run}_$s.log" -RedirectStandardError "$Root\logs\eval_${Run}_$s.err"
}
Log "started evals for 10k/20k/30k"
$procs | ForEach-Object { $_.WaitForExit() }
foreach ($s in "010000", "020000", "030000") { Log ("$s : " + ((Get-Content "data\eval\${Run}_${s}_te\summary.txt" -ErrorAction SilentlyContinue | Select-String "^success") -join " ")) }
Log "done"
