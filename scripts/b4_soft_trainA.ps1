# botmind: variant A = v4 (40k) fine-tuned 40k steps on the soft-servo dataset only, then evaluated with the fitted servos.
$root = "C:\Users\botmi\so101_stack_game"; $ds = "C:\Users\botmi\so101_datasets"; $runs = "C:\Users\botmi\so101_runs"; cd $root
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\soft_steps.log" }
Log "train A: 40k steps from v4 40k"
Remove-Item -Recurse -Force "$runs\act_softA" -ErrorAction SilentlyContinue
& "$root\.venv-lerobot\Scripts\lerobot-train.exe" --policy.path="$runs\act_v4_ft\checkpoints\040000\pretrained_model" `
  --dataset.repo_id=Maxctl352/so101_stack_sim_act_soft --dataset.root="$ds\so101_stack_sim_act_soft_half" `
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_softA" --job_name=act_softA `
  --batch_size=8 --steps=40000 --save_freq=20000 --log_freq=500 --num_workers=4 --wandb.enable=false *> "$root\logs\softA_train.log"
if (-not (Test-Path "$runs\act_softA\checkpoints\040000\pretrained_model\model.safetensors")) { Log "FAILED: train A did not reach 40k"; exit 1 }
Log "train A done"
& powershell -ExecutionPolicy Bypass -File scripts\b4_soft_eval.ps1 "$runs\act_softA\checkpoints\040000\pretrained_model" softA 8
