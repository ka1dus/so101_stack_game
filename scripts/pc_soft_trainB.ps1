# 3060 Ti PC: variant B = v4 (40k) fine-tuned 80k steps on the soft-servo dataset only (checkpoints at 40k and 80k;
# both are evaluated on botmind). Dataset and v4 checkpoint copied from botmind.
$envd = "$env:USERPROFILE\miniforge3\envs\lerobot"; $env:PATH = "$envd;$envd\Library\bin;$envd\Scripts;$env:PATH"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
$root = "$env:USERPROFILE\so101_stack_game"; $ds = "$env:USERPROFILE\so101_datasets"; $runs = "$env:USERPROFILE\so101_runs"; cd $root
Remove-Item -Recurse -Force "$runs\act_softB" -ErrorAction SilentlyContinue
& lerobot-train --policy.path="$runs\act_v4_ft_040000" `
  --dataset.repo_id=Maxctl352/so101_stack_sim_act_soft --dataset.root="$ds\so101_stack_sim_act_soft_half" `
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --output_dir="$runs\act_softB" --job_name=act_softB `
  --batch_size=8 --steps=80000 --save_freq=40000 --log_freq=500 --num_workers=4 --wandb.enable=false
