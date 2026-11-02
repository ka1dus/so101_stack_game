# 3060 Ti PC: restart of worker 0 of scripts\pc_wcal_gen.ps1 (it failed to create its OpenGL context at start-up).
$root = "C:\Users\User\so101_stack_game"; cd $root
$env:PYTHONUTF8 = "1"; $env:PATH = "C:\Users\User\miniforge3\envs\lerobot\Library\bin;" + $env:PATH
& "$root\.venv-sim\Scripts\python.exe" scripts\generate.py --n 200 --mode act --seed 72000 --run act_pcwcal_w0 `
  --servo sim\assets\servo_fit.yaml --servo-jitter 0.3 *> "$root\data\raw\logs\pcwcal_w0.log"
