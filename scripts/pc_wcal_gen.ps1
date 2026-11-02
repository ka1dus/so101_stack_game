# 3060 Ti PC: its share of the calibrated-wrist game episodes (2026-10-28, see scripts\b4_wcal_gen.ps1). Stopped by the
# Mac once botmind + PC reach 2,000 game episodes; the raw episodes then go to botmind. Log: logs\pc_wcal_gen.log
$root = "C:\Users\User\so101_stack_game"; cd $root
$env:PYTHONUTF8 = "1"; $env:PATH = "C:\Users\User\miniforge3\envs\lerobot\Library\bin;" + $env:PATH
& "$root\.venv-sim\Scripts\python.exe" scripts\run_generation.py --workers 5 --act 1000 --vla 0 --prefix pcwcal --seed-base 72000 `
  --servo sim\assets\servo_fit.yaml --servo-jitter 0.3 *> "$root\logs\pc_wcal_gen.log"
