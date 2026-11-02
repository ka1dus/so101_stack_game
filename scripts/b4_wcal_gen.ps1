# botmind: data with the calibrated wrist camera (2026-10-28, user approved): same recipe as scripts\b4_soft_gen.ps1
# (fitted soft servos +-30 %, expert compensation; 1,000 targeted + game episodes) on the rebuilt scene (wrist mount
# and intrinsics from scripts\wrist_calib_fit.py, wrist blur sigma 0.6-2.5 px). The PC adds game episodes
# (act_pcwcal_w*); the Mac stops both at 1,000 targeted + 2,000 game. Log: logs\wcal_steps.log
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$sim = "$root\.venv-sim\Scripts\python.exe"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\wcal_steps.log" }
$servo = @("--servo", "sim\assets\servo_fit.yaml", "--servo-jitter", "0.3")
Log "stage 1a: 1,000 targeted episodes (calibrated wrist camera)"
& $sim scripts\run_generation.py --workers 16 --act 1000 --vla 0 --prefix wctall --seed-base 71000 `
  --height-min 4 --hard-p 1.0 --want-types no3,no35 --min-crooked-mm 10 --crooked-share 0.7 @servo *> "$root\logs\wcal_generate_tall.log"
Log "stage 1b: game episodes (calibrated wrist camera)"
& $sim scripts\run_generation.py --workers 16 --act 2000 --vla 0 --prefix wcgame --seed-base 70000 @servo *> "$root\logs\wcal_generate_game.log"
$n = (Get-ChildItem -Directory "$root\data\raw\act_wctall_w*\ep_*", "$root\data\raw\act_wcgame_w*\ep_*").Count
Log "generation done: $n episodes on botmind"
