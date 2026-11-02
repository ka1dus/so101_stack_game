# botmind: soft-servo data (2026-10-27, user approved): episodes generated with the real-arm servo fit
# (sim\assets\servo_fit.yaml, kp/kv jittered +-30 % per episode) and the expert's soft-servo compensation.
#   1,000 targeted (heights 4-7, rule picks no3/no35, 70 % crooked >= 10 mm: the weakest bucket), then
#   2,000 game-distribution episodes (the 3060 Ti PC may add part of these as act_pcsoft_w*; the game stage is
#   stopped early once both together reach 2,000). Log: logs\soft_steps.log
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$sim = "$root\.venv-sim\Scripts\python.exe"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\soft_steps.log" }
$servo = @("--servo", "sim\assets\servo_fit.yaml", "--servo-jitter", "0.3")
Log "stage 1a: 1,000 targeted soft-servo episodes"
& $sim scripts\run_generation.py --workers 16 --act 1000 --vla 0 --prefix softtall --seed-base 61000 `
  --height-min 4 --hard-p 1.0 --want-types no3,no35 --min-crooked-mm 10 --crooked-share 0.7 @servo *> "$root\logs\soft_generate_tall.log"
Log "stage 1b: 2,000 game soft-servo episodes"
& $sim scripts\run_generation.py --workers 16 --act 2000 --vla 0 --prefix softgame --seed-base 60000 @servo *> "$root\logs\soft_generate_game.log"
$n = (Get-ChildItem -Directory "$root\data\raw\act_softtall_w*\ep_*", "$root\data\raw\act_softgame_w*\ep_*").Count
Log "generation done: $n episodes on botmind"
