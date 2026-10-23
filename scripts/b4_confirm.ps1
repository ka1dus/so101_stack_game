# botmind: confirmation evaluation of ACT v2 (300k, 320x240, temporal ensembling) against the Section 1 sim criteria.
#   A: 600 more game-rule states (seeds 3005-3016); B: crooked (>= 10 mm) height 4-7 states per picked type, 60 each
#   (seeds 4001-4006); C: large onto small, 80 (seeds 5001-5002). Report pools A-C + the first game set (3001-3004).
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$py = "$root\.venv-lerobot\Scripts\python.exe"; $ck = "C:\Users\botmi\so101_runs\act_v2_half\checkpoints\300000\pretrained_model"
$env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content "$root\logs\confirm.log" }
$common = @("--policy", $ck, "--states", "game", "--videos", "0", "--temporal-ensemble", "0.01", "--resize", "320x240")
$jobs = @()
foreach ($s in 3005..3016) { $jobs += ,@("A_game_s$s", @("--n", "50", "--seed", "$s")) }
$types = @{4001 = "no3"; 4002 = "no3"; 4003 = "no35"; 4004 = "no35"; 4005 = "no10"; 4006 = "no10"}
foreach ($s in 4001..4006) { $jobs += ,@("B_crooked_$($types[$s])_s$s", @("--n", "30", "--seed", "$s", "--want-type", $types[$s], "--height-min", "4", "--hard-p", "1", "--min-crooked-mm", "10")) }
foreach ($s in 5001..5002) { $jobs += ,@("C_large_on_small_s$s", @("--n", "40", "--seed", "$s", "--want-pairing", "no3_on_no10", "--hard-p", "1")) }
$procs = @()
foreach ($j in $jobs) {
  $out = "data\eval\confirm_$($j[0])"; Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList (@("scripts\sim_eval.py") + $common + $j[1] + @("--out", $out)) `
    -RedirectStandardOutput "$root\logs\confirm_$($j[0]).log" -RedirectStandardError "$root\logs\confirm_$($j[0]).err"
}
Log "started $($procs.Count) evaluation processes"
$procs | ForEach-Object { $_.WaitForExit() }
$dirs = @(Get-ChildItem -Directory "data\eval\confirm_*" | ForEach-Object { $_.FullName }) + @(Get-ChildItem -Directory "data\eval\v2half_300k_game_s*" | ForEach-Object { $_.FullName })
$missing = @($dirs | Where-Object { -not (Test-Path "$_\episodes.json") })
if ($missing.Count) { Log ("runs without results: " + ($missing -join ", ")) }
$ok = @($dirs | Where-Object { Test-Path "$_\episodes.json" })
& $py scripts\confirm_report.py --out "data\eval\confirmation.md" $ok *> "$root\logs\confirm_report.log"
Log "report written (data\eval\confirmation.md)"
Log "done"
