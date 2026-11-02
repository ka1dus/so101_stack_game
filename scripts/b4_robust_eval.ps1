# botmind: B80 robustness check: the fitted servos with kp/kv scaled x0.7 and x1.4 (both evals at once, 7 jobs each).
#   powershell -File scripts\run_detached_b4.ps1 robust "powershell -ExecutionPolicy Bypass -File scripts\b4_robust_eval.ps1"
$root = "C:\Users\botmi\so101_stack_game"; cd $root
$ck = "C:\Users\botmi\so101_runs\act_softB_080000\pretrained_model"
$p = @()
foreach ($x in @(@("070", "0.7"), @("140", "1.4"))) {
  $p += Start-Process -PassThru -NoNewWindow powershell -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", "scripts\b4_soft_eval.ps1", $ck, "softB80_x$($x[0])", "7", $x[1])
  Start-Sleep 60
}
$p | ForEach-Object { $_.WaitForExit() }
