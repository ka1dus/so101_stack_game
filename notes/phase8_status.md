# Phase 8 status (handoff)

Updated 2026-10-28 17:25.

## Done
- First real trial (`data/real_trials/20261027_115530`): failed. The arm carried the box low, pressed it on the table and timed out.
- Replay in sim + tracking tests (empty / box): the real servos lag and sag; the sim servos were far too stiff.
- Servo fit: `sim/assets/servo_fit.yaml` (kp 13–34, kv 0.7–2.7); tracking error 1.02 → 0.62 units.
- ACT v4 with the fitted servos (botmind, CPU): **276/460 (60 %)** vs 416/460 with the old servos. Results: botmind `data\eval\servo_*`.
- Expert fix: soft-servo compensation in `scripts/grasp_test.py` (`Sim.set_comp`: command = plan + (gravity + kv·plan velocity)/kp, clipped to ±10 units of the reading). With the fitted servos: 30/34 attempts ok, 0 knocks (12 knocks / 44 without it).
- `generate.py --servo --servo-jitter 0.3` (kp/kv × exp(U(±0.3)) per episode); `sim_eval.py --servo`.
- The user said to ignore wrong-box picks for now and focus on stacking.

## Running now (user approved: split across botmind + 3060 Ti PC)
1. DONE 16:28: 3,009 episodes (botmind 1,000 targeted + 1,334 game; PC 675 game, moved to botmind `data\raw`). Export started 16:36 (`softexport`, log `logs\soft_steps.log`, waits for "export done").
   Generation of 3,000 soft-servo episodes:
   - botmind `scripts\b4_soft_gen.ps1`: 1,000 targeted (`act_softtall_w*`), then game (`act_softgame_w*`); log `logs\soft_steps.log`.
   - PC (new sim venv `C:\Users\User\so101_stack_game\.venv-sim`, made from the lerobot env's Python 3.12; ffmpeg from its `Library\bin`) `scripts\pc_soft_gen.ps1`: game episodes `act_pcsoft_w*`.
   - Mac orchestrator `<scratchpad>/orchestrate_gen.sh` (log next to it): when tall ≥ 1,000 and game + PC ≥ 2,000 it stops both (`scripts\stop_generation.ps1`), moves the PC episodes to botmind and starts `scripts\b4_soft_export.ps1` (→ `so101_stack_sim_act_soft_half`).

## Next
- 2026-10-28 23:05 real no3 slot-2 failures = gripper opened only ~24 (stage 1 successes ~35; sim same scene ~27, enough in sim): jaws land on the box, miss, retry elsewhere. Not images. Options: opening floor in the runner during the approach (no retrain, test in sim twin first) and/or wider pre-grasp margin in the next data run.
- 2026-10-28 22:20 stage-2 sim twin done (21:20): no10 choice fails in sim too (0/12) -> data; no3 from slot 2 is right in sim 32/32 but real 3/8 -> sim-to-real gap at the pick, replay test next (notes/phase8.md §17).
- 2026-10-28 21:00 **Plan (user approved): ACT first, SmolVLA later as a comparison.** Step 1 running on botmind: `scripts\b4_stage2_twin.ps1` (task `stage2twin`): sim twin of the 22 real trials (stage 2 + stage-1 batch), 4 runs each, reports which box wcal60 picks in sim (`PICK` lines in botmind `logs\twin_<session>_t<n>.log`; log `logs\stage2_twin.log`, ETA ~21:45). First result: 201746 t1 picks the wrong box in sim too. Step 2 (needs the user's OK): new data + fine-tune targeting small-box choice / set-aside supplies, plus background behind the arm and arm-look randomization and a lower no3 release.
- 2026-10-28 20:45 stage 2: 3/12; all 9 failures are picks (6 wrong box, 3 missed grasps), placement with the right box 3/3. no10 never picked (0/4). One more false collapse call on a high no3 release. Wrist camera error at 640x360 mid-session. Waiting for the user: fix picking next?
- 2026-10-28 20:30 stage 1 batch (`20261028_195923`): right-box placements 8/8 (12/12 with --plain-wall overall), wrong box 2/10, no3 releases +2.3–2.8 cm high, perception 10/10. Next: stage 2 (mixed pairings) with --plain-wall.
- 2026-10-28 20:10 false collapse call (193906 t3) = the high release's fall counted as movement; fixed with `landed_tick` (trial + controller). Caveat: live check also missed a real failure (105812 t1); settle frames + features now logged per trial to tune the collapse check on real data (notes/phase8.md §14).
- 2026-10-28 19:45 **--plain-wall real trials 4/4** (`20261028_193906`): no35>no35 2/2 (release 2.8 cm), no3>no3, no3 no35>no10. One false collapse call by perception (trial 3) to check. Next: more stage-1 trials with --plain-wall toward ~10 per bucket; background randomization for the next data run.
- 2026-10-28 19:00 cover test (bag): no35>no35 still 0/2, no3>no3 1/1. Faithful replay (`--ensemble`, reproduces logged commands): the room behind the arm in the front image makes the policy stop ~1 cm high over a single no35 (sim room or plain-painted banner area -> sim height and release). Bag cover was glossy/crumpled. Next: real trial with a flat matte cover, or `phase8_trial.py --plain-wall` (software paint, skill/plain_wall.py).
- 2026-10-28 18:05 image-swap test: the **front** image stops the release (real front 0/18 open, sim front 16–17/18; wrist irrelevant). Background is the main factor (sim wall pasted in: 8/9; logo painted over: 6–7/9), arm look second (6/9). Gripper reading ruled out. Proposed: cover the banner with a plain dark cloth (real, no retrain), then add background/arm-look randomization for the next data run (notes/phase8.md §13).
- 2026-10-28 17:35 wcal60 real stage 1: 2/4 (no3>no3 and no35 no35>no35 stacked, offset; no35>no35 hovered twice, timeout). Releases 2–3 cm short toward the arm and 1–2.5 cm higher than sim. Proposed next: sim test feeding the real gripper reading (cardboard squeeze) to see if that alone makes the policy hover (notes/phase8.md §13).
- 2026-10-28 17:40 first wcal60 real trial (`20261028_171710`): right box, carried, hovered 3–4 cm short and 2 cm high over the stack, never released (timeout). Mapping and wrist camera check out; grasp differs (real gripper 15 vs sim 18–21). Asked the user for more stage-1 trials (notes/phase8.md §13).
- 2026-10-28 17:25 **wcal60 done:** 372/460, stacking with the right box held 92 %, knocks on release 11 -> 5 (notes/phase8.md §12). Copied to `data/policies/act_wcal_060000`; sim twin of trial 1 4/4. Waiting for the user's OK for stage 1 real trials with `--policy data/policies/act_wcal_060000`.
- 2026-10-28 12:30 **Calibrated-wrist run started** (user approved): wrist mount + intrinsics in `sim/assets/cameras.yaml` (scene rebuilt; only the wrist camera and housing changed), wrist blur 0.6–2.5 px. Generation: botmind `scripts\b4_wcal_gen.ps1` (act_wctall_w*, act_wcgame_w*), PC `scripts\pc_wcal_gen.ps1` (act_pcwcal_w*; worker 0 restarted with `pc_wcal_w0.ps1` after a GL start-up failure). Mac `<scratchpad>/orchestrate_wcal.sh` stops both at 1,000 + 2,000, moves PC episodes, starts `scripts\b4_wcal_chain.ps1` on botmind (export -> fine-tune B80 60k -> eval wcal30, wcal60, b80wcal; log `logs\wcal_steps.log`). Watchdog milestones: chain start 13:55, export 14:35, train 15:45, evals 17:00.
- 2026-10-28 real trials with B80 released ~2 cm high (notes/phase8.md §10); wrist camera gap measured. **Wrist calibration prepared:** board `notes/wrist_calib/charuco_board_A4.pdf`, poses `notes/wrist_calib/poses.json` (`scripts/wrist_calib_poses.py`, 14 views, sim-checked), capture `scripts/wrist_calib_capture.py` (--handheld for intrinsics, then the pose tour). Waiting for the user's captures; then write the fit (intrinsics + hand-eye), overlays, regenerate data, fine-tune.
- 2026-10-28 02:36 B80 robustness: stacking 90 / 90 / 88 % with servos ×0.7 / ×1.0 / ×1.4 (`notes/phase8.md` §9). Waiting for the user's access to the setup for the stage-1 real trial with B80 (runs 'train B longer' and 'sim games' held until after it).
- 21:21 **Results (fitted servos, 460 states)**: v4 276, A 324, B40 341, **B80 354/460 (77 %)**. Stacking with the right box held: v4 74 %, A 87 %, B40 90 %, B80 90 % (v4 with stiff servos: 92 %). B80 also halves missed grasps (52 → 27); wrong box 89. Criteria: large-on-small 86 % and crooked 4+ no35 confirmed; no10 buckets low only through wrong-box / missed grasps of the small box. **B80 picked for the next real trial**, copied to `data/policies/act_softB_080000` (waiting for the user's OK).
- 18:43 **A result (fitted servos): 324/460 (70 %)** vs v4 276/460. With the right box in hand: 295/339 (87 %) vs v4 234/318 (74 %); drops 26 → 5, knocks on approach 23 → 10. Left: knocks on release at height 4+ (13), misplaced (13), missed grasps 53 and wrong box 68 (pick choice, deferred). Copies of the eval folders: `<scratchpad>/eval/`.
- B done: 40k at 19:26, 80k at 20:22 (copied to botmind `C:\Users\botmi\so101_runs\act_softB_0{4,8}0000`). Mac asleep ~19:00–20:26, so botmind idled. 20:29 evals `softB40` / `softB80` started (7 parallel each), ETA ~21:15.
- (old) B training on the PC since 18:28 (log `logs\softB.log`), 40k ≈ 19:25, 80k ≈ 20:22. Watchdog `<scratchpad>/watch.sh` + `milestones.txt`.
- 17:05 export done (3,009 episodes, 8.2 GB). 17:30 A training started on botmind (`softA`, log `logs\softA_train.log`). Dataset relay to the PC + B start running from the Mac.
2. After the export: start `scripts\b4_soft_trainA.ps1` on botmind (A = v4 + 40k steps on soft data, then `b4_soft_eval.ps1` → `data\eval\softA_*`).
   Copy `so101_stack_sim_act_soft_half` to the PC (`C:\Users\User\so101_datasets\`) and start `scripts\pc_soft_trainB.ps1` there (B = v4 + 80k steps, checkpoints 40k/80k; v4 checkpoint already at `C:\Users\User\so101_runs\act_v4_ft_040000`).
3. Copy B's checkpoints to botmind and run `b4_soft_eval.ps1 <ck> softB40` / `softB80`.
4. Compare A, B40, B80 against v4 (old servos 416/460; fitted 276/460) by bucket and failure type; pick the policy for the next real trial.

Launch on botmind: `powershell -File C:\Users\botmi\so101_stack_game\scripts\run_detached_b4.ps1 <name> "<cmd>"`.
Launch on the PC: `powershell -ExecutionPolicy Bypass -File C:\Users\User\so101_stack_game\scripts\run_detached.ps1 <name> "<cmd>"`.

## 2026-10-29 01:05: gripper-opening floor test (sim twin, botmind CPU)
`skill/grip_floor.py` (runner `act_fn` hook): after the policy starts opening (> 15), command at least N until it starts closing; then pass-through.
Twin of the 22 stage-1/2 trials x 4 runs, three conditions in parallel: `floor33`, `cap24` (mimic the real narrow opening), `floor0` (baseline with the new neighbour-bump metric).
Launch: `scripts/b4_gripfloor_twin.ps1 <tag>`; logs `logs\gf_<tag>_summary.log`, `logs\gf_<tag>_<session>_t<n>.log`. ETA ~01:40 (3 x 6 processes; baseline alone took 23 min).
02:10: floor33 hurts in sim (no3 slot 2 right box 32 -> 21/32, neighbour bumps 4 -> 15/32): 33 = 60 mm tip gap vs 62 mm between the no3's neighbours. cap24 still grasps in sim (24 = 44.6 mm tip gap, 1.8 mm margin 1 cm back: sim aim is exact, real is not). Running floor28 / floor30 (ETA ~03:10).
03:15: floor28 / floor30 done (notes/phase8.md §18). floor28: no3 slot 2 32/32, nudges 4 → 9 (max 14 mm). `phase8_trial.py --grip-floor 28` added; waiting for the user's real stage-2 run.

## 2026-10-29 13:50: scripted place (user away ~1 h; sim only, no real runs)
- `scripts/phase8_inhand.py`: the real no3 is held within 3 mm of its centre along its length (earlier "held near one end" was wrong); real releases put the box centre 11-27 mm off the top centre (bias about (-15, -13) mm) and 3-22 mm above the face; sim twin 3-15 mm off, 0-11 mm up.
- `skill/scripted_place.py` + runner `handover_lift` + `phase8_trial.py --scripted-place`: policy picks and lifts 4 cm, then the script places (box in gripper from the template slot at the lift, IK, reading-based sag correction, 3 mm clearance, slow last 2 cm, open to 32, straight up). Top yaw from `fit_box` on the front image.
- Botmind batch: `scripts/b4_twin_scripted.ps1` (20 real trials x 4 runs), logs `logs\tsp_*`. Local check 4/4 placed, release < 0.5 mm, 2-3 mm clearance.
14:20: scripted place v3 in sim twin: 78/78 placed (2 policy pick timeouts), landed 2.9 mm mean / 7.2 max, stack moved 0. Waiting for the user for the first real trials (`--scripted-place`).
16:10: real stage 2 with the scripted place: 10/10 (landed 1-7 mm off centre). Open: false collapse call t6; no10 choice (data); stage 3.
18:05: camera layer proven on hardware (soak 17:58 all PASS incl. a real wrist unplug/re-plug, recovered by itself in 13 s). Next: stage-3 trials with --look.
18:50: stage 3 with the look 8/8. Next: collapse check false calls on 3-4 box stacks (4 real cases logged with settle videos + features).

## CHECKPOINT 2026-10-29 ~19:30 (user commuting; nothing running on the Mac, botmind or the PC)
Done today (all in notes/phase8.md §18-23, Decisions log): grip floor 28 (pick fixed), scripted place (stage 2 10/10), wrist look with two-look agreement (stage 3 8/8), crash-proof camera layer (soak all PASS incl. a real unplug), turn check v2 (real replay 0/56 false calls, 1/26 missed).
NEXT, in order (pick up exactly here):
1. Turn check v2 not yet wired in: check it on the sim collapse set (perception_eval / Phase 7 sets: 126 collapses, 274 standing) and the recorded clips (inputs/videos, Phase 7 §14 set) so it does not regress there; maybe tighten for the leaning-box miss (201746 t2).
2. Wire `turn_check` into `scripts/phase8_trial.py` (log both old and new calls side by side first) and `game/controller.py` (JUDGE after the settle window; keep drop_check and hand pause; the "is the new box on the stack?" y/n question only when turn_check is unsure).
3. Real check: a few trials logging both calls, then a sim game run (scripts/sim_game.py) for the controller.
4. Then: the next data run (no10 choice for small-on-large, background-behind-arm and cable randomization so --plain-wall is not needed, sim squeeze closer to real), SmolVLA comparison after ACT is reliable.
Open hardware item for the user: power the Orbbec wrist camera properly (powered USB 3 hub or USB-C 1.5 A); the software survives its resets but each costs a few seconds.
Commands in use: `phase8_trial.py ... --stage N --policy data/policies/act_wcal_060000 --plain-wall --grip-floor 28 --scripted-place --look`.
Nothing committed since 4e9c0b3 (user has not asked).

## 2026-10-30 00:50: turn check done (checkpoint steps 1-3)
turn_check wired into phase8_trial.py and game/controller.py; real 0/56 false, 1/26 missed; sim set 3/218, 6/76; clips 11/11; sim games 62/62 (0/53 false, 9/9 losses).
NEXT: (a) give the controller's robot turn the Phase 8 recipe (grip floor 28, plain wall, wrist look, scripted place; cameras already via RealArm) so a real game plays like the trials; (b) a real game; (c) data run (no10 choice, backgrounds, squeeze); (d) SmolVLA comparison.

## 2026-10-30 ~05:30: controller has the Phase 8 robot recipe
Robot turn in game/controller.py = wrist look + policy pick (grip floor 28, plain wall on the real arm) + scripted place + open_before_home; turn_check judges; soft servos in the sim game backend. Sim games (soft servos): looks 31/31, robot 24/31, calls 60/63. Real game NOT yet run.
NEXT: (a) first real game: `play_game.py real --port ...` (window UI: SPACE after the human's placement, e/Esc = e-stop, q = quit); (b) data run (no10 choice, background/cable randomization, squeeze, tall-stack knocks); (c) SmolVLA.
