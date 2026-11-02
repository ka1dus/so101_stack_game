# LinkedIn post draft (numbers checked against notes/phase8.md and notes/phase10.md, 30 Oct 2026)

## Post

I taught a robot arm to play a stacking game against me, without showing it a single real demonstration.

The game is simple. We take turns putting a staple box on a shared stack. Three box sizes are mixed together. If you knock the stack over or drop a box, you lose.

The rule I set myself: all the learning happens in simulation. I built a MuJoCo copy of my desk, arm and cameras. A scripted "expert" then stacked boxes there more than 10,000 times while I trained an ACT policy (LeRobot) on the recordings. The real setup was only used to measure things and to test.

What runs on the real SO-101 arm:
- a learned policy that picks the next box using two cameras
- a wrist-camera look at the stack, then a computed placement onto the top box
- classical computer vision that judges every turn: did the box land, did the stack move, is a hand in the way
- turn-taking, a pause when a hand comes in, and an emergency stop

Results so far:
- 30 of 30 real placement trials across three stages (same-size stacks, mixed pairs, crooked 3–4 box stacks)
- 0 false "collapse" calls in 56 real placements
- first real games: the robot placed 8 of 10 boxes, onto stacks up to 6 high

The policy was rarely the problem. The real issues were:
- the room behind the arm in the camera view (a banner made it stop 1 cm short)
- servos much softer than the simulated ones
- a wrist camera resetting on an underpowered USB port
- a box sliding off when the gripper opened

Each one meant measuring the real arm and fixing the simulation or the code, never adding real training data.

Next: telling the robot in plain language which box to pick, with a vision-language-action model (SmolVLA).

[video]

#robotics #simtoreal #reinforcementlearning #imitationlearning #lerobot #mujoco

## Shorter version

My robot arm plays a stacking game against me. It never saw a real demonstration: all the training happened in a MuJoCo simulation of my desk (10,000+ simulated demos, ACT policy, LeRobot).

On the real arm: 30/30 placement trials, 0 false collapse calls in 56 placements, 8 of 10 boxes placed in its first real games.

The hard part wasn't the AI. It was making the simulation match reality: camera background, soft servos, a flaky USB camera.

Next: telling it which box to pick, in plain language.

[video]

## Clips to record (next real games)

1. **Hero clip, 20–40 s:** one full robot turn on a 4–5 box stack: look at the stack, pick, place, return home. Front view, steady, good light.
2. **Game flow, 30–60 s, sped up 2–4×:** a whole game with turns alternating, ending on the result screen ("You win!" / "Robot wins").
3. **Close-up of the release:** phone camera at stack height as the box goes down onto a crooked stack.
4. **Safety:** a hand reaches into the workspace mid-turn, the arm freezes, then carries on after the hand leaves.
5. **Sim next to real:** the same turn in MuJoCo and on the real arm side by side. Can be rendered from the sim; ask me.
6. **Screen capture of the game window:** banner, live camera view, stack overlay, the y/n question.
7. **Optional, honest, often the most liked:** a robot failure (the box slides off), with the game correctly calling it.

## Accuracy notes (keep these true when editing)
- "30 of 30 real placement trials" = stage 1 12/12, stage 2 10/10, stage 3 8/8, each with the box the trial expected.
- "8 of 10 boxes in first real games" = robot turns across the 30 Oct games (5/6 + 3/4). The highest successful
  placement was onto a 5-box stack (making it 6 high). The 7-box attempt failed.
- The place step is computed, not learned. Say "picks using a learned policy", not "learned to stack" on its own.
- "10,000+ simulated demonstrations" = about 11,600 episodes used across the v2, targeted, trimmed-backdrop,
  soft-servo and calibrated-wrist datasets.
- Don't mention SmolVLA as done until it is trained and tested.

## Single-post version (763 words)

I built a robot arm that plays a stacking game against me, and it never saw a single real demonstration.

The game: we take turns putting staple boxes on a shared stack. Three box sizes, mixed. Drop a box or knock the stack over and you lose. The robot has to pick a box out of a tightly packed supply, place it on a stack a person built slightly crooked, and the game judges every turn from a camera.

My one rule: all robot learning happens in simulation (MuJoCo). The real setup is only for measuring and testing. So when the real robot misbehaves, the fix has to go into the simulation or the code, never into more real data.

𝗛𝗼𝘄 𝗶𝘁'𝘀 𝗯𝘂𝗶𝗹𝘁

1. Learn one skill, code the rest.
The robot learns to pick the next box (an ACT policy in LeRobot, two cameras). The placement is computed with inverse kinematics. Turn-taking, judging and safety are classical code. Success targets were set up front and every test is reported by box size and size pairing, so a failure points straight at its cause.

2. A digital twin built by measuring.
- Joint conversion verified against LeRobot's own code, with offsets fitted from real touch points
- Cameras calibrated against real images (2.6 px and 5.1 px error)
- Box physics tuned until simulated topple limits matched real tests within 1.5 mm for every pairing
- A full sweep of arm poses to find the best stacking spot for an 8-box stack

3. A scripted teacher designed to be imitated.
A planner with perfect knowledge of the simulation generated 11,600 demonstrations:
- a fixed rule for which box to take, so the policy never hesitates between two
- placement chosen for the best balance of the whole stack, not just the top box
- slow final approach, straight-up retract, safety margins around obstacles
- heavy randomisation of cameras, lighting, colours and box prints
Generation ran in parallel on a 20-core workstation at about 2 s per demo.

4. Train, diagnose, fix the weakest case.
The first policy reached 49% in simulation. Sorting the failures showed 92 of 102 were missed grasps, so I added recovery demos (noise injected into the motion, clean labels), retry demos and a more forgiving grasp. Next version: 93%. A 1,060-run confirmation test with confidence bounds found the real weak spot (big boxes on tall, crooked stacks), and targeted data closed it. The third policy met every simulation target.

5. Judge the game reliably.
A game that wrongly says "you knocked it over" is unplayable. The turn check fits the expected box on the stack and looks for changes anywhere else, hides the robot arm using the simulated twin, and asks the player when it isn't sure. False collapse calls on real placements: 18 of 56 in the first version, 0 of 56 now.

𝗖𝗹𝗼𝘀𝗶𝗻𝗴 𝘁𝗵𝗲 𝘀𝗶𝗺-𝘁𝗼-𝗿𝗲𝗮𝗹 𝗴𝗮𝗽

Each real failure got a measured cause and a fix:
- Soft servos: real ones were 30–75× softer than simulated. Fitted a servo model and taught the planner to compensate for sag. Placing a held box: 74% → 90%.
- Releases 2 cm too high: the wrist camera model was off by 21 mm. Re-calibrated it with a ChArUco board and retrained.
- Hovering 1 cm above the stack: replaying the run offline and swapping parts of the image showed the cause was the room behind the arm. Masked it now; background randomisation goes in next.
- Placements 1–3 cm off: kept learning where it's strong (finding and grasping a box) and computed the placement from the known box-in-gripper position.
- Where exactly is the stack top? The arm now looks down with its wrist camera first and only trusts the result when two looks agree within 6 mm. In a real game it correctly rejected a look that was 13 mm off.
- A camera that kept dropping out: the kernel log showed a USB power shortfall. Each camera now runs in its own supervised process that restarts itself. It passed fault-injection tests and a test where I unplugged the camera mid-run.

𝗥𝗲𝘀𝘂𝗹𝘁𝘀 𝗼𝗻 𝘁𝗵𝗲 𝗿𝗲𝗮𝗹 𝗮𝗿𝗺
- 30/30 placement trials (same-size stacks, mixed pairs, crooked 3–4 box stacks)
- 0/56 false collapse calls
- First real games against a person: 8 of 10 robot placements

The biggest lesson: the learning was rarely the problem. Matching the simulation to reality was the real work: cameras, servos, backgrounds, power.

Next: telling the robot which box to pick in plain language, with a vision-language-action model (SmolVLA).

#robotics #simtoreal #imitationlearning #lerobot #mujoco #robotlearning

## Single-post version, under the 3,000-character limit (2,255 characters)

I built a robot arm that plays a stacking game against me, and it never saw a single real demonstration.

We take turns placing staple boxes (three sizes, mixed) on a shared stack. Drop one or knock the stack over and you lose.

My rule: all robot learning happens in simulation (MuJoCo). The real setup is only for measuring and testing, so every real-world problem had to be fixed in the simulation or the code, never with real data.

𝗛𝗼𝘄 𝗶𝘁 𝘄𝗼𝗿𝗸𝘀
- Learn one skill, code the rest: an ACT policy (LeRobot) picks the box, inverse kinematics places it, and classical code runs the game.
- A digital twin built by measuring: joints, cameras and box physics calibrated against the real setup (topple limits within 1.5 mm).
- A scripted teacher generated 11,600 demos, with a fixed box-choice rule so the policy never hesitates, stability-aware placement, and heavy randomisation of cameras, lighting and colours.
- Train, diagnose, fix: v1 hit 49%. 92 of 102 failures were missed grasps, so I added recovery and retry demos: 93%. A 1,060-run confidence test found the last weak spot (tall, crooked stacks); targeted data closed it.
- A camera-based judge that hides the arm using the twin and asks the player when unsure. False collapse calls: 18/56 → 0/56.

𝗖𝗹𝗼𝘀𝗶𝗻𝗴 𝘁𝗵𝗲 𝘀𝗶𝗺-𝘁𝗼-𝗿𝗲𝗮𝗹 𝗴𝗮𝗽
- Real servos were 30–75× softer than simulated: fitted a servo model and compensated for sag (74% → 90%).
- Releases 2 cm too high: recalibrated the wrist camera with a ChArUco board.
- Hovering 1 cm short: an offline image-swap replay traced it to the room behind the arm.
- Placements 1–3 cm off: kept learning for the grasp and computed the placement.
- A wrist-camera look at the stack, trusted only when two looks agree within 6 mm.
- A camera dropping out from a USB power shortfall: supervised, self-restarting camera processes, proven by unplugging it mid-run.

𝗥𝗲𝘀𝘂𝗹𝘁𝘀 𝗼𝗻 𝘁𝗵𝗲 𝗿𝗲𝗮𝗹 𝗮𝗿𝗺
- 30/30 placement trials
- 0/56 false collapse calls
- 8 of 10 robot placements in its first games against a person

The lesson: the learning was rarely the problem. Matching simulation to reality was the real work.

Next: telling it which box to pick in plain language, with a vision-language-action model (SmolVLA).

#robotics #simtoreal #imitationlearning #lerobot #mujoco
