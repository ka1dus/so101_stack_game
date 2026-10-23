# Phase 6: ACT v1 checkpoint evaluations (sim, 60 episodes each, seed 777, nominal visuals)

## step 010000 (2026-10-22 05:58)
```
policy data/checkpoints/act_v1/010000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 2/60 (3.3 %)
failures: {"drop": 7, "knocked_approach": 1, "misplaced": 5, "missed_grasp": 45}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 7/60
by stack height: 1 0/4 (0%), 2 0/8 (0%), 3 1/9 (11%), 4 1/16 (6%), 5 0/8 (0%), 6 0/8 (0%), 7 0/7 (0%)
by height group: 0-3 1/21 (5%), 4+ 1/39 (3%)
by crookedness: <10mm 0/39 (0%), <20mm 0/6 (0%), <5mm 2/15 (13%)
by picked type (expected): no10 0/8 (0%), no3 1/19 (5%), no35 1/33 (3%)
by pairing: no10_on_no3 0/2 (0%), no10_on_no35 0/6 (0%), no35_on_no10 0/6 (0%), no35_on_no3 1/12 (8%), no35_on_no35 0/15 (0%), no3_on_no10 0/5 (0%), no3_on_no3 0/5 (0%), no3_on_no35 1/9 (11%)
by height 4+ and crooked >=10mm: no 2/55 (4%), yes 0/5 (0%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 020000 (2026-10-22 06:35)
```
policy data/checkpoints/act_v1/020000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 7/60 (11.7 %)
failures: {"drop": 3, "knocked_approach": 4, "knocked_release": 1, "misplaced": 2, "missed_grasp": 43}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 3/60
by stack height: 1 2/4 (50%), 2 1/8 (12%), 3 0/9 (0%), 4 3/16 (19%), 5 1/8 (12%), 6 0/8 (0%), 7 0/7 (0%)
by height group: 0-3 3/21 (14%), 4+ 4/39 (10%)
by crookedness: <10mm 3/39 (8%), <20mm 0/6 (0%), <5mm 4/15 (27%)
by picked type (expected): no10 0/8 (0%), no3 1/19 (5%), no35 6/33 (18%)
by pairing: no10_on_no3 0/2 (0%), no10_on_no35 0/6 (0%), no35_on_no10 0/6 (0%), no35_on_no3 2/12 (17%), no35_on_no35 4/15 (27%), no3_on_no10 0/5 (0%), no3_on_no3 0/5 (0%), no3_on_no35 1/9 (11%)
by height 4+ and crooked >=10mm: no 7/55 (13%), yes 0/5 (0%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 030000 (2026-10-22 07:14)
```
policy data/checkpoints/act_v1/030000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 7/60 (11.7 %)
failures: {"drop": 2, "misplaced": 2, "missed_grasp": 49}
arm touched the backdrop (safety flag): 1/60
wrong box picked: 2/60
by stack height: 1 1/4 (25%), 2 2/8 (25%), 3 1/9 (11%), 4 1/16 (6%), 5 0/8 (0%), 6 1/8 (12%), 7 1/7 (14%)
by height group: 0-3 4/21 (19%), 4+ 3/39 (8%)
by crookedness: <10mm 3/39 (8%), <20mm 0/6 (0%), <5mm 4/15 (27%)
by picked type (expected): no10 2/8 (25%), no3 2/19 (11%), no35 3/33 (9%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 1/6 (17%), no35_on_no10 0/6 (0%), no35_on_no3 2/12 (17%), no35_on_no35 1/15 (7%), no3_on_no10 0/5 (0%), no3_on_no3 1/5 (20%), no3_on_no35 1/9 (11%)
by height 4+ and crooked >=10mm: no 7/55 (13%), yes 0/5 (0%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 040000 (2026-10-22 07:50)
```
policy data/checkpoints/act_v1/040000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 20/60 (33.3 %)
failures: {"drop": 1, "knocked_approach": 3, "misplaced": 2, "missed_grasp": 34}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 5/60
by stack height: 1 4/4 (100%), 2 2/8 (25%), 3 4/9 (44%), 4 4/16 (25%), 5 3/8 (38%), 6 2/8 (25%), 7 1/7 (14%)
by height group: 0-3 10/21 (48%), 4+ 10/39 (26%)
by crookedness: <10mm 11/39 (28%), <20mm 2/6 (33%), <5mm 7/15 (47%)
by picked type (expected): no10 4/8 (50%), no3 6/19 (32%), no35 10/33 (30%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 3/6 (50%), no35_on_no10 1/6 (17%), no35_on_no3 4/12 (33%), no35_on_no35 5/15 (33%), no3_on_no10 1/5 (20%), no3_on_no3 2/5 (40%), no3_on_no35 3/9 (33%)
by height 4+ and crooked >=10mm: no 19/55 (35%), yes 1/5 (20%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 050000 (2026-10-22 08:25)
```
policy data/checkpoints/act_v1/050000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 21/60 (35.0 %)
failures: {"drop": 2, "knocked_approach": 1, "misplaced": 4, "missed_grasp": 32}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 9/60
by stack height: 1 4/4 (100%), 2 1/8 (12%), 3 3/9 (33%), 4 6/16 (38%), 5 4/8 (50%), 6 3/8 (38%), 7 0/7 (0%)
by height group: 0-3 8/21 (38%), 4+ 13/39 (33%)
by crookedness: <10mm 13/39 (33%), <20mm 1/6 (17%), <5mm 7/15 (47%)
by picked type (expected): no10 1/8 (12%), no3 8/19 (42%), no35 12/33 (36%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 0/6 (0%), no35_on_no10 2/6 (33%), no35_on_no3 4/12 (33%), no35_on_no35 6/15 (40%), no3_on_no10 1/5 (20%), no3_on_no3 3/5 (60%), no3_on_no35 4/9 (44%)
by height 4+ and crooked >=10mm: no 21/55 (38%), yes 0/5 (0%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 060000 (2026-10-22 09:01)
```
policy data/checkpoints/act_v1/060000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 17/60 (28.3 %)
failures: {"drop": 2, "knocked_approach": 3, "misplaced": 1, "missed_grasp": 37}
arm touched the backdrop (safety flag): 1/60
wrong box picked: 1/60
by stack height: 1 1/4 (25%), 2 3/8 (38%), 3 5/9 (56%), 4 6/16 (38%), 5 2/8 (25%), 6 0/8 (0%), 7 0/7 (0%)
by height group: 0-3 9/21 (43%), 4+ 8/39 (21%)
by crookedness: <10mm 9/39 (23%), <20mm 1/6 (17%), <5mm 7/15 (47%)
by picked type (expected): no10 3/8 (38%), no3 2/19 (11%), no35 12/33 (36%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 2/6 (33%), no35_on_no10 0/6 (0%), no35_on_no3 4/12 (33%), no35_on_no35 8/15 (53%), no3_on_no10 0/5 (0%), no3_on_no3 0/5 (0%), no3_on_no35 2/9 (22%)
by height 4+ and crooked >=10mm: no 16/55 (29%), yes 1/5 (20%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 070000 (2026-10-22 09:37)
```
policy data/checkpoints/act_v1/070000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 24/60 (40.0 %)
failures: {"drop": 4, "knocked_approach": 2, "misplaced": 1, "missed_grasp": 29}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 6/60
by stack height: 1 2/4 (50%), 2 4/8 (50%), 3 4/9 (44%), 4 5/16 (31%), 5 6/8 (75%), 6 1/8 (12%), 7 2/7 (29%)
by height group: 0-3 10/21 (48%), 4+ 14/39 (36%)
by crookedness: <10mm 12/39 (31%), <20mm 2/6 (33%), <5mm 10/15 (67%)
by picked type (expected): no10 4/8 (50%), no3 5/19 (26%), no35 15/33 (45%)
by pairing: no10_on_no3 2/2 (100%), no10_on_no35 2/6 (33%), no35_on_no10 4/6 (67%), no35_on_no3 5/12 (42%), no35_on_no35 6/15 (40%), no3_on_no10 0/5 (0%), no3_on_no3 1/5 (20%), no3_on_no35 4/9 (44%)
by height 4+ and crooked >=10mm: no 22/55 (40%), yes 2/5 (40%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 080000 (2026-10-22 10:52)
```
policy data/checkpoints/act_v1/080000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 27/60 (45.0 %)
failures: {"drop": 2, "knocked_approach": 2, "misplaced": 1, "missed_grasp": 28}
arm touched the backdrop (safety flag): 1/60
wrong box picked: 5/60
by stack height: 1 1/4 (25%), 2 3/8 (38%), 3 5/9 (56%), 4 8/16 (50%), 5 4/8 (50%), 6 3/8 (38%), 7 3/7 (43%)
by height group: 0-3 9/21 (43%), 4+ 18/39 (46%)
by crookedness: <10mm 17/39 (44%), <20mm 3/6 (50%), <5mm 7/15 (47%)
by picked type (expected): no10 4/8 (50%), no3 8/19 (42%), no35 15/33 (45%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 3/6 (50%), no35_on_no10 1/6 (17%), no35_on_no3 4/12 (33%), no35_on_no35 10/15 (67%), no3_on_no10 2/5 (40%), no3_on_no3 2/5 (40%), no3_on_no35 4/9 (44%)
by height 4+ and crooked >=10mm: no 24/55 (44%), yes 3/5 (60%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 090000 (2026-10-22 11:11)
```
policy data/checkpoints/act_v1/090000/pretrained_model
mode act, seed 777, nominal visuals, n=60
success 23/60 (38.3 %)
failures: {"drop": 2, "knocked_approach": 2, "missed_grasp": 33}
arm touched the backdrop (safety flag): 2/60
wrong box picked: 5/60
by stack height: 1 4/4 (100%), 2 1/8 (12%), 3 5/9 (56%), 4 4/16 (25%), 5 4/8 (50%), 6 4/8 (50%), 7 1/7 (14%)
by height group: 0-3 10/21 (48%), 4+ 13/39 (33%)
by crookedness: <10mm 15/39 (38%), <20mm 2/6 (33%), <5mm 6/15 (40%)
by picked type (expected): no10 5/8 (62%), no3 5/19 (26%), no35 13/33 (39%)
by pairing: no10_on_no3 1/2 (50%), no10_on_no35 4/6 (67%), no35_on_no10 0/6 (0%), no35_on_no3 6/12 (50%), no35_on_no35 7/15 (47%), no3_on_no10 2/5 (40%), no3_on_no3 3/5 (60%), no3_on_no35 0/9 (0%)
by height 4+ and crooked >=10mm: no 21/55 (38%), yes 2/5 (40%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

## step 100000
Evaluated in the re-planning screen instead (same 60 states, seed 777): see data/eval/replan_100000_*/summary.txt and notes/phase6_overnight.md.

## step 100000 (2026-10-22 12:05)
```
policy data/checkpoints/act_v1/100000/pretrained_model
mode act, seed 777, nominal visuals, n=60
inference: n_action_steps=trained default (100), temporal_ensemble=None
success 25/60 (41.7 %)
failures: {"knocked_approach": 2, "misplaced": 2, "missed_grasp": 31}
arm touched the backdrop (safety flag): 0/60
wrong box picked: 4/60
by stack height: 1 4/4 (100%), 2 3/8 (38%), 3 3/9 (33%), 4 7/16 (44%), 5 3/8 (38%), 6 4/8 (50%), 7 1/7 (14%)
by height group: 0-3 10/21 (48%), 4+ 15/39 (38%)
by crookedness: <10mm 13/39 (33%), <20mm 4/6 (67%), <5mm 8/15 (53%)
by picked type (expected): no10 4/8 (50%), no3 4/19 (21%), no35 17/33 (52%)
by pairing: no10_on_no3 2/2 (100%), no10_on_no35 2/6 (33%), no35_on_no10 0/6 (0%), no35_on_no3 8/12 (67%), no35_on_no35 9/15 (60%), no3_on_no10 0/5 (0%), no3_on_no3 2/5 (40%), no3_on_no35 2/9 (22%)
by height 4+ and crooked >=10mm: no 21/55 (38%), yes 4/5 (80%)
eval states skipped for resting-stack drift > 0.5 mm in 2 s: 0
```

