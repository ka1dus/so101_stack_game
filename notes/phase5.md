# Phase 5: Scripted expert, game states and data generation

Date: 2026-10-21. Status: **in progress, blocked on a supply-layout decision (section 2).**

## 1. `scripts/game_states.py`

- **Stack:** heights 1–7. The human starts, so the robot never places onto the empty
  base, and in a strict game its turns come onto stacks of 1, 3, 5, 7.
- **Box types per level** from the real mix (3 / 4 / 1).
- **Placement error:** per-level offsets (σ 4 mm, or 9 mm for "precarious" stacks, clipped
  at ± 20 mm) and yaw (σ 7°, or 15° on the top level of hard stacks).
- **Hard cases** over-sampled: a small top box (large onto small next), precarious stacks.
- **Standing check:** every stack is settled 0.4 s, then must not move > 1 mm or tilt > 2°
  over 0.6 s.
- **Supply:** non-overlapping, entirely on the arm side of the reach-line tape, mostly
  square to the mat (a loose grid like the real one), 2 mm minimum gaps.
- **Pick rule** (Section 3c): ACT = the nearest box to the stacking spot with room for the
  jaws; VLA = the nearest such box of the requested type. "Room for the jaws" = 10 mm free
  on the fixed-jaw side and 22 mm on the moving-jaw side (the expert puts the moving jaw on
  the roomier side).

## 2. Blocking finding: the supply can't hold enough boxes

(`notes/phase5_supply_layout_check.txt`)

With the reach-line tape, only ~61–66 % of the black mat is usable. Random layouts that
also leave one pickable box:

| boxes in the supply | mat as now | mat moved 4 cm toward the arm | 5 cm | 6 cm |
|---|---|---|---|---|
| 7 (robot's first turn) | **0 / 60** | 40 / 60 | 51 / 60 | 55 / 60 |
| 6 | **0 / 60** | 58 / 60 | 60 / 60 | 59 / 60 |
| 5 | 21 / 60 | 58 / 60 | 59 / 60 | 60 / 60 |

So with the mat where it is, the robot's first turns (7 or 6 boxes in the supply) usually
have no box it can pick. A straight 5 cm shift toward the pan axis puts two mat corners off
the table near the arm's corner, and grasps near the inner corner then make the arm touch
the table. A new position must be optimized: slid along the table's left edge, kept on the
table, clear of the arm base and the yellow area, with every part pickable.

Options put to the user:
A. Re-position the black mat (I compute the exact placement and verify it; you re-tape it
   and re-touch its 4 corners, ~5 min).
B. Keep the mat; at most 5 boxes on it at a time, the rest held in reserve off the mat for
   the human to refill.
C. Fewer boxes per game (6), which contradicts the decided 8.

## 3. Re-evaluation after user feedback (2026-10-21; `notes/phase5_supply_reevaluation.txt`)

User: all boxes fit on the mat, and every placement frees space for the next pick. Section
2 was **too pessimistic**:
1. **It tested random dense layouts.** A real game starts from a loose grid, and each
   removal frees the neighbours.
2. **Jaw clearance was over-strict** (22 mm along the whole long side). The jaws only touch
   the middle of the box (±1 cm along its length) and are 6–7 mm thick, so **10 mm free on
   each side near the middle** is enough with a tighter pre-grasp opening (width + 6 mm).
   This must be confirmed by the grasp test with neighbours present.

Still true: vertical picks only reach ~29 cm, so boxes beyond the tape can only be taken
by the human. Results with a 1 cm-gap grid of all 8 boxes, simulating games (human takes
a box, robot takes the nearest pickable one, alternating):

| set-up | fit | robot stuck |
|---|---|---|
| whole mat, human picks at random | 8/8 (2 beyond reach) | 167/300 games |
| **whole mat, human takes the beyond-tape boxes first** | 8/8 | **0/300** |
| **mat expanded 2 cm toward the arm** (strip along the corner-1/corner-4 edge) | 8/8 on the arm side | **0/300** (random human) |

Section 2's "move the mat" option is withdrawn.

## 4. Decision: expand the black mat 2 cm toward the arm (user, 2026-10-21)

- **What:** a 2 cm wide, 19.2 cm long black matte strip along the mat's edge between
  corner 1 (nearest the arm base) and corner 4 (nearest the yellow area), on the outside,
  toward the arm. The new mat is 22.9 × 19.2 cm (`notes/img/phase5_mat_expansion.png`).
- **Unchanged:** corners 2 and 3 and the reach-line tape. The tape's ends stay put; the end
  on edge 1→2 is now 16.7 cm from the new corner 1.
- **Re-touch:** the 4 supply corners with `so101_record_points.py --points supply`
  → `inputs/supply_points_<date>.json`. `scene_layout.py` uses the newest such file and
  the expanded size.
