"""Phase 9: play the stacking game (game/controller.py) against one of three worlds.

    # real arm and cameras (the arm at home first: scripts/so101_go_home.py); eval venv
    ~/.venvs/so101_eval/bin/python scripts/play_game.py real --port /dev/tty.usbmodem5B910311481
    # same, but the robot's turn runs the policy without motion and you place its box by hand (stand-in)
    ~/.venvs/so101_eval/bin/python scripts/play_game.py real --port /dev/tty.usbmodem5B910311481 --no-motion
    # full games in MuJoCo through the same controller (scripted human; headless unless --show)
    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/play_game.py sim --games 10
    # one human turn replayed from a recorded clip (SPACE pressed at --space-s)
    ~/.venvs/so101_eval/bin/python scripts/play_game.py clip --clip human_turn_1 --space-s 9.0 --stack no3 no35 --placed no10
Logs: data/games/<backend>_<time>/ (per game: turns.jsonl, game.json, front.mp4, frames).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.controller import Controller  # noqa: E402
from ui.display import UI, NullUI  # noqa: E402

POLICY = ROOT / "data/policies/act_wcal_060000"   # the Phase 8 trial policy (calibrated wrist camera, soft servos)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("backend", choices=("real", "sim", "clip"))
    ap.add_argument("--policy", default=str(POLICY)); ap.add_argument("--device", default="mps")
    ap.add_argument("--port"); ap.add_argument("--id", default="my_follower_arm")
    ap.add_argument("--front", type=int, default=0); ap.add_argument("--wrist", type=int, default=1)
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--games", type=int, default=1); ap.add_argument("--seed", type=int, default=12001)
    ap.add_argument("--show", action="store_true", help="sim: open the game window too")
    ap.add_argument("--clip"); ap.add_argument("--space-s", type=float); ap.add_argument("--stack", nargs="*", default=[])
    ap.add_argument("--placed")
    a = ap.parse_args()
    out = ROOT / "data/games" / f"{a.backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    from skill.runner import load_policy
    if a.backend == "real":
        from game.backends import RealBackend
        if not a.port:
            sys.exit("--port is required")
        policy = load_policy(a.policy, a.device); b = RealBackend(a.port, a.id, a.front, a.wrist, no_motion=a.no_motion)
        try:
            r = Controller(b, policy, UI(), out, no_motion=a.no_motion).play()
        finally:
            b.close()
        print(json.dumps({k: v for k, v in r.items() if k != "refit"}))
    elif a.backend == "clip":
        from game.backends import ClipBackend
        b = ClipBackend(ROOT / f"inputs/videos/phase7/{a.clip}.mp4", ROOT / "inputs/photos/phase7/front_cam_ref_supply_full.png", a.placed)
        keys = [None] * int(a.space_s * 30) + ["space"]
        r = Controller(b, None, NullUI(keys, then="space"), out, initial_stack=a.stack, threaded_hand=False, max_turns=1).play()
        print(json.dumps(json.loads((out / "game.json").read_text())["turns"][0], indent=1))
    else:
        from game.backends import SimBackend
        policy = load_policy(a.policy, a.device); b = SimBackend(a.policy, a.device, a.seed); games = []
        for g in range(a.games):
            t0 = time.time()
            r = Controller(b, policy, UI() if a.show else NullUI(), out / f"game_{g:03d}", threaded_hand=False).play()
            turns = json.loads((out / f"game_{g:03d}" / "game.json").read_text())["turns"]; games.append(turns)
            print(f"game {g}: {r['why']} (loser {r['loser']}), {len(turns)} turns, {(time.time() - t0) / 60:.1f} min", flush=True)
        turns = [t for g in games for t in g]
        c = Counter((t["truth"] != "ok", t["call"] != "ok") for t in turns)
        lines = [f"games {len(games)}, turns {len(turns)} (robot {sum(t['player'] == 'robot' for t in turns)})",
                 f"turn calls correct {sum(t['call_ok'] for t in turns)}/{len(turns)}; losses caught {c[(True, True)]}/{c[(True, True)] + c[(True, False)]};"
                 f" false loss calls {c[(False, True)]}/{c[(False, True)] + c[(False, False)]}",
                 f"asked type {sum(t['asked'] for t in turns)}, asked on-stack {sum(t.get('asked_on_stack') is not None for t in turns)}, watchdogs {sum(t.get('watchdogs', 0) for t in turns)}; robot skill events {dict(Counter(t.get('skill_event') for t in turns if t['player'] == 'robot'))}",
                 f"robot pauses for a hand {sum(t.get('pauses', 0) for t in turns)}; settle restarts {sum(t.get('settle_restarts', 0) for t in turns)}",
                 f"games called right {sum(all(t['call_ok'] for t in g) for g in games)}/{len(games)}"]
        txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
