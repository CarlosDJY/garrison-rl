"""
Boss-round data for the fight model after the simulation changed (leader skills): the boss-round boards already logged
(rounds 14 and 15, real and what-if) are fought again by the gym's --relabel with the current build, at their own
difficulty and at a random other one, and the 隐秘核心 (round 15) from 规程等级5. Output: data/fights_v2/*.jsonl, in
the --log-fights format plus "src" (the data file the board came from, so held-out files stay held out).

  python relabel_boss.py --limit 40000 --threads 18
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

from fight_model import DATA, ROOT, load_rows
from gym_client import GYM_EXE

OUT = os.path.join(ROOT, "python", "garrison_rl", "data", "fights_v2")
LEVELS = [1, 4, 6, 8, 11, 14, 16, 20]
PLANNER = ["--iterations", "2000", "--restarts", "2", "--sim", "200", "--no-flying"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000, help="boards to fight (after adding levels)")
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--chunk", type=int, default=5000, help="boards per gym call (one output file each)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(11)
    seen, jobs = set(), []
    for r in load_rows():
        if r["round"] < 14 or r["sim"]["Bosses"] <= 0:
            continue
        key = (r["boss"], r["map"], json.dumps(r["units"], ensure_ascii=False, sort_keys=True),
               json.dumps({k: round(v) for k, v in r["layers"].items()}, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        level = int(r["mode"].replace("规程等级", "") or 0)
        base = dict(hp=r["hp"], shopLevel=r["shopLevel"], fieldBonus=r["fieldBonus"], map=r["map"], boss=r["boss"], types=r["types"],
                    units=r["units"], layers=r["layers"], src=r["_file"])
        other = rng.choice([lv for lv in LEVELS if lv != level])
        for lv, rd in ((level, 14), (other, 14)) + (((level, 15),) if level >= 5 else ()):
            jobs.append(dict(base, mode=f"规程等级{lv}", round=rd, seed=rng.randrange(1_000_000)))
    rng.shuffle(jobs)
    jobs = jobs[: args.limit]
    print(f"{len(seen)} distinct boss boards; fighting {len(jobs)}", flush=True)
    for c in range(0, len(jobs), args.chunk):
        part = jobs[c:c + args.chunk]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        inp = os.path.join(OUT, f"_in_{stamp}.jsonl")
        out = os.path.join(OUT, f"boss_{stamp}.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            for j in part:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        t0 = time.time()
        cmd = [GYM_EXE, "--data", os.path.join(ROOT, "data"), "--envs", str(args.threads), "--relabel", inp, "--out", out] + PLANNER
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=subprocess.DEVNULL)
        os.remove(inp)
        print(f"chunk {c // args.chunk + 1}: {len(part)} boards in {time.time() - t0:.0f}s -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
