"""
The game AI of Stronghold-Protocol ("legacy") plays whole solo 规程 matches; every shop call it makes is written as the
observation before it with the action's index (behaviour cloning data), and every real fight as a row (fight model data).

  python sp_record.py --levels 4,8,11,14,16,20 --matches 400 --procs 16 --seed 1000000 --tag a
"""
import argparse
import json
import os
import subprocess
import time

from gym_client import SP_GYM, SP_NODE, SP_VERSION

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,8,11,14,16,20")
    ap.add_argument("--matches", type=int, default=400, help="per level")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1_000_000)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--rehearsal", type=int, default=-1, help="layouts rehearsed per prep; with it every candidate layout and its simulated result is written (layouts_*.jsonl)")
    ap.add_argument("--per-job", type=int, default=60, help="matches per process (one file each; file 0 of a level is held out)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for lv in args.levels.split(","):
        per = args.per_job
        k, left = 0, args.matches
        while left > 0:
            n = min(per, left)
            jobs.append((int(lv), k, n, args.seed + int(lv) * 100_000 + k * 1000))
            left -= n
            k += 1
    t0 = time.time()
    stats = {}

    def run(job):
        lv, k, n, seed = job
        bc = os.path.join(OUT, f"bc_{args.tag}_L{lv}_{k}.jsonl")
        fights = os.path.join(OUT, f"fights_{args.tag}_L{lv}_{k}.jsonl")
        for f in (bc, fights):
            if os.path.exists(f):
                os.remove(f)
        p = subprocess.Popen([SP_NODE, SP_GYM, "--version", SP_VERSION, "--mode", f"规程等级{lv}", "--envs", "1", "--fights", fights]
                             + (["--rehearsal", str(args.rehearsal), "--layouts", os.path.join(OUT, f"layouts_{args.tag}_L{lv}_{k}.jsonl")] if args.rehearsal >= 0 else []),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=os.path.dirname(SP_GYM))
        cmds = "".join(json.dumps(c) + chr(10) for c in ({"cmd": "record", "matches": n, "seed": seed, "out": bc}, {"cmd": "close"}))
        out, _ = p.communicate(cmds.encode())
        return job, out.decode("utf-8")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(args.procs) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for n_done, fut in enumerate(as_completed(futures), 1):
            job, text = fut.result()
            for line in text.splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if "recorded" in d:
                    s = stats.setdefault(job[0], dict(matches=0, wins=0, recorded=0, unrepresented=0, elites=[], top=[]))
                    for key in ("matches", "wins", "recorded", "unrepresented"):
                        s[key] += d[key]
                    s["elites"] += [i["elites"] for i in d["info"]]
                    s["top"] += [(i["layers"] or [0])[0] for i in d["info"]]
            print(f"{n_done}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    for lv, s in sorted(stats.items()):
        print(f"L{lv}: {s['matches']} matches, clear {s['wins'] / max(1, s['matches']):.0%}, {s['recorded']} decisions ({s['unrepresented']} not representable), "
              f"elites {sum(s['elites']) / max(1, len(s['elites'])):.1f}, top layers {sum(s['top']) / max(1, len(s['top'])):.0f}")


if __name__ == "__main__":
    main()
