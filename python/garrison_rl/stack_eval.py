"""
Scripted experts side by side in the real simulation (the mid placement search, no flying, no 隐秘核心), on the same
seeds: clear rate, and how the layers were stacked — the top covenant's layers at the end, the layers gained per round,
the HP each part of the run cost.

  python stack_eval.py --kinds lineup,stack,stack-balanced --levels 8,14 --runs 48
"""
import argparse
import json
import os
import time

import numpy as np

from gym_client import MID_PLANNER, ROOT, GymClient


def play(mode, runs, seed, kind):
    # "kind+known": the field is planned against the previewed enemy group (--plan-known).
    kind, _, flag = kind.partition("+")
    gym = GymClient(mode=mode, envs=runs, extra=MID_PLANNER + ["--no-flying", "--no-hidden-core", "--expert-kind", kind] + (["--plan-known"] if flag == "known" else []))
    gym.reset(seed)
    res = [None] * runs
    while any(x is None for x in res):
        r = gym.step([-2] * runs)
        for i, info in enumerate(r["info"]):
            if info and res[i] is None:
                res[i] = info
    gym.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="lineup,stack,stack-balanced")
    ap.add_argument("--levels", default="8,14")
    ap.add_argument("--runs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=400000000)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "playground", "stack_eval.md"))
    args = ap.parse_args()
    lines = ["# 脚本专家对比（真实模拟，中等站位搜索，无飞行、无隐秘核心）\n", f"每格 {args.runs} 局，种子 {args.seed} 起。\n",
             "| 专家 | 难度 | 通关 | 平均回合 | 精锐 | 最高盟约层数 中位/p90 | 前三盟约合计 中位 | 每回合+层：回合3-5 / 6-8 / 9-11 / 12-13 | 掉血：回合1-8 / 9-13 | 用时 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for lv in args.levels.split(","):
        mode = f"规程等级{lv}"
        for kind in args.kinds.split(","):
            t0 = time.time()
            res = play(mode, args.runs, args.seed, kind)
            top = [(x["layers"] or [0])[0] for x in res]
            top3 = [sum(x["layers"] or [0]) for x in res]

            def span(key, a, b):
                v = [g for x in res for g in x[key][a - 1:b]]
                return float(np.mean(v)) if v else 0.0

            def lost(a, b):
                return float(np.mean([sum(x["losses"][a - 1:b]) for x in res]))

            row = dict(kind=kind, mode=mode, clear=float(np.mean([x["cleared"] for x in res])), rounds=float(np.mean([x["rounds"] for x in res])),
                       elites=float(np.mean([x["elites"] for x in res])), top50=float(np.median(top)), top90=float(np.percentile(top, 90)),
                       top3=float(np.median(top3)), gains=[span("gains", 3, 5), span("gains", 6, 8), span("gains", 9, 11), span("gains", 12, 13)],
                       loss=[lost(1, 8), lost(9, 13)])
            rows.append(row)
            line = (f"| {kind} | {mode} | {row['clear']:.0%} | {row['rounds']:.1f} | {row['elites']:.1f} | {row['top50']:.0f} / {row['top90']:.0f} | {row['top3']:.0f} | "
                    + " / ".join(f"{g:.0f}" for g in row["gains"]) + f" | {row['loss'][0]:.1f} / {row['loss'][1]:.1f} | {time.time() - t0:.0f}s |")
            lines.append(line)
            print(line, flush=True)
            open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rows, open(args.out.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
