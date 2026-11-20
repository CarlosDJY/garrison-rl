"""
Reference clear rates in the real simulation (the placement search the fight data was made with, no flying, no
隐秘核心): the scripted 跟阵容 policy steering for the saved lineups (「我的阵容」), and optionally PPO checkpoints, on the
same held-out seeds per difficulty.

  python expert_eval.py --levels 1,4,6,8,11,14 --runs 48 [--policy runs/ppo_sur2/best_规程等级4.pt ...]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from gym_client import MID_PLANNER, ROOT, GymClient
from model import Encoder, Policy, load_compatible, to_torch


def play(mode, runs, seed, model=None, enc=None, device=None, plan=False, known=False, plan_extra=()):
    gym = GymClient(mode=mode, envs=runs, extra=MID_PLANNER + ["--no-flying", "--no-hidden-core"] + (["--plan", *plan_extra] if plan else []) + (["--plan-known"] if known else []))
    r = gym.reset(seed)
    res = [None] * runs
    while any(x is None for x in res):
        if model is None:
            acts = [-2] * runs
        else:
            from train_ppo import pad_actions
            with torch.no_grad():
                logits, _ = model(to_torch(pad_actions(enc.encode(r["obs"])), device))
            acts = logits.argmax(-1).cpu().numpy()
        r = gym.step(acts)
        for i, info in enumerate(r["info"]):
            if info and res[i] is None:
                res[i] = info
    gym.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,6,8,11,14")
    ap.add_argument("--runs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--policy", nargs="*", default=[])
    ap.add_argument("--plan", action="store_true", help="policies trained with the target lineup as input")
    ap.add_argument("--no-expert", action="store_true")
    ap.add_argument("--plan-lock", type=int, default=5, help="as train_mix.py (the policy must be evaluated as it was trained)")
    ap.add_argument("--plan-staged", action="store_true")
    ap.add_argument("--plan-known", action="store_true", help="the field is planned against the previewed enemy group")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "playground", "baseline.md"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    who = ([] if args.no_expert else [("跟阵容（脚本）", None)]) + [(os.path.relpath(p, os.path.join(ROOT, "python", "garrison_rl")), p) for p in args.policy]
    lines = ["# 参考通关率（真实模拟，生成战斗数据时的站位搜索，无飞行、无隐秘核心）\n",
             f"每格 {args.runs} 局，种子 {args.seed} 起。\n",
             "| 策略 | 难度 | 通关 | 平均回合 | 商店等级 | 精锐 | 用时 |", "|---|---|---|---|---|---|---|"]
    rows = []
    for name, path in who:
        model = enc = None
        for lv in args.levels.split(","):
            mode = f"规程等级{lv}"
            t0 = time.time()
            if path:
                if model is None:
                    g = GymClient(mode=mode, envs=1)
                    enc = Encoder(g.spec)
                    g.close()
                    model = Policy(enc).to(device)
                    load_compatible(model, torch.load(path, map_location=device))
                    model.eval()
            res = play(mode, args.runs, args.seed, model, enc, device, plan=args.plan and path is not None, known=args.plan_known,
                       plan_extra=["--plan-lock", str(args.plan_lock)] + (["--plan-staged"] if args.plan_staged else []))
            c = np.mean([x["cleared"] for x in res])
            line = (f"| {name} | {mode} | {c:.0%} | {np.mean([x['rounds'] for x in res]):.1f} | {np.mean([x['level'] for x in res]):.1f} | "
                    f"{np.mean([x['elites'] for x in res]):.1f} | {time.time() - t0:.0f}s |")
            lines.append(line)
            rows.append(dict(policy=name, mode=mode, clear=float(c)))
            print(line, flush=True)
            open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rows, open(args.out.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
