"""
A trained policy in the real simulation with and without a buying constraint (--plan-only N: from round N only the
locked target lineup's members can be bought), on the same seeds.

  python mask_eval.py --init runs/mix6/last.pt --levels 8,11,14 --runs 96 --only 0,5
"""
import argparse
import json

import numpy as np
import torch

from gym_client import MID_PLANNER, GymClient
from model import Encoder, Policy, load_compatible, to_torch
from train_ppo import pad_actions


def play(model_of, mode, runs, seed, extra, device):
    gym = GymClient(mode=mode, envs=runs, extra=extra)
    enc, model = model_of(gym.spec)
    r = gym.reset(seed)
    res = [None] * runs
    with torch.no_grad():
        while any(x is None for x in res):
            logits, _ = model(to_torch(pad_actions(enc.encode(r["obs"])), device))
            r = gym.step(logits.argmax(-1).cpu().numpy())
            for i, info in enumerate(r["info"]):
                if info and res[i] is None:
                    res[i] = info
    gym.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--levels", default="8,11,14")
    ap.add_argument("--runs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=400000000)
    ap.add_argument("--only", default="0,5", help="--plan-only values to compare (0: no constraint)")
    ap.add_argument("--out", default="runs/mask_eval.jsonl")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.init, map_location=device)
    cache = {}

    def model_of(spec):
        if "m" not in cache:
            enc = Encoder(spec)
            model = Policy(enc).to(device)
            load_compatible(model, state)
            model.eval()
            cache["m"] = (enc, model)
        return cache["m"]

    for lv in args.levels.split(","):
        for only in args.only.split(","):
            extra = MID_PLANNER + ["--no-flying", "--no-hidden-core", "--plan", "--plan-lock", "1", "--plan-staged"]
            if int(only) > 0:
                extra += ["--plan-only", only]
            res = play(model_of, f"规程等级{lv}", args.runs, args.seed, extra, device)
            row = dict(level=int(lv), only=int(only), n=len(res),
                       clear=float(np.mean([x["cleared"] for x in res])), rounds=float(np.mean([x["rounds"] for x in res])),
                       top=float(np.mean([(x.get("layers") or [0])[0] for x in res])),
                       elites=float(np.mean([x["elites"] for x in res])),
                       held=float(np.mean([x["plan"]["held"] for x in res if x.get("plan")])),
                       elite=float(np.mean([x["plan"]["elite"] for x in res if x.get("plan")])),
                       plan_layers=float(np.mean([x["plan"]["layers"] for x in res if x.get("plan")])),
                       gain9_13=float(np.mean([sum(x["gains"][8:13]) for x in res])))
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
