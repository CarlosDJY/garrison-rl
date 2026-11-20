"""
Fight data for a learned fight model: plays runs in src/ASPAssistant.Gym with --log-fights, one batch at a time, each
with a random setting so the boards cover what training will meet:
  - difficulty: 规程等级1–20, mostly 4–14;
  - shop: the real one mostly, sometimes widened (the curriculum's boards);
  - policy: a trained checkpoint sampled with a temperature and a share of random actions, or a buy-heavy random policy.
Placement uses the middle planner (as training), so the logged outcome is what the model has to stand in for.

  python gen_fights.py --fights 300000            # until that many fights are logged (in data/fights/)
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch

from gym_client import GymClient, ROOT
from model import Encoder, Policy, to_torch
from train_ppo import pad_actions

OUT = os.path.join(ROOT, "python", "garrison_rl", "data", "fights")
MODES = {1: 1, 4: 3, 6: 3, 8: 3, 11: 3, 14: 3, 16: 1, 20: 1}
EXTRA = {0: 7, 4: 1, 8: 1, 16: 1}
PLANNER = ["--iterations", "2000", "--restarts", "2", "--sim", "200", "--no-flying"]


def checkpoints():
    found = sorted(glob.glob(os.path.join(ROOT, "python", "garrison_rl", "runs", "*", "best_*.pt")))
    return found


def logged():
    n = 0
    for f in glob.glob(os.path.join(OUT, "*.jsonl")):
        with open(f, "rb") as fh:
            n += sum(1 for _ in fh)
    return n


def pick(weights):
    keys = list(weights)
    return random.choices(keys, [weights[k] for k in keys])[0]


def load(path, enc, device):
    m = Policy(enc).to(device)
    own = m.state_dict()
    sd = {k: v for k, v in torch.load(path, map_location=device).items() if k in own and own[k].shape == v.shape}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def random_action(o, kinds):
    names = [kinds[a[0]] for a in o["acts"]]
    for want, p in (("TakeOffer", 0.9), ("Buy", 0.55), ("Upgrade", 0.25), ("Refresh", 0.2), ("BuyItem", 0.2), ("Sell", 0.05)):
        idx = [i for i, n in enumerate(names) if n == want]
        if idx and random.random() < p:
            return random.choice(idx)
    for want in ("SkipOffer", "ChooseGroup", "EndRound"):
        if want in names:
            return names.index(want)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=300000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500, help="steps per batch before the setting changes")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cks = checkpoints()
    print(f"{len(cks)} checkpoints; {logged()} fights already logged", flush=True)
    batch = 0
    while logged() < args.fights:
        batch += 1
        level, extra = pick(MODES), pick(EXTRA)
        mode = f"规程等级{level}"
        use_random = not cks or random.random() < 0.2
        ck = None if use_random else random.choice(cks)
        temp, eps = random.choice([1.0, 1.0, 1.5, 2.0]), random.choice([0.0, 0.05, 0.15, 0.3])
        path = os.path.join(OUT, f"fights_{time.strftime('%Y%m%d_%H%M%S')}_{level}_{extra}.jsonl")
        flags = PLANNER + ["--log-fights", path] + (["--extra-slots", str(extra)] if extra else [])
        t0 = time.time()
        gym = GymClient(mode=mode, envs=args.envs, extra=flags)
        enc = Encoder(gym.spec)
        model = None if use_random else load(ck, enc, device)
        kinds = gym.spec["kinds"]
        r = gym.reset(random.randrange(1_000_000))
        with torch.no_grad():
            for _ in range(args.steps):
                if model is None:
                    acts = [random_action(o, kinds) for o in r["obs"]]
                else:
                    logits, _ = model(to_torch(pad_actions(enc.encode(r["obs"])), device))
                    acts = torch.distributions.Categorical(logits=logits / temp).sample().cpu().numpy().tolist()
                    acts = [random_action(o, kinds) if random.random() < eps else a for o, a in zip(r["obs"], acts)]
                r = gym.step(acts)
        gym.close()
        with open(path, "rb") as fh:
            n = sum(1 for _ in fh)
        who = "random" if model is None else f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} T{temp} eps{eps}"
        print(f"batch {batch}: {mode} +{extra} slots, {who}: {n} fights in {time.time() - t0:.0f}s; total {logged()}", flush=True)


if __name__ == "__main__":
    main()
