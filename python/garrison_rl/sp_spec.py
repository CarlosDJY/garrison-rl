"""Writes the simulator's catalogue (operators, 盟约, items, action kinds: what the encoder is built from) to
runs/sp_bc/spec.json, where sp_band_eval.py looks for it. The checkpoints only fit the catalogue of the simulator
version they were trained on.

  python sp_spec.py [--out runs/sp_bc/spec.json]
"""
import argparse
import json
import os

os.environ["ASP_SIM"] = "sp"

from gym_client import SPGymClient

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sp_bc/spec.json")
    args = ap.parse_args()
    gym = SPGymClient(mode="规程等级8", envs=1, per_proc=1)
    path = os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(gym.spec, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    for p in gym.procs:
        p.proc.kill()
    print(path, {k: len(v) for k, v in gym.spec.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
