"""Clear rate of policies on the SP simulator with the 策略 (band) fixed: every checkpoint plays the same runs of every
band and level (the simulator's default rehearsal, the most likely action).

  python sp_band_eval.py --ckpt runs/sp2/last.pt,runs/sp3/last.pt --bands band_amiya,band_takecare,band_bonus
"""
import argparse
import json
import os
import time

os.environ["ASP_SIM"] = "sp"
os.environ.setdefault("ASP_SP_PER_PROC", "4")

import torch

import train_mix as tm
from model import Encoder, Policy

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoints, comma separated")
    ap.add_argument("--spec", default="runs/sp_bc/spec.json")
    ap.add_argument("--bands", required=True, help="策略 ids, comma separated; 'any' is the game AI's own pick")
    ap.add_argument("--levels", default="11,14,16,20")
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--draw", default=None, help="a run's draw.json: the given 盟约 are drawn per leader by its win rates")
    ap.add_argument("--best", type=int, default=0, help="--draw: every leader's best 盟约 instead of a draw")
    ap.add_argument("--extra", default="", help="further simulator options, ; separated (e.g. --sp:focus=kjeragShip)")
    args = ap.parse_args()
    if args.draw:
        d = json.load(open(os.path.join(HERE, args.draw), encoding="utf-8"))
        tm.DRAW = tm.DrawTable(d["mains"], d["seconds"]).load(os.path.join(HERE, args.draw))
        tm.DRAW_BEST = bool(args.best)
    enc = Encoder(json.load(open(os.path.join(HERE, args.spec), encoding="utf-8")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = [int(x) for x in args.levels.split(",")]
    rows = []
    for ck in args.ckpt.split(","):
        model = Policy(enc).to(dev)
        model.load_state_dict(torch.load(os.path.join(HERE, ck), map_location=dev))
        for band in args.bands.split(","):
            res = []
            for lv in levels:
                t0 = time.time()
                clear, rounds, layers = tm.evaluate(model, enc, f"规程等级{lv}", dev, args.runs, args.seed, ([] if band == "any" else [f"--sp:bands={band}"]) + [x for x in args.extra.split(";") if x])
                res.append(clear)
                rows.append({"ckpt": ck, "band": band, "level": lv, "runs": args.runs, "clear": clear, "rounds": rounds, "layers": layers})
                print(f"{ck} {band} L{lv}: {clear * 100:.0f}% ({rounds:.1f} rounds, {layers:.0f} layers, {time.time() - t0:.0f}s)", flush=True)
            print(f"== {ck} {band}: " + " / ".join(f"{c * 100:.0f}%" for c in res) + f"  mean {sum(res) / len(res) * 100:.1f}%", flush=True)
            if args.out:
                json.dump(rows, open(os.path.join(HERE, args.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
