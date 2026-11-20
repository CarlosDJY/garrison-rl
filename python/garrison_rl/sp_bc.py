"""
Behaviour cloning on Stronghold-Protocol's game AI (sp_record.py's data): the policy learns its shop decisions, then
plays whole matches in the simulator to see how much of the teacher's clear rate it keeps.

  python sp_bc.py --tag sp_bc --data a --epochs 10
  python sp_bc.py --tag sp_bc --eval-only --levels 4,8,11,14,16,20 --runs 64
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ["ASP_SIM"] = "sp"
from gym_client import GymClient  # noqa: E402
from model import Encoder, Policy, to_torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sp")
A_PAD = 48
SMALL = {"unit_op": np.int16, "unit_items": np.int16, "shop": np.int16, "offer": np.int16, "hand": np.int16, "item": np.int16,
         "kind": np.int16, "ptr": np.int16, "expert": np.int16}


def pad(batch, A=A_PAD):
    a = batch["kind"].shape[1]
    B = batch["kind"].shape[0]
    if a >= A:
        for k in ("kind", "ptr", "amask"):
            batch[k] = batch[k][:, :A]
        return batch
    batch["kind"] = np.concatenate([batch["kind"], np.zeros((B, A - a), batch["kind"].dtype)], 1)
    batch["ptr"] = np.concatenate([batch["ptr"], np.zeros((B, A - a, 2), batch["ptr"].dtype)], 1)
    batch["amask"] = np.concatenate([batch["amask"], np.zeros((B, A - a), bool)], 1)
    return batch


def load(enc, files, chunk=4096):
    parts, rounds = [], []
    for f in files:
        rows = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
                if len(rows) == chunk:
                    parts.append(encode(enc, rows))
                    rounds += [r["round"] for r in rows]
                    rows = []
        if rows:
            parts.append(encode(enc, rows))
            rounds += [r["round"] for r in rows]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["_round"] = np.array(rounds, np.int16)
    return out


def encode(enc, rows):
    b = pad(enc.encode(rows))
    return {k: (v.astype(SMALL[k]) if k in SMALL else v) for k, v in b.items()}


def batch_of(data, idx, device):
    b = {k: v[idx] for k, v in data.items() if not k.startswith("_")}
    b = {k: (v.astype(np.int64) if k in SMALL else v) for k, v in b.items()}
    return to_torch(b, device)


def play(model, enc, mode, device, runs, seed, greedy=True, extra=()):
    gym = GymClient(mode=mode, envs=runs, extra=extra)
    r = gym.reset(seed)
    res = [None] * runs
    model.eval()
    with torch.no_grad():
        while any(x is None for x in res):
            logits, _ = model(to_torch(pad(enc.encode(r["obs"])), device))
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            r = gym.step(a.cpu().numpy())
            for i, info in enumerate(r["info"]):
                if info and res[i] is None:
                    res[i] = info
    gym.close()
    return res


def summary(res):
    return (f"{np.mean([x['cleared'] for x in res]):.0%}（{np.mean([x['rounds'] for x in res]):.1f} 回合，精锐 {np.mean([x['elites'] for x in res]):.1f}，"
            f"最高盟约 {np.mean([(x.get('layers') or [0])[0] for x in res]):.0f} 层）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp_bc")
    ap.add_argument("--data", default="a", help="sp_record.py tags, comma separated")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--levels", default="4,8,11,14,16,20")
    ap.add_argument("--runs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--sample", action="store_true", help="evaluate by sampling instead of the most likely action")
    ap.add_argument("--gym", default="", help="options for the gym, comma separated (rehearsal=0; picker,rehearsal=8 with ASP_SP_PICKER set)")
    ap.add_argument("--init", default=None, help="evaluate this checkpoint instead of the run's own")
    args = ap.parse_args()
    device = torch.device("cuda")
    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    report = os.path.join(HERE, "..", "..", "reports", "playground", f"{args.tag}.md")

    def P(s):
        print(s, flush=True)
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    spec_path = os.path.join(run_dir, "spec.json")
    if os.path.exists(spec_path):
        spec = json.load(open(spec_path, encoding="utf-8"))
    else:
        g = GymClient(mode="规程等级8", envs=1)
        spec = g.spec
        g.close()
        json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False)
    enc = Encoder(spec)
    model = Policy(enc).to(device)
    ckpt = os.path.join(run_dir, "last.pt")
    if not args.eval_only:
        files = sorted(f for t in args.data.split(",") for f in glob.glob(os.path.join(DATA, f"bc_{t}_L*_*.jsonl")))
        held = [f for f in files if f.endswith("_0.jsonl")]
        train_files = [f for f in files if f not in held]
        t0 = time.time()
        tr = load(enc, train_files)
        va = load(enc, held)
        n, nv = len(tr["expert"]), len(va["expert"])
        P(f"\n# 模仿学习（{args.tag}）\n\n- 数据：训练 {n} 个决策（{len(train_files)} 个文件），留出 {nv} 个（{len(held)} 个文件）；读入 {time.time() - t0:.0f} 秒")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total = args.epochs * (n // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
        kinds = spec["kinds"]
        for ep in range(1, args.epochs + 1):
            model.train()
            perm = np.random.permutation(n)
            losses = []
            for s0 in range(0, n - args.batch + 1, args.batch):
                idx = np.sort(perm[s0:s0 + args.batch])
                b = batch_of(tr, idx, device)
                logits, _ = model(b)
                loss = F.cross_entropy(logits, b["expert"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
            model.eval()
            hit, vl, by = 0, [], {}
            with torch.no_grad():
                for s0 in range(0, nv, 2048):
                    idx = np.arange(s0, min(nv, s0 + 2048))
                    b = batch_of(va, idx, device)
                    logits, _ = model(b)
                    vl.append(F.cross_entropy(logits, b["expert"], reduction="sum").item())
                    pred = logits.argmax(-1)
                    ok = (pred == b["expert"]).cpu().numpy()
                    hit += ok.sum()
                    k = torch.gather(b["kind"], 1, b["expert"].unsqueeze(1)).squeeze(1).cpu().numpy()
                    for kk, o in zip(k, ok):
                        c = by.setdefault(int(kk), [0, 0])
                        c[0] += int(o)
                        c[1] += 1
            per = "，".join(f"{kinds[k]} {c[0] / c[1]:.0%}（{c[1]}）" for k, c in sorted(by.items()))
            P(f"- 第 {ep} 轮：训练损失 {np.mean(losses):.3f}，留出损失 {sum(vl) / nv:.3f}，留出命中 {hit / nv:.1%}；按动作：{per}")
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(args.init or ckpt, map_location=device))
    extra = [f"--sp:{x}" for x in args.gym.split(",") if x]
    res = {}
    for lv in args.levels.split(","):
        res[lv] = play(model, enc, f"规程等级{lv}", device, args.runs, args.seed, greedy=not args.sample, extra=extra)
        P(f"- 模拟器里打规程 {lv}（{args.runs} 局，{'采样' if args.sample else '取最可能的动作'}{'，环境选项 ' + args.gym if args.gym else ''}"
          f"{'，' + os.path.basename(args.init) if args.init else ''}）：{summary(res[lv])}")


if __name__ == "__main__":
    main()
