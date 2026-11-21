"""Does a larger policy network fit the same decisions better? The behaviour-cloning data (sp_record.py's: the game AI's
shop decisions) is learnt by networks of several sizes with one recipe; what counts is the held-out loss and hit rate.

  python sp_capacity.py --sizes 128x3x4,192x4x6,256x6x8,384x6x8,512x8x8 --epochs 10
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
from model import Encoder, Policy  # noqa: E402
from sp_bc import DATA, HERE, batch_of, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="a")
    ap.add_argument("--sizes", default="128x3x4,192x4x6,256x6x8,384x6x8,512x8x8", help="width x layers x heads, comma separated")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--share", type=float, default=1.0, help="share of the training decisions used (data scaling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sp_capacity")
    args = ap.parse_args()
    device = torch.device("cuda")
    report = os.path.join(HERE, "..", "..", "reports", "playground", f"{args.tag}.md")

    def P(s):
        print(s, flush=True)
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    enc = Encoder(json.load(open(os.path.join(HERE, "runs", "sp_bc", "spec.json"), encoding="utf-8")))
    files = sorted(f for t in args.data.split(",") for f in glob.glob(os.path.join(DATA, f"bc_{t}_L*_*.jsonl")))
    held = [f for f in files if f.endswith("_0.jsonl")]
    t0 = time.time()
    tr = load(enc, [f for f in files if f not in held])
    va = load(enc, held)
    n, nv = len(tr["expert"]), len(va["expert"])
    rng = np.random.default_rng(args.seed)
    pool = np.sort(rng.permutation(n)[:int(n * args.share)])
    P(f"\n# 模型规模对比（{args.tag}）\n\n- 数据：训练 {len(pool)} 个决策（共 {n}），留出 {nv} 个；读入 {time.time() - t0:.0f} 秒；每种规模 {args.epochs} 轮，批大小 {args.batch}，学习率 {args.lr}")
    hard = va["_round"] >= 9   # the late rounds, where the decisions are harder
    for size in args.sizes.split(","):
        d, layers, heads = (int(x) for x in size.split("x"))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = Policy(enc, d=d, layers=layers, heads=heads).to(device)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * (len(pool) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
        best = (9.0, 0.0, 0.0, 0)
        t0 = time.time()
        rows = []
        for ep in range(1, args.epochs + 1):
            model.train()
            perm = pool[np.random.permutation(len(pool))]
            losses = []
            for s0 in range(0, len(perm) - args.batch + 1, args.batch):
                b = batch_of(tr, np.sort(perm[s0:s0 + args.batch]), device)
                logits, _ = model(b)
                loss = F.cross_entropy(logits, b["expert"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
            model.eval()
            vl, ok = [], []
            with torch.no_grad():
                for s0 in range(0, nv, 2048):
                    b = batch_of(va, np.arange(s0, min(nv, s0 + 2048)), device)
                    logits, _ = model(b)
                    vl.append(F.cross_entropy(logits, b["expert"], reduction="none").cpu().numpy())
                    ok.append((logits.argmax(-1) == b["expert"]).cpu().numpy())
            vl, ok = np.concatenate(vl), np.concatenate(ok)
            row = (float(vl.mean()), float(ok.mean()), float(ok[hard].mean()), ep)
            rows.append((float(np.mean(losses)),) + row)
            if row[0] < best[0]:
                best = row
        dt = time.time() - t0
        P(f"- {d} 宽 × {layers} 层（{params / 1e6:.1f}M 参数，{dt / args.epochs:.0f} 秒/轮）：最好的一轮是第 {best[3]} 轮——留出损失 {best[0]:.4f}，留出命中 {best[1]:.2%}，"
          f"第 9 回合起 {best[2]:.2%}；最后一轮训练损失 {rows[-1][0]:.4f} / 留出损失 {rows[-1][1]:.4f}")
        P("  - 各轮（训练损失 / 留出损失 / 留出命中）：" + "，".join(f"{r[0]:.3f}/{r[1]:.3f}/{r[2]:.1%}" for r in rows))
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
