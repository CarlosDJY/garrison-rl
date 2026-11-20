"""
A small fight-outcome model on Stronghold-Protocol's real battles (the rows spgym.mjs --fights writes): from the field
(operators, 精锐, tiles, equipment), the covenants' counts and layers, the enemy group, the 规程 level and the round, it
predicts the HP the fight costs, whether nothing leaks, the layers gained, and on a leader round the share of the leader's
HP taken and whether it dies.

  python sp_fight.py --tag sp_fight_v1 --data a --epochs 30
  python sp_fight.py --tag sp_fight_v2 --data a,b --extra "data/sp_train/sp2_L*_*.jsonl" --d 192 --layers 4 --per-cov \
      --epochs 20 --batch 512 --compare runs/sp_fight_v1/fightnet.pt

v2 (--per-cov) also predicts each covenant's layers gained (what the next round starts with) and the funds the next prep
starts with minus the funds this one ended with (rows that carry funds: the ones train_mix.py --sp-fights keeps).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sp")
MAX_U, MAX_E = 10, 16
DIRS = {"RIGHT": 0, "UP": 1, "DOWN": 2, "LEFT": 3}


class Vocab:
    def __init__(self, rows=None, state=None):
        if state:
            self.__dict__.update(state)
            return
        self.enemies = sorted({k for r in rows for k in r["enemies"]})
        self.stages = sorted({r["stage"] for r in rows})
        self.bands = sorted({r["band"] or "" for r in rows})
        self.bosses = sorted({r["boss"] or "" for r in rows})
        self.n_ops = 1 + max(u[0] for r in rows for u in r["units"])
        self.n_items = 2 + max([i for r in rows for u in r["units"] for i in u[5]] + [0])
        self.n_cov = 1 + max(b[0] for r in rows for b in r["bonds"] if b[0] is not None)

    def state(self):
        return dict(self.__dict__)


def encode(v, rows):
    B = len(rows)
    e_idx = {k: i for i, k in enumerate(v.enemies)}
    op = np.full((B, MAX_U), -1, np.int64)
    uf = np.zeros((B, MAX_U, 8), np.float32)
    items = np.full((B, MAX_U, 2), -1, np.int64)
    en = np.full((B, MAX_E), -1, np.int64)
    ec = np.zeros((B, MAX_E), np.float32)
    cov = np.zeros((B, v.n_cov, 4), np.float32)
    g = np.zeros((B, 8 + 12), np.float32)
    ids = np.zeros((B, 3), np.int64)
    # lost / 10, no leak, layers gained / 100, leader round, share, killed, funds gained / 10, funds known
    y = np.zeros((B, 8), np.float32)
    yc = np.zeros((B, v.n_cov), np.float32)   # each covenant's layers gained / 100
    for b, r in enumerate(rows):
        for i, u in enumerate(r["units"][:MAX_U]):
            op[b, i] = u[0]
            d = np.zeros(4)
            d[DIRS.get(u[4], 0)] = 1
            uf[b, i] = [u[1], u[2] / 20, u[3] / 20, 1, *d]
            for j, it in enumerate(u[5][:2]):
                items[b, i, j] = it
        for i, (k, c) in enumerate(sorted(r["enemies"].items(), key=lambda kv: -kv[1])[:MAX_E]):
            en[b, i] = e_idx.get(k, -1)
            ec[b, i] = c / 20
        before = {}
        for c, cnt, tier, layers, active in r["bonds"]:
            if c is None or c >= v.n_cov:
                continue
            cov[b, c] = [cnt / 6, tier / 3, layers / 100, active]
            before[c] = (layers, active)
        gain = sum(max(0, ly - before.get(c, (0, 0))[0]) for c, ly in r["out"]["bonds"] if c is not None)
        for c, ly in r["out"]["bonds"]:
            if c is not None and c < v.n_cov:
                yc[b, c] = min(3.0, max(0, ly - before.get(c, (0, 0))[0]) / 100)
        g[b, :8] = [r["level"] / 20, r["round"] / 15, r["lp"] / 45, r["startLp"] / 45, r["shopLevel"] / 6, r["bench"] / 10, len(r["units"]) / 8, 1]
        g[b, 8:] = r["preview"]
        ids[b] = [v.stages.index(r["stage"]) if r["stage"] in v.stages else 0, v.bands.index(r["band"] or "") if (r["band"] or "") in v.bands else 0,
                  v.bosses.index(r["boss"] or "") if (r["boss"] or "") in v.bosses else 0]
        o = r["out"]
        leader = 1.0 if o["share"] >= 0 else 0.0
        funds = r.get("funds") is not None and o.get("funds") is not None and not o.get("ended")
        y[b] = [0 if leader else min(1.0, o["lost"] / 10), 1.0 if o["lost"] <= 0 else 0.0, min(3.0, gain / 100), leader, max(0.0, o["share"]), o["killed"],
                min(3.0, max(0.0, (o["funds"] - r["funds"]) / 10)) if funds else 0.0, 1.0 if funds else 0.0]
    return dict(op=op, uf=uf, items=items, en=en, ec=ec, cov=cov, g=g, ids=ids, y=y, yc=yc)


class FightNet(nn.Module):
    def __init__(self, v, d=128, layers=3, heads=4, per_cov=False):
        super().__init__()
        self.per_cov = per_cov
        if per_cov:
            self.head_cov = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))   # a covenant's layers gained / 100
            self.head_funds = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))  # funds gained / 10
        self.op = nn.Embedding(v.n_ops + 1, d)
        self.item = nn.Embedding(v.n_items + 1, d)
        self.uf = nn.Linear(8, d)
        self.en = nn.Embedding(len(v.enemies) + 1, d)
        self.ec = nn.Linear(1, d)
        self.cov_id = nn.Parameter(torch.zeros(v.n_cov, d))
        self.cov = nn.Linear(4, d)
        self.g = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.Linear(d, d))
        self.stage = nn.Embedding(len(v.stages) + 1, d)
        self.band = nn.Embedding(len(v.bands) + 1, d)
        self.boss = nn.Embedding(len(v.bosses) + 1, d)
        self.tok = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 5))
        self.n_ops, self.n_items, self.n_en = v.n_ops, v.n_items, len(v.enemies)

    def forward(self, t):
        B = t["g"].shape[0]
        pad_id = lambda x, n: torch.where(x < 0, torch.full_like(x, n), x)
        glob = (self.g(t["g"]) + self.stage(t["ids"][:, 0]) + self.band(t["ids"][:, 1]) + self.boss(t["ids"][:, 2]) + self.tok.weight[0]).unsqueeze(1)
        units = self.op(pad_id(t["op"], self.n_ops)) + self.uf(t["uf"]) + self.item(pad_id(t["items"], self.n_items)).sum(2) + self.tok.weight[1]
        enemies = self.en(pad_id(t["en"], self.n_en)) + self.ec(t["ec"].unsqueeze(-1)) + self.tok.weight[2]
        cov = self.cov_id.unsqueeze(0).expand(B, -1, -1) + self.cov(t["cov"]) + self.tok.weight[3]
        x = torch.cat([glob, units, enemies, cov], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), t["op"] < 0, t["en"] < 0,
                         torch.zeros(B, cov.shape[1], dtype=torch.bool, device=x.device)], 1)
        hs = self.norm(self.body(x, src_key_padding_mask=pad))
        h = hs[:, 0]
        out = self.head(h)   # lost / 10 (logit), no leak (logit), layers / 100, leader share (logit), killed (logit)
        if not self.per_cov:
            return out
        n_cov = cov.shape[1]
        return out, F.softplus(self.head_cov(hs[:, -n_cov:]).squeeze(-1)), F.softplus(self.head_funds(h).squeeze(-1))


def loss_of(out, y, yc=None):
    extra = None
    if isinstance(out, tuple):
        out, cov_gain, funds = out
        extra = (cov_gain, funds)
    normal = y[:, 3] < 0.5
    leader = ~normal
    z = out.new_zeros(())
    l_lost = F.mse_loss(torch.sigmoid(out[normal, 0]), y[normal, 0]) if normal.any() else z
    l_clean = F.binary_cross_entropy_with_logits(out[normal, 1], y[normal, 1]) if normal.any() else z
    l_layer = F.smooth_l1_loss(F.softplus(out[normal, 2]), y[normal, 2]) if normal.any() else z
    l_share = F.mse_loss(torch.sigmoid(out[leader, 3]), y[leader, 4]) if leader.any() else z
    l_kill = F.binary_cross_entropy_with_logits(out[leader, 4], y[leader, 5]) if leader.any() else z
    total = 4 * l_lost + l_clean + 2 * l_layer + 2 * l_share + l_kill
    if extra is not None:
        cov_gain, funds = extra
        known = normal & (y[:, 7] > 0.5)
        if normal.any():
            total = total + 20 * F.smooth_l1_loss(cov_gain[normal], yc[normal])
        if known.any():
            total = total + 2 * F.smooth_l1_loss(funds[known], y[known, 6])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp_fight_v1")
    ap.add_argument("--data", default="a", help="record tags, comma separated")
    ap.add_argument("--extra", default=None, help="more fight files (a glob under this directory): train_mix.py --sp-fights' rows")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--per-cov", action="store_true", help="also predict each covenant's layers gained and the funds gained")
    ap.add_argument("--compare", default=None, help="an earlier fightnet.pt (128 wide, 3 layers) scored on the same held-out fights")
    args = ap.parse_args()
    device = torch.device("cuda")
    files = sorted(f for t in args.data.split(",") if t for f in glob.glob(os.path.join(DATA, f"fights_{t}_L*_*.jsonl")))
    own = sorted(glob.glob(os.path.join(HERE, args.extra))) if args.extra else []
    load = lambda fs: [json.loads(line) for f in fs for line in open(f, encoding="utf-8")]
    held_files = [f for f in files + own if f.endswith("_0.jsonl")]
    train = load([f for f in files + own if f not in held_files])
    held = load(held_files)
    held_own = np.array([f in own for f in held_files for _ in open(f, encoding="utf-8")])
    v = Vocab(train + held)
    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    report = os.path.join(HERE, "..", "..", "reports", "playground", f"{args.tag}.md")

    def P(s):
        print(s, flush=True)
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    tr = encode(v, train)
    va = encode(v, held)
    n = len(train)
    del train
    T = lambda d, idx: {k: torch.as_tensor(x[idx], device=device) for k, x in d.items()}
    model = FightNet(v, d=args.d, layers=args.layers, per_cov=args.per_cov).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (n // args.batch), pct_start=0.1)
    P(f"\n# 战斗结果预测模型（{args.tag}，Stronghold-Protocol 模拟器）\n\n- 数据：训练 {n} 场，留出 {len(held)} 场（各档的第 0 个文件，对局不重叠；"
      f"其中训练时存下的策略对局 {int(held_own.sum())} 场）；敌人种类 {len(v.enemies)}，参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    sig = lambda x: 1 / (1 + np.exp(-x))
    y = va["y"]
    nm, ld = y[:, 3] < 0.5, y[:, 3] >= 0.5

    def predict(net, data, rows):
        net.eval()
        outs, covs, funds = [], [], []
        with torch.no_grad():
            for s0 in range(0, rows, 2048):
                o = net(T(data, np.arange(s0, min(rows, s0 + 2048))))
                if isinstance(o, tuple):
                    covs.append(o[1].cpu().numpy())
                    funds.append(o[2].cpu().numpy())
                    o = o[0]
                outs.append(o.cpu().numpy())
        return np.concatenate(outs), (np.concatenate(covs) if covs else None), (np.concatenate(funds) if funds else None)

    def auc(p, t):
        pos, neg = p[t > 0.5], p[t <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        order = np.argsort(np.concatenate([pos, neg]))
        rank = np.empty(len(order))
        rank[order] = np.arange(1, len(order) + 1)
        return (rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    def score(name, out, covs, funds, m=None):
        m = np.ones(len(y), bool) if m is None else m
        a, b = nm & m, ld & m
        lost_p, lost_r = sig(out[a, 0]) * 10, y[a, 0] * 10
        clean_p = sig(out[a, 1])
        lay_p = (covs[a].sum(1) if covs is not None else np.log1p(np.exp(out[a, 2]))) * 100
        lay_r = y[a, 2] * 100
        kill_p = sig(out[b, 4])
        s = (f"- {name}：掉血误差 {np.abs(lost_p - lost_r).mean():.2f}（只猜均值 {np.abs(lost_r - lost_r.mean()).mean():.2f}；真实均值 {lost_r.mean():.2f}，预测均值 {lost_p.mean():.2f}），"
             f"不漏怪判断准确率 {((clean_p > 0.5) == (y[a, 1] > 0.5)).mean():.1%}（不漏占 {y[a, 1].mean():.0%}，区分度 {auc(clean_p, y[a, 1]):.3f}），"
             f"层数误差 {np.abs(lay_p - lay_r).mean():.1f}（只猜均值 {np.abs(lay_r - lay_r.mean()).mean():.1f}；真实 {lay_r.mean():.1f}，预测 {lay_p.mean():.1f}），"
             f"首领战 {int(b.sum())} 场：击杀判断准确率 {((kill_p > 0.5) == (y[b, 5] > 0.5)).mean():.1%}（真实击杀率 {y[b, 5].mean():.0%}，预测 {kill_p.mean():.0%}，区分度 {auc(kill_p, y[b, 5]):.3f}），"
             f"进度误差 {np.abs(sig(out[b, 3]) - y[b, 4]).mean():.3f}")
        if covs is not None:
            k = a & (y[:, 7] > 0.5)
            cr, cp = va["yc"][a] * 100, covs[a] * 100
            grew = cr > 0
            s += (f"；单个盟约层数误差 {np.abs(cp - cr)[grew].mean():.1f}（只看真实有增长的，平均增长 {cr[grew].mean():.1f}）"
                  f"；资金误差 {np.abs(funds[k] - y[k, 6]).mean() * 10:.2f}（{int(k.sum())} 场；只猜均值 {np.abs(y[k, 6] - y[k, 6].mean()).mean() * 10:.2f}）" if k.any() else "")
        P(s)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        ls = []
        for s0 in range(0, n - args.batch + 1, args.batch):
            b = T(tr, np.sort(perm[s0:s0 + args.batch]))
            loss = loss_of(model(b), b["y"], b["yc"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
        if ep % 5 == 0 or ep == args.epochs:
            score(f"第 {ep} 轮（训练损失 {np.mean(ls):.3f}），留出", *predict(model, va, len(held)))
            torch.save(dict(model=model.state_dict(), vocab=v.state(), d=args.d, layers=args.layers, per_cov=args.per_cov), os.path.join(run_dir, "fightnet.pt"))
    out, covs, funds = predict(model, va, len(held))
    if held_own.any() and not held_own.all():
        score("其中策略对局", out, covs, funds, held_own)
        score("其中自带 AI 对局", out, covs, funds, ~held_own)
    if args.compare:
        ck = torch.load(args.compare, map_location=device)
        v0 = Vocab(state=ck["vocab"])
        old = FightNet(v0).to(device)
        old.load_state_dict(ck["model"])
        o0, _, _ = predict(old, encode(v0, held), len(held))
        score(f"对照 {os.path.basename(os.path.dirname(args.compare))}（同一批留出）", o0, None, None)
        if held_own.any() and not held_own.all():
            score("对照，其中策略对局", o0, None, None, held_own)
    # By 规程 level: the model's calibration where the fields differ most.
    levels = np.array([r["level"] for r in held])
    for lv in sorted(set(levels)):
        m = (levels == lv) & nm
        md = (levels == lv) & ld
        if m.sum() == 0:
            continue
        lay = (covs[m].sum(1) if covs is not None else np.log1p(np.exp(out[m, 2]))) * 100
        P(f"  - 规程 {lv}：掉血 真实 {y[m, 0].mean() * 10:.2f} / 预测 {(sig(out[m, 0]) * 10).mean():.2f}；层数 真实 {y[m, 2].mean() * 100:.1f} / 预测 {lay.mean():.1f}；"
          f"首领击杀 真实 {y[md, 5].mean() if md.any() else float('nan'):.0%} / 预测 {sig(out[md, 4]).mean() if md.any() else float('nan'):.0%}")


if __name__ == "__main__":
    main()
