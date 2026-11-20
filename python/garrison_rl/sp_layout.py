"""
A layout picker for Stronghold-Protocol's simulator: the game AI "rehearses" a prep's candidate layouts by simulating
each one's fight and keeps the best (fewest counted leaks, then leader damage, then layers gained). This model scores
the candidates from the field alone (tiles, facings, operators, enemies, covenants), so training can pick a layout
without simulating any of them. Data: sp_record.py --rehearsal 8 (every candidate played out, layouts_*.jsonl).

  python sp_layout.py --tag sp_layout_v1 --data b --epochs 30
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sp_fight import DIRS, MAX_E, MAX_U, FightNet, Vocab

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sp")
K_MAX = 10
ROWS, COLS = 24, 24


class LayoutVocab(Vocab):
    def __init__(self, rows=None, state=None):
        if state:
            self.__dict__.update(state)
            return
        self.enemies = sorted({k for r in rows for k in r["enemies"]})
        self.stages = sorted({r["stage"] for r in rows})
        self.bands = sorted({r["band"] or "" for r in rows})
        self.bosses = sorted({r["boss"] or "" for r in rows})
        self.n_ops = 1 + max(u[0] for r in rows for c in r["cands"] for u in c)
        self.n_items = 2 + max([i for r in rows for c in r["cands"] for u in c for i in u[5]] + [0])
        self.n_cov = 1 + max(b[0] for r in rows for b in r["bonds"] if b[0] is not None)


def encode_state(v, r):
    """What the candidates of one prep share: enemies, covenants, globals, ids."""
    e_idx = {k: i for i, k in enumerate(v.enemies)}
    en = np.full(MAX_E, -1, np.int64)
    ec = np.zeros(MAX_E, np.float32)
    for i, (k, c) in enumerate(sorted(r["enemies"].items(), key=lambda kv: -kv[1])[:MAX_E]):
        en[i] = e_idx.get(k, -1)
        ec[i] = c / 20
    cov = np.zeros((v.n_cov, 4), np.float32)
    for c, cnt, tier, layers, active in r["bonds"]:
        if c is not None and c < v.n_cov:
            cov[c] = [cnt / 6, tier / 3, layers / 100, active]
    g = np.zeros(20, np.float32)
    g[:8] = [r["level"] / 20, r["round"] / 15, r["lp"] / 45, r["startLp"] / 45, r["shopLevel"] / 6, r["bench"] / 10, 0, 1]
    g[8:] = r["preview"]
    ids = np.array([v.stages.index(r["stage"]) if r["stage"] in v.stages else 0, v.bands.index(r["band"] or "") if (r["band"] or "") in v.bands else 0,
                    v.bosses.index(r["boss"] or "") if (r["boss"] or "") in v.bosses else 0], np.int64)
    return en, ec, cov, g, ids


def encode_cand(units):
    op = np.full(MAX_U, -1, np.int64)
    uf = np.zeros((MAX_U, 8), np.float32)
    items = np.full((MAX_U, 2), -1, np.int64)
    tile = np.zeros(MAX_U, np.int64)
    for i, u in enumerate(units[:MAX_U]):
        op[i] = u[0]
        d = np.zeros(4)
        d[DIRS.get(u[4], 0)] = 1
        uf[i] = [u[1], u[2] / 20, u[3] / 20, 1, *d]
        tile[i] = min(ROWS - 1, max(0, u[2])) * COLS + min(COLS - 1, max(0, u[3]))
        for j, it in enumerate(u[5][:2]):
            items[i, j] = it
    return op, uf, items, tile


def encode_rows(v, rows):
    """Flat arrays over every candidate of every row, and for each row the slice of its candidates."""
    out = {k: [] for k in ("op", "uf", "items", "tile", "en", "ec", "cov", "g", "ids", "y")}
    spans, choice = [], []
    for r in rows:
        en, ec, cov, g, ids = encode_state(v, r)
        scores = {s["i"]: s for s in r["scores"]}
        k0 = len(out["op"])
        top = max([s["boss"] for s in r["scores"]] + [1.0])
        for i, c in enumerate(r["cands"][:K_MAX]):
            if i not in scores:
                continue
            s = scores[i]
            op, uf, items, tile = encode_cand(c)
            gg = g.copy()
            gg[6] = len(c) / 8
            for k, x in zip(("op", "uf", "items", "tile", "en", "ec", "cov", "g", "ids"), (op, uf, items, tile, en, ec, cov, gg, ids)):
                out[k].append(x)
            # leaks / 10, layers gained / 100, leader field, leader damage (share of the row's best), the candidate's index
            out["y"].append([min(3.0, s["leaks"] / 10), min(3.0, s["gain"] / 100), s["leader"], s["boss"] / top, i])
        n = len(out["op"]) - k0
        if n >= 2:
            spans.append((k0, n))
            ys = np.array(out["y"][k0:k0 + n])
            choice.append(int(np.argmax(utility(ys))))
        else:
            for k in out:
                del out[k][k0:]
    arr = {k: np.stack(x) for k, x in out.items()}
    arr["y"] = arr["y"].astype(np.float32)
    return arr, np.array(spans, np.int64), np.array(choice, np.int64)


def utility(y):
    """The game AI's order as one number (bigger is better): a normal field by leaks, then layers gained; a leader field by
    leader damage less a leak's cost. Earlier candidates win ties (the default plan is the first)."""
    leaks, gain, leader, boss, idx = y[..., 0] * 10, y[..., 1] * 100, y[..., 2], y[..., 3], y[..., 4]
    normal = -1000 * leaks + gain - 0.001 * idx
    lead = 1000 * boss * np.maximum(0, 1 - leaks / 150) - 0.001 * idx
    return np.where(leader > 0.5, lead, normal)


class LayoutNet(FightNet):
    def __init__(self, v, d=128, layers=3, heads=4):
        super().__init__(v, d, layers, heads)
        self.tile = nn.Embedding((len(v.stages) + 1) * ROWS * COLS, d)
        nn.init.normal_(self.tile.weight, std=0.02)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))   # score, leaks / 10, layers / 100, leader share

    def forward(self, t):
        B = t["g"].shape[0]
        pad_id = lambda x, n: torch.where(x < 0, torch.full_like(x, n), x)
        glob = (self.g(t["g"]) + self.stage(t["ids"][:, 0]) + self.band(t["ids"][:, 1]) + self.boss(t["ids"][:, 2]) + self.tok.weight[0]).unsqueeze(1)
        tile = self.tile(t["ids"][:, :1] * (ROWS * COLS) + t["tile"])
        units = self.op(pad_id(t["op"], self.n_ops)) + self.uf(t["uf"]) + self.item(pad_id(t["items"], self.n_items)).sum(2) + tile + self.tok.weight[1]
        enemies = self.en(pad_id(t["en"], self.n_en)) + self.ec(t["ec"].unsqueeze(-1)) + self.tok.weight[2]
        cov = self.cov_id.unsqueeze(0).expand(B, -1, -1) + self.cov(t["cov"]) + self.tok.weight[3]
        x = torch.cat([glob, units, enemies, cov], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), t["op"] < 0, t["en"] < 0,
                         torch.zeros(B, cov.shape[1], dtype=torch.bool, device=x.device)], 1)
        return self.head(self.norm(self.body(x, src_key_padding_mask=pad))[:, 0])


def group_loss(out, y, spans_local, choice):
    """Listwise: the game AI's pick among each prep's candidates; plus what each candidate's fight did."""
    ce = out.new_zeros(())
    for (k0, n), c in zip(spans_local, choice):
        ce = ce + F.cross_entropy(out[k0:k0 + n, 0].unsqueeze(0), torch.as_tensor([c], device=out.device))
    ce = ce / max(1, len(spans_local))
    normal = y[:, 2] < 0.5
    z = out.new_zeros(())
    l_leak = F.smooth_l1_loss(F.softplus(out[:, 1]), y[:, 0])
    l_gain = F.smooth_l1_loss(F.softplus(out[normal, 2]), y[normal, 1]) if normal.any() else z
    l_boss = F.mse_loss(torch.sigmoid(out[~normal, 3]), y[~normal, 3]) if (~normal).any() else z
    return ce + 4 * l_leak + 2 * l_gain + l_boss


class LayoutPicker:
    """Picks a layout among a prep's candidates with a trained LayoutNet (the gym's layoutQueries)."""

    def __init__(self, run_dir, device="cuda", margin=0.0):
        ck = torch.load(os.path.join(run_dir, "layoutnet.pt"), map_location=device)
        self.v = LayoutVocab(state=ck["vocab"])
        self.model = LayoutNet(self.v).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device, self.margin = device, margin
        self.queries = self.changed = 0

    def answer(self, rows):
        feats = {k: [] for k in ("op", "uf", "items", "tile", "en", "ec", "cov", "g", "ids")}
        spans = []
        for r in rows:
            en, ec, cov, g, ids = encode_state(self.v, r)
            k0 = len(feats["op"])
            for c in r["cands"][:K_MAX]:
                op, uf, items, tile = encode_cand(c)
                op = np.where(op >= self.v.n_ops, -1, op)
                items = np.where(items >= self.v.n_items, -1, items)
                gg = g.copy()
                gg[6] = len(c) / 8
                for k, x in zip(feats, (op, uf, items, tile, en, ec, cov, gg, ids)):
                    feats[k].append(x)
            spans.append((k0, len(feats["op"]) - k0))
        with torch.no_grad():
            out = self.model({k: torch.as_tensor(np.stack(x), device=self.device) for k, x in feats.items()})[:, 0].cpu().numpy()
        picks = []
        for k0, n in spans:
            s = out[k0:k0 + n]
            j = int(np.argmax(s))
            if s[j] - s[0] <= self.margin:
                j = 0   # the default plan unless another is clearly better
            picks.append(j)
        self.queries += len(picks)
        self.changed += sum(1 for p in picks if p)
        return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp_layout_v1")
    ap.add_argument("--data", default="b")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--groups", type=int, default=48, help="preps per batch")
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    device = torch.device("cuda")
    files = sorted(f for t in args.data.split(",") for f in glob.glob(os.path.join(DATA, f"layouts_{t}_L*_*.jsonl")))
    def load(fs):
        rows = []
        for f in fs:
            for line in open(f, encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass   # a file still being written ends mid-line
        return rows
    held_files = [f for f in files if f.endswith("_0.jsonl")]
    train, held = load([f for f in files if f not in held_files]), load(held_files)
    v = LayoutVocab(train + held)
    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    report = os.path.join(HERE, "..", "..", "reports", "playground", f"{args.tag}.md")

    def P(s):
        print(s, flush=True)
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    tr, tr_spans, tr_choice = encode_rows(v, train)
    va, va_spans, va_choice = encode_rows(v, held)
    model = LayoutNet(v).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr_spans) // args.groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    T = lambda d, idx: {k: torch.as_tensor(x[idx], device=device) for k, x in d.items()}
    P(f"\n# 站位挑选模型（{args.tag}，Stronghold-Protocol 模拟器）\n\n- 数据：训练 {len(tr_spans)} 次休整期（{len(tr['op'])} 个候选站位），留出 {len(va_spans)} 次（{len(va['op'])} 个）；"
      f"每次平均 {len(tr['op']) / max(1, len(tr_spans)):.1f} 个候选；参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(T(va, np.arange(s0, min(len(va["op"]), s0 + 2048)))) for s0 in range(0, len(va["op"]), 2048)]).cpu().numpy()
        y = va["y"]
        res = {k: [] for k in ("agree", "pick_leaks", "base_leaks", "best_leaks", "pick_gain", "base_gain", "best_gain", "pick_boss", "base_boss", "differs")}
        lead = {k: [] for k in ("pick", "base")}
        for (k0, n), c in zip(va_spans, va_choice):
            ys = y[k0:k0 + n]
            u = utility(ys)
            pick = int(np.argmax(out[k0:k0 + n, 0]))
            if ys[0, 2] > 0.5:
                lead["pick"].append(ys[pick, 3] * max(0, 1 - ys[pick, 0] * 10 / 150))
                lead["base"].append(ys[0, 3] * max(0, 1 - ys[0, 0] * 10 / 150))
                continue
            res["agree"].append(u[pick] >= u[c] - 0.02)   # the same result at a later index counts as right
            res["differs"].append(u[c] > u[0] + 1e-6)
            for name, j in (("pick", pick), ("base", 0), ("best", c)):
                res[f"{name}_leaks"].append(ys[j, 0] * 10)
                res[f"{name}_gain"].append(ys[j, 1] * 100)
        m = lambda k: float(np.mean(res[k])) if res[k] else float("nan")
        return (f"普通回合 {len(res['agree'])} 次：挑中最优 {m('agree'):.1%}（默认站位本来就最优的占 {1 - m('differs'):.1%}）；"
                f"每回合漏怪——默认站位 {m('base_leaks'):.3f}，模型挑的 {m('pick_leaks'):.3f}，真打挑的 {m('best_leaks'):.3f}；"
                f"每回合层数——默认 {m('base_gain'):.2f}，模型 {m('pick_gain'):.2f}，真打 {m('best_gain'):.2f}；"
                f"首领回合 {len(lead['pick'])} 次：伤害（占候选里最高的比例）默认 {np.mean(lead['base']) if lead['base'] else float('nan'):.3f}，模型 {np.mean(lead['pick']) if lead['pick'] else float('nan'):.3f}")

    n_groups = len(tr_spans)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n_groups)
        ls = []
        for s0 in range(0, n_groups - args.groups + 1, args.groups):
            gi = perm[s0:s0 + args.groups]
            idx, local, k = [], [], 0
            for g in gi:
                k0, n = tr_spans[g]
                idx.extend(range(k0, k0 + n))
                local.append((k, n))
                k += n
            b = T(tr, np.array(idx))
            loss = group_loss(model(b), b["y"], local, tr_choice[gi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
        if ep % 5 == 0 or ep == args.epochs:
            P(f"- 第 {ep} 轮：训练损失 {np.mean(ls):.3f}；留出——{evaluate()}")
    torch.save(dict(model=model.state_dict(), vocab=v.state()), os.path.join(run_dir, "layoutnet.pt"))


if __name__ == "__main__":
    main()
