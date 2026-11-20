"""
Trains an ensemble of FightNet on the logged fights and checks it on held-out data files (other settings / policies):
  - accuracy per target (leaks, boss kill / progress / overtime, layer gains, HP lost);
  - whether the ensemble's spread tells where it is wrong (error by spread decile);
  - selection bias: the boards it rates best, against what they really cost (what an RL policy would exploit), and how
    much a pessimistic estimate (mean + k x spread) removes;
  - extrapolation: a model trained without the hardest difficulties, scored on them.
Writes the models to runs/fightnet/ and the report to reports/playground/fightnet.md.

  python train_fight.py --models 5 --epochs 30
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch

from fight_model import ROOT, FightNet, Vocab, featurize, load_rows, losses, to_t

OUT = os.path.join(ROOT, "python", "garrison_rl", "runs", "fightnet")
REPORT = os.path.join(ROOT, "reports", "playground", "fightnet.md")


def train_one(x, y, idx, vocab, args, seed, device, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FightNet(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * max(1, len(idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    boot = np.random.choice(idx, len(idx), replace=True)     # bootstrap: each model sees its own resample
    yt = {k: torch.as_tensor(v, device=device) for k, v in y.items()}
    model.train()
    for ep in range(args.epochs):
        np.random.shuffle(boot)
        tot = 0
        for s in range(0, len(boot) - args.batch + 1, args.batch):
            b = boot[s:s + args.batch]
            p = model(to_t(x, b, device))
            loss, _ = losses(p, {k: v[torch.as_tensor(b, device=device)] for k, v in yt.items()})
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            log(f"  model {seed} epoch {ep + 1}: train loss {tot / max(1, len(boot) // args.batch):.3f}")
    model.eval()
    return model


@torch.no_grad()
def predict(models, x, idx, device, batch=4096):
    outs = []
    for m in models:
        parts = []
        for s in range(0, len(idx), batch):
            p = m(to_t(x, idx[s:s + batch], device))
            p["kill"] = torch.sigmoid(p["kill"])
            parts.append({k: v.float().cpu().numpy() for k, v in p.items()})
        outs.append({k: np.concatenate([q[k] for q in parts]) for k in parts[0]})
    stack = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    return {k: v.mean(0) for k, v in stack.items()}, {k: v.std(0) for k, v in stack.items()}, stack


def hp_cost(leak, over, boss, hp):
    return np.where(boss > 0, np.minimum(hp, over), leak)


def calib(P, spread, err, title):
    """Does the spread know where it is wrong? Error by spread decile, and the correlation."""
    if len(err) < 20:
        return float("nan")
    dec = np.array_split(np.argsort(spread), 10)
    P(f"\n{title}：分歧由低到高 10 档，每档的实际误差\n")
    P("| 档 | " + " | ".join(str(k + 1) for k in range(10)) + " |\n|---|" + "---|" * 10)
    P("| 平均分歧 | " + " | ".join(f"{spread[d].mean():.2f}" for d in dec) + " |")
    P("| 实际误差 | " + " | ".join(f"{err[d].mean():.2f}" for d in dec) + " |")
    corr = float(np.corrcoef(spread, err)[0, 1])
    P(f"\n相关系数 {corr:.2f}")
    return corr


def evaluate(models, x, y, rows, idx, device, title, P):
    mean, std, stack = predict(models, x, idx, device)
    boss = y["boss"][idx] > 0
    normal = ~boss
    hp = np.array([rows[i]["hp"] for i in idx], np.float32)
    real = hp_cost(y["leak"][idx], y["over"][idx], y["boss"][idx], hp)
    # Each network's HP cost, then their mean and spread (on a boss round overtime past the HP left costs nothing more).
    per = np.stack([hp_cost(stack["leak"][m], stack["over"][m], y["boss"][idx], hp) for m in range(stack["leak"].shape[0])])
    pred, spread = per.mean(0), per.std(0)
    err = np.abs(pred - real)
    P(f"\n### {title}（{len(idx)} 场，其中首领回合 {boss.sum()}）\n")
    P("| 指标 | 数值 |\n|---|---|")
    if normal.any():
        P(f"| 普通回合漏怪 MAE（只） | {np.abs(mean['leak'] - y['leak'][idx])[normal].mean():.2f} |")
        P(f"| 普通回合漏怪：真值平均 / 预测平均 | {y['leak'][idx][normal].mean():.2f} / {mean['leak'][normal].mean():.2f} |")
        P(f"| 普通回合“漏不漏怪”判断准确率 | {((mean['leak'] >= 0.5) == (y['leak'][idx] >= 0.5))[normal].mean():.1%} |")
    if boss.any():
        kr = y["kill"][idx][boss].mean()
        P(f"| 首领击杀判断准确率（真实击杀率 {kr:.0%}） | {((mean['kill'] >= 0.5) == (y['kill'][idx] >= 0.5))[boss].mean():.1%} |")
        pk, yk = mean["kill"][boss], y["kill"][idx][boss]
        P(f"| 首领击杀概率 Brier 分（越低越好；只猜真实击杀率是 {kr * (1 - kr):.3f}） | {((pk - yk) ** 2).mean():.3f} |")
        P(f"| 首领打掉比例 MAE | {np.abs(mean['prog'] - y['prog'][idx])[boss].mean():.3f} |")
        P(f"| 首领回合掉血 MAE（真值平均 {real[boss].mean():.1f}） | {err[boss].mean():.2f} |")
        killed = boss & (y["kill"][idx] > 0)
        if killed.any():
            P(f"| 其中击杀局的超时秒数 MAE | {np.abs(mean['over'] - y['over'][idx])[killed].mean():.1f} |")
    P(f"| 掉血 MAE（全部回合） | {err.mean():.2f} |")
    P(f"| 掉血偏差（预测 − 真值，全部） | {(pred - real).mean():+.2f} |")
    P(f"| 层数增量 MAE（每盟约） | {np.abs(mean['layer'] - y['layer'][idx]).mean():.3f}（真值平均 {y['layer'][idx].mean():.3f}） |")
    tl, pl = y['layer'][idx].sum(1), mean['layer'].sum(1)
    P(f"| 每场层数合计：真值平均 / 预测平均 / MAE | {tl.mean():.1f} / {pl.mean():.1f} / {np.abs(pl - tl).mean():.1f} |")
    big = tl >= 80
    if big.any():
        P(f"| 其中真值 ≥ 80 层的场（{big.sum()}）：真值平均 / 预测平均 | {tl[big].mean():.1f} / {pl[big].mean():.1f} |")
    if boss.any():
        P(f"| 首领击杀率：真实 / 预测平均 | {y['kill'][idx][boss].mean():.1%} / {mean['kill'][boss].mean():.1%} |")
    P(f"| 处理比例 MAE | {np.abs(mean['hand'] - y['hand'][idx]).mean():.3f} |")
    # By difficulty.
    modes = np.array([rows[i]["mode"] for i in idx])
    P("\n| 难度 | 场数 | 普通回合漏怪 MAE | 首领回合场数 | 首领击杀准确率 | 首领掉血 MAE | 掉血偏差 |\n|---|---|---|---|---|---|---|")
    for m in sorted(set(modes), key=lambda m: int(m.replace("规程等级", "") or 0)):
        k = modes == m
        nk, bk = k & normal, k & boss
        P(f"| {m} | {k.sum()} | {np.abs(mean['leak'] - y['leak'][idx])[nk].mean() if nk.any() else float('nan'):.2f} | {bk.sum()} | "
          f"{((mean['kill'] >= 0.5) == (y['kill'][idx] >= 0.5))[bk].mean() if bk.any() else float('nan'):.1%} | "
          f"{err[bk].mean() if bk.any() else float('nan'):.2f} | {(pred - real)[k].mean():+.2f} |")
    if boss.any():
        pk, yk = mean["kill"][boss], y["kill"][idx][boss]
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        P("\n首领击杀概率的校准（预测概率分档 → 实际击杀率）：\n")
        P("| 预测概率 | " + " | ".join(f"{a:.0%}–{min(1, b):.0%}" for a, b in zip(bins, bins[1:])) + " |\n|---|" + "---|" * (len(bins) - 1))
        P("| 场数 | " + " | ".join(str(int(((pk >= a) & (pk < b)).sum())) for a, b in zip(bins, bins[1:])) + " |")
        P("| 实际击杀率 | " + " | ".join(f"{yk[(pk >= a) & (pk < b)].mean():.0%}" if ((pk >= a) & (pk < b)).any() else "-" for a, b in zip(bins, bins[1:])) + " |")
    c1 = calib(P, spread[normal], err[normal], "普通回合")
    c2 = calib(P, spread[boss], err[boss], "首领回合")
    # Selection bias: among boards facing the same thing (difficulty, round, boss), the ones it likes best.
    groups = {}
    for j, i in enumerate(idx):
        r = rows[i]
        groups.setdefault((r["mode"], r["round"], r["boss"], bool(boss[j])), []).append(j)
    P("\n同难度、同回合、同首领的阵容里，挑模型认为最好的 10%（预测掉血最少）时，预测和实际的差（正数 = 偏乐观，是策略会钻的空子）：\n")
    P("| 估计方式 | 选中阵容的预测掉血 | 实际掉血 | 偏乐观 |\n|---|---|---|---|")
    for k in (0.0, 0.5, 1.0, 2.0):
        est = pred + k * spread
        top = []
        for g in groups.values():
            if len(g) >= 20:
                g = np.array(g)
                top.extend(g[np.argsort(est[g])[: max(1, len(g) // 10)]])
        top = np.array(top, int)
        if len(top):
            P(f"| 均值 + {k:g}×分歧 | {pred[top].mean():.2f} | {real[top].mean():.2f} | {real[top].mean() - pred[top].mean():+.2f} |")
    # And the boards it likes worst (would the policy be scared off good boards?).
    return dict(err=float(err.mean()), corr_normal=c1, corr_boss=c2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--holdout", type=float, default=0.15, help="share of data files held out")
    ap.add_argument("--extrapolate", action="store_true", help="also train without 规程等级16/20 and score on them")
    ap.add_argument("--boss-v2", action="store_true",
                    help="boss rounds from data/fights_v2 (fought again with the leader skills), ordinary rounds from data/fights")
    ap.add_argument("--max-round", type=int, default=99, help="drop rounds after this one (15: the 隐秘核心)")
    ap.add_argument("--lineup", action="store_true", help="add data/fights_lineup (gen_lineup_boards.py): boards toward the saved lineups")
    ap.add_argument("--online", type=int, default=0, help="add this many of the policy's own boards (data/fights_online, the newest runs first)")
    ap.add_argument("--no-pot", action="store_true", help="the old layer head (no per-unit layer potential)")
    ap.add_argument("--tag", default="fightnet", help="models go to runs/<tag>, the report to reports/playground/<tag>.md")
    args = ap.parse_args()
    device = torch.device("cuda")
    global OUT, REPORT
    OUT = os.path.join(ROOT, "python", "garrison_rl", "runs", args.tag)
    REPORT = os.path.join(ROOT, "reports", "playground", args.tag + ".md")
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def P(s):
        lines.append(s)
        print(s, flush=True)
        open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    t0 = time.time()
    rows = load_rows()
    if args.boss_v2:
        v2 = load_rows(sorted(glob.glob(os.path.join(ROOT, "python", "garrison_rl", "data", "fights_v2", "boss_*.jsonl"))))
        for r in v2:
            r["_file"] = r.get("src") or r["_file"]   # held out with the data file the board came from
        rows = [r for r in rows if r["sim"]["Bosses"] == 0] + v2
    base_dir = os.path.join(ROOT, "python", "garrison_rl", "data")
    extra_held, test_rows = set(), []
    if args.lineup:
        fs = sorted(glob.glob(os.path.join(base_dir, "fights_lineup", "lineup_train_*.jsonl")))
        extra_held.update(os.path.basename(f) for f in (fs[:1] + fs[-1:]))   # a chunk of the mixed boards and one of the finished ones
        rows += load_rows(fs)
        test_rows = load_rows(sorted(glob.glob(os.path.join(base_dir, "fights_lineup", "lineup_test_*.jsonl"))))   # lineups never trained on
    if args.online > 0:
        fs = sorted((f for f in glob.glob(os.path.join(base_dir, "fights_online", "*.jsonl")) if not os.path.basename(f).startswith("_")),
                    key=os.path.getmtime, reverse=True)
        extra_held.update(os.path.basename(f) for f in fs[:3])             # the newest policy boards
        got = []
        for f in fs:
            if len(got) >= args.online:
                break
            got += load_rows([f])
        rows += got
    rows = [r for r in rows if r["round"] <= args.max_round]
    test_rows = [r for r in test_rows if r["round"] <= args.max_round]
    n_main = len(rows)
    rows += test_rows
    files = sorted({r["_file"] for r in rows[:n_main] if r["_file"].startswith("fights_") or r["_file"].startswith("规程") or "_" in r["_file"] and r["_file"].split("_")[0] not in ("lineup", "mix1", "mix2", "mix3", "mix4")})
    def level_of(f):
        p = f.split("_")
        return p[3] if len(p) > 3 else p[0]
    held = set()
    for lv in sorted({level_of(f) for f in files}):
        fs = [f for f in files if level_of(f) == lv]
        random.Random(7).shuffle(fs)
        if len(fs) >= 2:
            held.update(fs[: max(1, round(len(fs) * args.holdout))])
    held |= extra_held
    tr = np.array([i for i, r in enumerate(rows[:n_main]) if r["_file"] not in held])
    va = np.array([i for i, r in enumerate(rows[:n_main]) if r["_file"] in held and r["_file"] not in extra_held])
    va_lineup = np.array([i for i, r in enumerate(rows[:n_main]) if r["_file"] in extra_held and r["_file"].startswith("lineup_")], int)
    va_online = np.array([i for i, r in enumerate(rows[:n_main]) if r["_file"] in extra_held and not r["_file"].startswith("lineup_")], int)
    va_test = np.arange(n_main, len(rows))
    vocab = Vocab([rows[i] for i in tr])
    if not args.no_pot:
        vocab.tables["pot"] = 2   # units carry their layer potential (fight_model.layer_potentials)
    vocab.save(os.path.join(OUT, "vocab.json"))
    x, y = featurize(rows, vocab)
    P(f"# 战斗预测网络（{len(rows)} 场战斗，{len(files)} 个数据文件；训练 {len(tr)}，留出 {len(va)}，留出的是整份文件）\n")
    P(f"特征化 {time.time() - t0:.0f} 秒；干员 {vocab.size('op') - 1} 个，出怪组 {vocab.size('stream') - 1} 种。")
    models = []
    for k in range(args.models):
        models.append(train_one(x, y, tr, vocab, args, seed=k, device=device, log=P))
        torch.save(models[-1].state_dict(), os.path.join(OUT, f"fightnet_{k}.pt"))
    evaluate(models, x, y, rows, va, device, "留出数据（原始对局）", P)
    if len(va_online):
        evaluate(models, x, y, rows, va_online, device, "留出：策略自己的棋盘（最新 3 份在线校正）", P)
    if len(va_lineup):
        evaluate(models, x, y, rows, va_lineup, device, "留出：阵容衍生棋盘（训练用到的阵容）", P)
    if len(va_test):
        evaluate(models, x, y, rows, va_test, device, "留出：从未见过的 4 套阵容的棋盘", P)

    if args.extrapolate:
        hard = np.array([i for i in range(len(rows)) if rows[i]["mode"] in ("规程等级16", "规程等级20")])
        easy = np.array([i for i in tr if rows[i]["mode"] not in ("规程等级16", "规程等级20")])
        if len(hard) > 100:
            ex = [train_one(x, y, easy, vocab, args, seed=100 + k, device=device, log=P) for k in range(min(3, args.models))]
            evaluate(ex, x, y, rows, hard, device, "外推：不用规程等级16/20训练，在这两档上评估", P)
    P(f"\n总耗时 {(time.time() - t0) / 60:.0f} 分钟。")


if __name__ == "__main__":
    main()
