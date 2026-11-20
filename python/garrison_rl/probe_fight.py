"""
What the fight model thinks of logged boss-round boards moved to another difficulty: every logged round-14 board
(real or what-if) is scored as if fought at --mode, and the ones with the highest predicted kill chance are printed with
each network's own estimate (their spread shows how far to trust it).

  python probe_fight.py --mode 规程等级20 --top 5
"""
import argparse
import copy
import glob
import json
import os

import numpy as np
import torch

from fight_model import FightNet, Vocab, featurize, load_rows, to_t
from train_fight import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="规程等级20")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--dump", help="also write the shown boards (at --mode) as JSON for FightCheckProbe")
    args = ap.parse_args()
    device = torch.device("cuda")
    vocab = Vocab(path=os.path.join(OUT, "vocab.json"))
    models = []
    for f in sorted(glob.glob(os.path.join(OUT, "fightnet_*.pt"))):
        m = FightNet(vocab).to(device)
        m.load_state_dict(torch.load(f, map_location=device))
        m.eval()
        models.append(m)
    rows = [r for r in load_rows() if r["round"] == 14 and r["sim"]["Bosses"] > 0]
    moved = []
    for r in rows:
        q = copy.copy(r)
        q["mode"] = args.mode
        q["hidden"] = False
        moved.append(q)
    x, _ = featurize(moved, vocab)
    idx = np.arange(len(moved))
    kills, progs = [], []
    with torch.no_grad():
        for m in models:
            ks, ps = [], []
            for s in range(0, len(idx), 4096):
                p = m(to_t(x, idx[s:s + 4096], device))
                ks.append(torch.sigmoid(p["kill"]).cpu().numpy()); ps.append(p["prog"].cpu().numpy())
            kills.append(np.concatenate(ks)); progs.append(np.concatenate(ps))
    kills, progs = np.stack(kills), np.stack(progs)
    mean = kills.mean(0)
    print(f"{len(rows)} 个第 14 回合阵容，按 {args.mode} 估计；预测击杀率 ≥50% 的有 {(mean >= 0.5).sum()} 个，≥20% 的 {(mean >= 0.2).sum()} 个\n")
    seen = set()
    shown = 0
    dump = []
    for i in np.argsort(-mean):
        r = rows[i]
        key = (r["boss"], tuple(sorted((u[0], u[1], tuple(u[2])) for u in r["units"])))
        if key in seen:
            continue
        seen.add(key)
        shown += 1
        sim = r["sim"]
        print(f"#{shown} 首领 {r['boss']}  地图 {r['map']}  原难度 {r['mode']}（{'what-if' if r['whatIf'] else '实战'}，"
              f"真实模拟：{'击杀' if sim['BossesKilled'] >= sim['Bosses'] else '未击杀'}，打掉 {sim['bossProgress'] / max(1, sim['Bosses']):.0%}，{sim['seconds']} 秒）")
        print(f"   按 {args.mode} 预测击杀概率 {mean[i]:.0%}（5 个网络各自：{' '.join(f'{k:.0%}' for k in kills[:, i])}），"
              f"预测打掉 {progs[:, i].mean():.0%}")
        print(f"   商店等级 {r['shopLevel']}，血量 {r['hp']}；盟约层数 " + "，".join(f"{c} {v:g}" for c, v in sorted(r["layers"].items(), key=lambda kv: -kv[1])))
        print("   干员：" + "、".join(f"{u[0]}{'★' if u[1] else ''}{'[' + '/'.join(u[2]) + ']' if u[2] else ''}" for u in r["units"]))
        dump.append(dict(mode=args.mode, round=14, hp=r["hp"], shopLevel=r["shopLevel"], fieldBonus=r["fieldBonus"], map=r["map"],
                         boss=r["boss"], types=r["types"], units=r["units"], layers=r["layers"], predicted=float(mean[i]),
                         note=f"{r['mode'].replace('规程等级', 'L')} {'what-if' if r['whatIf'] else '实战'}"))
        on = [f[1] for f in r.get("field", [])]
        print(f"   上场 {len(on)} 人：" + "、".join(on) + "\n")
        if shown >= args.top:
            break
    if args.dump:
        json.dump(dump, open(args.dump, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
