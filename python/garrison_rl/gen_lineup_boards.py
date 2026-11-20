"""
Fight data the policy does not bring by itself: boards on the way to (and at) the saved lineups (「我的阵容」), fought in
the real simulation with the placement search the other fight data was made with (the gym's --relabel).

The fight model trained on the scripted expert's and the policy's own boards is right about those and blind to good
ones (a finished elite lineup: ~180 layers a round in the simulation, ~35 predicted; the 规程14 leader killed every
time, 4 % predicted), so a policy trained on it is never paid for building one. Three kinds of boards fill that in:
  - lineup: part or all of a saved lineup (its early / mid pieces before the finished ones), some of it elite, with
    layers from none to a strong run's (600 per covenant);
  - hybrid: a board the policy really brought, with some of its units replaced by a lineup's;
  - scaled: a policy board as it is, with more layers and more elites (what layers alone do to a fight).
Rounds 5-14 (a share of them the leader round), every difficulty. 阿戈尔 lineups are left out, and a few lineups are
held out whole (--part test builds boards from only those) to check the model on lineups it never saw.

  python gen_lineup_boards.py --n 40000 --part train
  python gen_lineup_boards.py --n 3000 --part test
"""
import argparse
import glob
import json
import os
import random
import subprocess
import time

from gym_client import GYM_EXE, MID_PLANNER, ROOT

OUT = os.path.join(ROOT, "python", "garrison_rl", "data", "fights_lineup")
ONLINE = os.path.join(ROOT, "python", "garrison_rl", "data", "fights_online")
LEVELS = [4, 6, 8, 11, 14, 16, 20]
LEVEL_WEIGHTS = [1, 1, 2, 2, 2, 2, 2]
# Lineups kept out of the training boards altogether.
HELD_OUT = ("莱塔尼亚 · 大嘴莫布", "谢拉格 · 小皇帝", "炎 · 劫匪之王拜松", "卡西米尔 · 不屈突袭")
KEYS = ("mode", "round", "hp", "shopLevel", "fieldBonus", "map", "boss", "types", "units", "layers")


class Tables:
    def __init__(self):
        ops = json.load(open(os.path.join(ROOT, "data", "operators.json"), encoding="utf-8"))["operators"]
        self.ops = {o["name"]: o for o in ops if 1 <= o.get("tier", 0) <= 6 and not o.get("isDiy") and not o.get("requiredStrategyId")}
        self.covs = {n: list(o.get("coreCovenants", [])) + list(o.get("additionalCovenants", [])) for n, o in self.ops.items()}
        self.need = {c["name"]: c.get("activateCount") for c in json.load(open(os.path.join(ROOT, "data", "covenants.json"), encoding="utf-8"))["covenants"]}
        self.by_tier = {t: [n for n, o in self.ops.items() if o["tier"] == t] for t in range(1, 7)}

    def lineups(self):
        out = []
        folder = os.path.join(os.environ.get("APPDATA", ""), "ASPAssistant", "lineups")
        for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            names = lambda key: list(dict.fromkeys(s["operatorName"] for s in d.get(key) or [] if s.get("operatorName") in self.ops))
            final = names("slots")
            if len(final) < 6:
                continue
            aegir = sum(1 for n in final if "阿戈尔" in self.covs[n])
            if aegir >= (self.need.get("阿戈尔") or 3) or "阿戈尔" in d.get("name", ""):
                continue
            out.append(dict(name=d.get("name", os.path.basename(f)), final=final,
                            early=[n for n in names("earlySlots") if n not in final], mid=[n for n in names("midSlots") if n not in final]))
        return out

    def active(self, names):
        count = {}
        for n in set(names):
            for c in self.covs.get(n, []):
                count[c] = count.get(c, 0) + 1
        return [c for c, k in count.items() if self.need.get(c) and k >= self.need[c]], count


def policy_rows(limit, rng):
    """Boards the policy brought (the online relabels of every run so far), a random sample."""
    files = [f for f in glob.glob(os.path.join(ONLINE, "*.jsonl")) if not os.path.basename(f).startswith("_")]
    rng.shuffle(files)
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("hidden") or not r.get("units") or r["round"] < 4:
                    continue
                rows.append({k: r[k] for k in KEYS})
        if len(rows) >= limit:
            break
    rng.shuffle(rows)
    return rows[:limit]


def layer_cap(rnd):
    """A strong run's layers per covenant by this round (the best reach 500-600 at the leader)."""
    return 60.0 * max(0, rnd - 4)


def add_layers(t, rng, units, rnd, layers, strength=None):
    """Layers for the covenants the units could have active: anything from none to a strong run's."""
    act, count = t.active([u[0] for u in units])
    strength = rng.random() if strength is None else strength
    out = dict(layers)
    for c in act:
        if rng.random() < 0.85:
            out[c] = round(max(out.get(c, 0), layer_cap(rnd) * strength * rng.uniform(0.4, 1.0)), 1)
    for c in count:
        if c not in out and rng.random() < 0.3:
            out[c] = round(rng.uniform(0, 15), 1)
    return {c: v for c, v in out.items() if v > 0}


def context(rng, base, rnd=None):
    """Map, boss, enemy types and the rest from a real board; a difficulty of its own, and the leader round for a share."""
    level = rng.choices(LEVELS, LEVEL_WEIGHTS)[0]
    if rnd is None:
        rnd = 14 if rng.random() < 0.28 else base["round"] if 5 <= base["round"] <= 13 else rng.randint(5, 13)
    shop = min(6, max(base["shopLevel"], 2 + (rnd >= 6) + (rnd >= 8) + (rnd >= 10) if rng.random() < 0.7 else base["shopLevel"]))
    return dict(mode=f"规程等级{level}", round=rnd, hp=round(rng.uniform(6, 40), 1), shopLevel=shop, fieldBonus=base["fieldBonus"],
                map=base["map"], boss=base["boss"], types=base["types"])


def lineup_board(t, rng, lineup, base):
    b = context(rng, base, 14 if rng.random() < 0.3 else rng.randint(5, 13))
    rnd = b["round"]
    progress = min(1.0, max(0.15, rng.uniform(0.1, 1.0) + (rnd - 9) * 0.06))
    final = rng.sample(lineup["final"], max(1, round(len(lineup["final"]) * progress)))
    names = list(final)
    pieces = lineup["mid"] + lineup["early"]
    rng.shuffle(pieces)
    while len(names) < min(8, 2 + rnd) and pieces:
        names.append(pieces.pop())
    elite_p = rng.random() * min(1.0, (rnd - 2) / 8)
    units = [[n, 1 if rng.random() < (elite_p if n in final else elite_p * 0.5) else 0, []] for n in names]
    # The bench: second copies on the way to an elite, a piece or two more, something else from the shop.
    for n in rng.sample(names, min(len(names), rng.randint(0, 4))):
        if not any(u[0] == n and u[1] for u in units):
            units.append([n, 0, []])
    for n in pieces[: rng.randint(0, 2)]:
        units.append([n, 0, []])
    for _ in range(rng.randint(0, 3)):
        tier = rng.randint(1, max(1, min(6, b["shopLevel"])))
        units.append([rng.choice(t.by_tier[tier]), 1 if rng.random() < elite_p * 0.3 else 0, []])
    equip(rng, units, base)
    b["units"] = units[:18]
    b["layers"] = add_layers(t, rng, b["units"], rnd, {}, None if rng.random() > 0.12 else 0.0)
    return b


def finished_board(t, rng, lineup, base):
    """The top end: a whole lineup (one piece missing at most), mostly elite, with a strong run's layers; the leader
    round more often than not, the harder difficulties more often. These are rare among the lineup boards, and the
    fight model was too careful about them (规程11-14: killed every time, 50-60 % predicted)."""
    rnd = 14 if rng.random() < 0.6 else rng.randint(8, 13)
    b = context(rng, base, rnd)
    b["mode"] = f"规程等级{rng.choices(LEVELS, [1, 1, 2, 3, 3, 2, 2])[0]}"
    b["shopLevel"] = rng.choice([5, 6, 6])
    final = list(lineup["final"])
    if rng.random() < 0.3:
        final.remove(rng.choice(final))
    elite_p = rng.uniform(0.6, 1.0)
    units = [[n, 1 if rng.random() < elite_p else 0, []] for n in final]
    pieces = lineup["mid"] + lineup["early"]
    rng.shuffle(pieces)
    for n in pieces[: rng.randint(0, 3)]:
        units.append([n, 1 if rng.random() < 0.3 else 0, []])
    for n in rng.sample(final, rng.randint(0, 3)):
        if not any(u[0] == n and u[1] for u in units):
            units.append([n, 0, []])
    equip(rng, units, base)
    b["units"] = units
    act, _ = t.active([u[0] for u in units])
    top = min(650.0, layer_cap(rnd) * 1.1)
    b["layers"] = {c: round(rng.uniform(0.25, 1.0) * top, 1) for c in act if rng.random() < 0.9}
    return b


def equip(rng, units, base):
    """The equipment of a real board, handed to this one's units (strongest-looking first: elites)."""
    sets = [u[2] for u in base["units"] if u[2]]
    rng.shuffle(sets)
    order = sorted(range(len(units)), key=lambda i: -units[i][1] + rng.random() * 0.5)
    for i, s in zip(order, sets):
        units[i][2] = list(s)


def hybrid_board(t, rng, lineups, base):
    b = context(rng, base)
    held = {u[0] for u in base["units"]}
    lineup = max(rng.sample(lineups, min(5, len(lineups))), key=lambda l: len(held & set(l["final"])) + rng.random())
    units = [[u[0], u[1], list(u[2])] for u in base["units"]]
    missing = [n for n in lineup["final"] if n not in held]
    rng.shuffle(missing)
    # The cheapest pieces that are not the lineup's make room.
    out = sorted((i for i, u in enumerate(units) if u[0] not in lineup["final"]), key=lambda i: (units[i][1], t.ops.get(units[i][0], {}).get("tier", 1)))
    for i, n in zip(out[: rng.randint(1, 6)], missing):
        units[i] = [n, 1 if rng.random() < 0.5 else 0, units[i][2]]
    b["units"] = units
    b["layers"] = add_layers(t, rng, units, b["round"], base["layers"], rng.random())
    return b


def aspire_board(t, rng, lineups, base):
    """A board the policy brought, moved toward a saved lineup (some units replaced, the layers it could have by
    then), at its own difficulty, round and HP: what online.py has fought next to the policy's own boards, so the fight
    model stays right about the boards the policy should be heading for."""
    b = hybrid_board(t, rng, lineups, base)
    for k in ("mode", "round", "hp", "shopLevel"):
        b[k] = base[k]
    b["layers"] = add_layers(t, rng, b["units"], b["round"], base["layers"], rng.random())
    return b


def scaled_board(t, rng, base):
    b = context(rng, base)
    units = [[u[0], 1 if u[1] or rng.random() < 0.3 else 0, list(u[2])] for u in base["units"]]
    k = rng.uniform(1.5, 5.0)
    b["units"] = units
    b["layers"] = {c: round(min(700.0, v * k), 1) for c, v in base["layers"].items()}
    if rng.random() < 0.5:
        b["layers"] = add_layers(t, rng, units, b["round"], b["layers"], rng.random())
    return b


def build(n, part, seed, finished=False):
    rng = random.Random(seed)
    t = Tables()
    lineups = t.lineups()
    mine = [l for l in lineups if (l["name"] in HELD_OUT) == (part == "test")]
    base = policy_rows(max(2000, n), rng)
    print(f"{len(lineups)} lineups ({len(mine)} for {part}), {len(base)} policy boards", flush=True)
    boards = []
    for i in range(n):
        b0 = base[i % len(base)]
        kind = "finished" if finished else rng.choices(("lineup", "hybrid", "scaled"), (0.45, 0.35, 0.20 if part == "train" else 0.0))[0]
        b = (finished_board(t, rng, rng.choice(mine), b0) if kind == "finished" else lineup_board(t, rng, rng.choice(mine), b0) if kind == "lineup"
             else hybrid_board(t, rng, mine, b0) if kind == "hybrid" else scaled_board(t, rng, b0))
        b["seed"] = rng.randrange(1_000_000)
        b["src"] = f"{kind}_{part}"
        boards.append(b)
    return boards


def fight(boards, out, threads):
    inp = out.replace(".jsonl", ".in.tmp")
    with open(inp, "w", encoding="utf-8") as fh:
        for b in boards:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    if os.path.exists(out):
        os.remove(out)
    cmd = [GYM_EXE, "--data", os.path.join(ROOT, "data"), "--envs", str(threads), "--relabel", inp, "--out", out] + MID_PLANNER + ["--no-flying"]
    with open(out.replace(".jsonl", ".err.log"), "w", encoding="utf-8") as fe:
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=fe).returncode
    os.remove(inp)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--part", choices=("train", "test"), default="train")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--finished", action="store_true", help="only whole, mostly elite lineups with a strong run's layers")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    boards = build(args.n, args.part, args.seed, args.finished)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    for k in range(0, len(boards), args.chunk):
        out = os.path.join(OUT, f"lineup_{args.part}_{stamp}_{k // args.chunk:03d}.jsonl")
        code = fight(boards[k:k + args.chunk], out, args.threads)
        done = sum(1 for _ in open(out, encoding="utf-8")) if os.path.exists(out) else 0
        print(f"chunk {k // args.chunk}: {done}/{len(boards[k:k + args.chunk])} boards, exit {code}, {time.time() - t0:.0f}s so far", flush=True)


if __name__ == "__main__":
    main()
