"""
A learned stand-in for placement search + fight simulation (data from gen_fights.py).

One record = one fight: the board going in (operators with elite / equipment, covenant layers), the round, difficulty,
shop level, map, boss, the run's enemy types and the enemy group actually drawn. Targets:
  - leaks: enemies that got through or were left alive (the HP an ordinary round costs, capped at 10);
  - boss rounds: share of the boss HP taken, killed, and the seconds past the fixed 115 s it took (the HP it costs);
  - layers: what each covenant gained over the round (before the fight and from it);
  - handled share (handled / worth).
The model is an entity transformer over the operators plus a context token; an ensemble of them gives the mean and
the spread (disagreement), used as uncertainty.
"""
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "python", "garrison_rl", "data", "fights")
MAX_UNITS = 22
BOSS_SECONDS = 115.0
PROFESSIONS = ["近卫", "术师", "辅助", "狙击", "重装", "特种", "医疗", "先锋"]
OP_STATIC = 11 + len(PROFESSIONS)


def leader_scale(level):
    """Leader HP of 规程等级 n against level 1 (GarrisonSpawnStore.LeaderScale): how much harder the level is."""
    if level <= 8:
        return (level + 2) / 3.0
    if level <= 14:
        return 10 / 3.0 + (level - 8) * 5 / 6.0
    return 25 / 3.0 + (level - 14) * 25 / 27.0


class OpTable:
    """Fixed numbers for each operator (normal / elite) from data/operators.json and operator_profiles.json, so an
    operator the model has rarely seen still comes with its tier, class, stats and covenants."""

    def __init__(self):
        ops = json.load(open(os.path.join(ROOT, "data", "operators.json"), encoding="utf-8"))["operators"]
        prof = json.load(open(os.path.join(ROOT, "data", "operator_profiles.json"), encoding="utf-8"))["operators"]
        self.covs = {o["name"]: list(o.get("coreCovenants", [])) + list(o.get("additionalCovenants", [])) for o in ops}
        self.feats = {}
        for o in ops:
            p = prof.get(o["name"], {})
            for elite in (0, 1):
                st = p.get("eliteStats" if elite else "normalStats") or {}
                f = np.zeros(OP_STATIC, np.float32)
                f[0] = o.get("tier", 1) / 6
                f[1] = np.log1p(st.get("atk", 0)) / 8
                f[2] = np.log1p(st.get("hp", 0)) / 9
                f[3] = st.get("def", 0) / 1000
                f[4] = st.get("res", 0) / 50
                f[5] = st.get("interval", 0) / 3
                f[6] = np.log1p(st.get("dps", 0)) / 7
                f[7] = st.get("block", p.get("blockCount", 0)) / 3
                f[8] = st.get("skillFactor", 0)
                f[9] = min(10, st.get("skillTargets", 1)) / 10
                f[10] = 1.0 if p.get("position") == "远程" else 0.0
                if p.get("profession") in PROFESSIONS:
                    f[11 + PROFESSIONS.index(p["profession"])] = 1
                self.feats[(o["name"], elite)] = f

    def of(self, name, elite):
        return self.feats.get((name, int(elite)), np.zeros(OP_STATIC, np.float32))


POTENTIAL = os.path.join(ROOT, "python", "garrison_rl", "data", "layer_potential.json")
_POT = None


def layer_potentials():
    """Per operator (plain / elite), the layers its stacking trait feeds each covenant in a round, by the rules
    (the gym's spec: LayerCounter.Potential at typical counts, 获得时 at a fifth). Read from the gym once and kept
    in data/layer_potential.json (delete it after the operators' data changes)."""
    global _POT
    if _POT is None:
        if not os.path.exists(POTENTIAL):
            from gym_client import GymClient
            g = GymClient(envs=1)
            covs = [c["name"] for c in g.spec["covenants"]]
            pick = lambda o, k: dict((c, v) for c, v in zip(covs, o[k]) if v > 0)
            # Per operator: [plain, elite], each the typical count and the count with capped traits at their cap.
            table = {o["name"]: [dict(typ=pick(o, a), cap=pick(o, b)) for a, b in (("layersNormal", "layersNormalCap"), ("layersElite", "layersEliteCap"))]
                     for o in g.spec["operators"]}
            g.close()
            json.dump(table, open(POTENTIAL, "w", encoding="utf-8"), ensure_ascii=False)
        _POT = json.load(open(POTENTIAL, encoding="utf-8"))
    return _POT


def kill_overtime(sim):
    """Seconds past 115 s until the bosses are down (extrapolated when they were not), capped at 600."""
    if sim["BossesKilled"] >= sim["Bosses"] and sim["Finished"]:
        t = sim["seconds"]
    else:
        share = sim["bossProgress"] / max(1, sim["Bosses"])
        t = 715.0 if share < 0.02 else sim["seconds"] / share
    return min(600.0, max(0.0, t - BOSS_SECONDS))


class Vocab:
    """Ids for operators, items, covenants, maps, bosses, enemy types and enemy groups (0 = unknown)."""

    def __init__(self, rows=None, path=None):
        if path:
            self.tables = json.load(open(path, encoding="utf-8"))
            return
        t = {k: {} for k in ("op", "item", "cov", "map", "boss", "type", "stream")}

        def add(kind, v):
            if v is not None and v not in t[kind]:
                t[kind][v] = len(t[kind]) + 1
        for r in rows:
            for u in r["units"]:
                add("op", u[0])
                for e in u[2]:
                    add("item", e.rstrip("+"))
            for c in r["layers"]:
                add("cov", c)
            for part in (r.get("beforeFight") or {}, r.get("fromFight") or {}):
                for c in part:
                    add("cov", c)
            add("map", r["map"]); add("boss", r["boss"]); add("stream", r["stream"])
            for ty in r["types"]:
                add("type", ty)
        self.tables = t

    def id(self, kind, v):
        return self.tables[kind].get(v, 0)

    def size(self, kind):
        return len(self.tables[kind]) + 1

    def save(self, path):
        json.dump(self.tables, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def load_rows(files=None):
    rows = []
    for f in files or sorted(glob.glob(os.path.join(DATA, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue   # a line cut off by a crash
                if r["sim"] is None or not r["units"]:
                    continue   # nobody fielded: nothing to learn
                r["_file"] = os.path.basename(f)
                r.pop("field", None)   # the placement: not a feature, and the bulk of a row
                rows.append(r)
    return rows


_OPS = None


def featurize(rows, vocab):
    global _OPS
    _OPS = _OPS or OpTable()
    n, nc = len(rows), vocab.size("cov")
    unit_op = np.zeros((n, MAX_UNITS), np.int64)
    unit_f = np.zeros((n, MAX_UNITS, 2), np.float32)      # elite, number of items
    unit_items = np.zeros((n, MAX_UNITS, 2), np.int64)
    unit_adv = np.zeros((n, MAX_UNITS, 2), np.float32)
    unit_static = np.zeros((n, MAX_UNITS, OP_STATIC), np.float32)
    unit_cov = np.zeros((n, MAX_UNITS, nc), np.float32)
    # Layers per round each unit's trait feeds each covenant (the rules' count): the layer head sums these.
    # vocab "pot" 1: the typical count; 2: the count at the traits' caps (summed by the head), the typical one beside it.
    use_pot = int(vocab.tables.get("pot") or 0)
    unit_pot = np.zeros((n, MAX_UNITS, nc), np.float32) if use_pot else None
    unit_pot2 = np.zeros((n, MAX_UNITS, nc), np.float32) if use_pot >= 2 else None
    pots = layer_potentials() if use_pot else {}
    layers = np.zeros((n, nc), np.float32)
    ctx = np.zeros((n, 7), np.float32)                     # round, difficulty, shop level, field bonus, hidden core, hp, log leader scale
    cat = np.zeros((n, 3), np.int64)                       # map, boss, stream
    types = np.zeros((n, vocab.size("type")), np.float32)
    y_leak = np.zeros(n, np.float32); y_boss = np.zeros(n, np.float32)
    y_prog = np.zeros(n, np.float32); y_kill = np.zeros(n, np.float32); y_over = np.zeros(n, np.float32)
    y_layer = np.zeros((n, nc), np.float32); y_hand = np.zeros(n, np.float32)
    for i, r in enumerate(rows):
        for j, u in enumerate(r["units"][:MAX_UNITS]):
            unit_op[i, j] = vocab.id("op", u[0])
            unit_f[i, j] = (u[1], len(u[2]) / 2)
            unit_static[i, j] = _OPS.of(u[0], u[1])
            for c in _OPS.covs.get(u[0], []):
                unit_cov[i, j, vocab.id("cov", c)] = 1
            if use_pot and u[0] in pots:
                pt = pots[u[0]][1 if u[1] else 0]
                for c, v in pt["cap" if use_pot >= 2 else "typ"].items():
                    unit_pot[i, j, vocab.id("cov", c)] = v
                if use_pot >= 2:
                    for c, v in pt["typ"].items():
                        unit_pot2[i, j, vocab.id("cov", c)] = v
            for k, e in enumerate(u[2][:2]):
                unit_items[i, j, k] = vocab.id("item", e.rstrip("+"))
                unit_adv[i, j, k] = 1.0 if e.endswith("+") else 0.0
        for c, v in r["layers"].items():
            layers[i, vocab.id("cov", c)] = v / 100
        level = int(r["mode"].replace("规程等级", "")) if r["mode"].startswith("规程等级") else 0
        ctx[i] = (r["round"] / 15, level / 20, r["shopLevel"] / 6, r["fieldBonus"], 1.0 if r["hidden"] else 0.0, r["hp"] / 40,
                  np.log(leader_scale(max(1, level))) / 2)
        cat[i] = (vocab.id("map", r["map"]), vocab.id("boss", r["boss"]), vocab.id("stream", r["stream"]))
        for ty in r["types"]:
            types[i, vocab.id("type", ty)] = 1
        sim = r["sim"]
        boss = sim["Bosses"] > 0
        y_boss[i] = 1.0 if boss else 0.0
        y_leak[i] = 0 if boss else min(10, sim["Leaked"] + sim["Alive"])
        if boss:
            y_prog[i] = min(1.0, sim["bossProgress"] / max(1, sim["Bosses"]))
            y_kill[i] = 1.0 if sim["BossesKilled"] >= sim["Bosses"] else 0.0
            y_over[i] = kill_overtime(sim)
        for part in (r.get("beforeFight") or {}, r.get("fromFight") or {}):
            for c, v in part.items():
                y_layer[i, vocab.id("cov", c)] += v
        y_hand[i] = min(1.0, r["handled"] / max(1e-6, r["worth"]))
    x = dict(unit_op=unit_op, unit_f=unit_f, unit_items=unit_items, unit_adv=unit_adv, unit_static=unit_static, unit_cov=unit_cov,
             layers=layers, ctx=ctx, cat=cat, types=types)
    if use_pot:
        x["unit_pot"] = unit_pot
    if use_pot >= 2:
        x["unit_pot2"] = unit_pot2
    y = dict(leak=y_leak, boss=y_boss, prog=y_prog, kill=y_kill, over=y_over, layer=y_layer, hand=y_hand)
    return x, y


class FightNet(nn.Module):
    def __init__(self, vocab, d=192, layers=4, heads=6, id_drop=0.15):
        super().__init__()
        self.id_drop = id_drop          # share of operators / groups whose learned id is hidden in training (lean on the stats)
        nc = vocab.size("cov")
        # vocab "pot": the units carry their layer potential, and the layers gained are those potentials summed with a
        # learned share per unit (is it fielded, is its covenant active, how far the fight lets it trigger) plus a rest.
        self.pot = int(vocab.tables.get("pot") or 0)
        self.op = nn.Embedding(vocab.size("op"), d)
        self.item = nn.Embedding(vocab.size("item"), d)
        self.unit_f = nn.Sequential(nn.Linear(2 + 2 + OP_STATIC + nc + nc * self.pot, d), nn.GELU(), nn.Linear(d, d))
        if self.pot:
            self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, nc))
            with torch.no_grad():
                self.gate[-1].weight.mul_(0.1)
                self.gate[-1].bias.fill_(-1.5 if self.pot == 1 else -1.0)   # about half of the potential to start with
        self.ctx = nn.Sequential(nn.Linear(7 + nc + vocab.size("type"), d), nn.GELU(), nn.Linear(d, d))
        self.map = nn.Embedding(vocab.size("map"), d)
        self.boss = nn.Embedding(vocab.size("boss"), d)
        self.stream = nn.Embedding(vocab.size("stream"), d)
        self.cov = nn.Linear(nc, d)                      # the layers again, added to every operator token
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 5 + nc))
        self.nc = nc
        # Start near typical values (about 1 leak, 30 s over, small layer gains), not at the output's midpoints.
        with torch.no_grad():
            b = self.head[-1].bias
            b.zero_()
            b[0] = -2.0      # leak: 10 * sigmoid(-2) ~ 1.2
            b[3] = -1.0      # overtime: 100 * softplus(-1) ~ 31 s
            b[5:] = -4.0     # layer gains: 10 * softplus(-4) ~ 0.18

    def forward(self, x):
        B = x["unit_op"].shape[0]
        pad = x["unit_op"] == 0
        items = (self.item(x["unit_items"]) * (x["unit_items"] > 0).unsqueeze(-1)).sum(2)
        op_id, stream_id = x["unit_op"], x["cat"][:, 2]
        if self.training and self.id_drop > 0:
            op_id = torch.where((torch.rand(op_id.shape, device=op_id.device) < self.id_drop) & ~pad, torch.zeros_like(op_id), op_id)
            stream_id = torch.where(torch.rand(stream_id.shape, device=op_id.device) < self.id_drop, torch.zeros_like(stream_id), stream_id)
        feats = torch.cat([x["unit_f"], x["unit_adv"], x["unit_static"], x["unit_cov"]] + ([x["unit_pot"] / 10] if self.pot else []) + ([x["unit_pot2"] / 10] if self.pot >= 2 else []), -1)
        u = self.op(op_id) + items + self.unit_f(feats) + self.cov(x["layers"]).unsqueeze(1)
        c = self.ctx(torch.cat([x["ctx"], x["layers"], x["types"]], -1)) + self.map(x["cat"][:, 0]) + self.boss(x["cat"][:, 1]) + self.stream(stream_id)
        h = torch.cat([c.unsqueeze(1), u], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        h = self.norm(self.body(h, src_key_padding_mask=mask))
        pooled = (h[:, 1:] * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        o = self.head(torch.cat([h[:, 0], pooled], -1))
        layer = F.softplus(o[:, 5:]) * 10
        if self.pot:
            # Per unit and covenant: the share of its potential it really stacks (0-3: caps raised, more triggers).
            g = (3 if self.pot == 1 else 2) * torch.sigmoid(self.gate(torch.cat([h[:, 1:], h[:, :1].expand(-1, h.shape[1] - 1, -1)], -1)))
            layer = layer + (g * x["unit_pot"] * (~pad).unsqueeze(-1)).sum(1)
        # leak (0-10), boss progress (0-1), kill logit, overtime (s / 100), handled share (0-1), layer gains (/ 10)
        return dict(leak=10 * torch.sigmoid(o[:, 0]), prog=torch.sigmoid(o[:, 1]), kill=o[:, 2], over=F.softplus(o[:, 3]) * 100,
                    hand=torch.sigmoid(o[:, 4]), layer=layer)


def losses(p, y):
    boss = y["boss"]
    normal = 1 - boss
    l_leak = (F.smooth_l1_loss(p["leak"], y["leak"], reduction="none") * normal).sum() / normal.sum().clamp(min=1)
    l_prog = (F.smooth_l1_loss(p["prog"] * 10, y["prog"] * 10, reduction="none") * boss).sum() / boss.sum().clamp(min=1)
    l_kill = (F.binary_cross_entropy_with_logits(p["kill"], y["kill"], reduction="none") * boss).sum() / boss.sum().clamp(min=1)
    l_over = (F.smooth_l1_loss(p["over"] / 20, y["over"] / 20, reduction="none") * boss).sum() / boss.sum().clamp(min=1)
    l_hand = F.smooth_l1_loss(p["hand"] * 10, y["hand"] * 10)
    # Huber with a wide quadratic part: the mean of the layers gained, not their median (the big gains were cut short).
    l_layer = F.huber_loss(p["layer"], y["layer"], delta=10.0) / 5
    total = l_leak + l_prog + l_kill + l_over + 0.5 * l_hand + 0.5 * l_layer
    return total, dict(leak=l_leak, prog=l_prog, kill=l_kill, over=l_over, hand=l_hand, layer=l_layer)


def to_t(x, idx, device):
    return {k: torch.as_tensor(v[idx], device=device) for k, v in x.items()}
