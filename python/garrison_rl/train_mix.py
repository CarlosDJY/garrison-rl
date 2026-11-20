"""
PPO over several difficulties at once, fights answered by the learned fight model (surrogate.py), with:
  - mixed difficulties: one gym per 规程等级, the policy told the level; envs are moved every --mix-every updates
    toward the levels at the edge of what it can do (recent clear rate c: weight c(1 - c), at least --min-envs each);
  - shaping by the fight model: potential = how close the board is to beating the clear round's leader
    (Surrogate.potential), x --shape-pot, potential-based (the optimal policy is unchanged);
  - the scripted 跟阵容 policy as a prior: imitation loss toward its action (--bc, fading to 0 over --bc-updates),
    optionally a pure imitation start (--bc-pretrain updates, DAgger-style: its actions with chance --bc-beta);
  - online correction of the fight model (online.py): every --refresh-every updates a sample of the policy's boards
    is fought in the real simulation and the models are fine-tuned on them; the gap is logged;
  - evaluation on every level with the fight model (--eval-runs held-out seeds) and in the real simulation;
  - a target lineup (--plan): the scripted 跟阵容's choice from the saved lineups, locked at round 5, is an input,
    and the progress toward it a potential (x --shape-plan);
  - a large effective batch (--minibatch x --accum per optimizer step) and a KL anchor to the starting policy
    (--kl-anchor x KL(start || policy)), which keep the policy from drifting off late in a long run.

  python train_mix.py --levels 1,4,6,8,11,14 --init runs/ppo_sur2/best_规程等级4.pt --tag mix1
"""
import argparse
import collections
import re
import concurrent.futures
import ctypes
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from gym_client import MID_PLANNER, ROOT, GymClient, MultiGym
from model import Encoder, Policy, load_compatible, to_torch
from train_ppo import A_MAX, pad_actions


STACK_TARGET = (500.0, 300.0, 200.0, 200.0)


class DrawTable:
    """Which 盟约 a match is given (--sp-draw): for the match's leader, each main / second 盟约 in proportion to its recent
    win rate against that leader, with a floor so that none stops being played. Kept apart for the 规程 16+ runs (their
    normal rounds are the same) and the lower ones; a cell with few runs leans on the 盟约's rate over all leaders."""

    def __init__(self, mains, seconds, window=6000, prior=20.0, floor_main=0.05, floor_second=0.03):
        self.mains, self.seconds = list(mains), list(seconds)
        self.window, self.prior = window, prior
        self.floors = {"main": floor_main, "second": floor_second}
        self.rows = {"hi": collections.deque(maxlen=window), "lo": collections.deque(maxlen=window)}

    @staticmethod
    def tier(level):
        return "hi" if int(level) >= 16 else "lo"

    def add(self, level, info):
        if info.get("focus") and info.get("boss"):
            self.rows[self.tier(level)].append((info["boss"], info["focus"], info.get("second"), 1.0 if info.get("cleared") else 0.0))

    def rates(self, tier, kind):
        """{leader: {盟约: (shrunk win rate, runs)}} and the 盟约's rates over all leaders."""
        names = self.mains if kind == "main" else self.seconds
        col = 1 if kind == "main" else 2
        rows = self.rows[tier]
        if not rows or not names:
            return {}, {}
        base = sum(r[3] for r in rows) / len(rows)
        tot = {c: [0.0, 0] for c in names}
        cell = {}
        for r in rows:
            c = r[col]
            if c not in tot:
                continue
            tot[c][0] += r[3]; tot[c][1] += 1
            x = cell.setdefault(r[0], {}).setdefault(c, [0.0, 0])
            x[0] += r[3]; x[1] += 1
        over = {c: (w + self.prior * base) / (n + self.prior) for c, (w, n) in tot.items()}
        out = {}
        for boss, d in cell.items():
            out[boss] = {c: ((d.get(c, [0, 0])[0] + self.prior * over[c]) / (d.get(c, [0, 0])[1] + self.prior), d.get(c, [0, 0])[1]) for c in names}
        return out, over

    def weights(self, tier, best=False):
        """The message for the simulator: per leader each 盟约's share of the draws (best: all on the highest rate)."""
        msg = {"cmd": "weights", "main": {}, "second": {}}
        for kind in ("main", "second"):
            rates, _ = self.rates(tier, kind)
            floor = self.floors[kind]
            for boss, d in rates.items():
                if best:
                    top = max(d, key=lambda c: d[c][0])
                    msg[kind][boss] = {c: (1.0 if c == top else 0.0) for c in d}
                    continue
                total = sum(v[0] for v in d.values())
                n = len(d)
                msg[kind][boss] = {c: (floor + (1 - floor * n) * (v[0] / total) if total > 0 else 1.0 / n) for c, v in d.items()}
        return msg

    def send(self, gym, best=False):
        for c in getattr(gym, "clients", [gym]):
            if not hasattr(c, "procs"):
                continue
            msg = self.weights(self.tier(c.level), best)
            for p in c.procs:
                p.write(msg)

    def save(self, path):
        json.dump({"mains": self.mains, "seconds": self.seconds, "rows": {k: list(v) for k, v in self.rows.items()}},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)

    def load(self, path):
        d = json.load(open(path, encoding="utf-8"))
        for k, v in d.get("rows", {}).items():
            self.rows[k].extend(tuple(x) for x in v)
        return self

    def report(self, tier, name=lambda x: x, boss_name=lambda x: x):
        rates, _ = self.rates(tier, "main")
        sec, _ = self.rates(tier, "second")
        out = []
        for boss in sorted(rates):
            d = rates[boss]
            line = f"{boss_name(boss)}：" + " ".join(f"{name(c)} {v[0] * 100:.0f}%({v[1]})" for c, v in sorted(d.items(), key=lambda kv: -kv[1][0]))
            if boss in sec:
                top = sorted(sec[boss].items(), key=lambda kv: -kv[1][0])[:3]
                line += "｜小盟约 " + " ".join(f"{name(c)} {v[0] * 100:.0f}%({v[1]})" for c, v in top)
            out.append(line)
        return out


DRAW = None        # the run's DrawTable (--sp-draw), also used by evaluate()
DRAW_BEST = False  # evaluate(): every leader's best 盟约 instead of a draw


def evaluate(model, enc, mode, device, runs, seed, extra, fighter=None):
    res = [None] * runs
    model.eval()
    if os.environ.get("ASP_SIM") == "sp" and os.environ.get("ASP_SP_ASYNC", "1") != "0":
        # The simulator processes step apart (as in training): the same runs and results, without every step waiting
        # for the slowest process. A process stops once all of its runs have ended.
        gym = MultiGym({mode: runs}, extra=extra)
        if DRAW is not None:
            DRAW.send(gym, best=DRAW_BEST)
        r = gym.clients[0].reset(seed)
        groups = gym.groups()
        cur = [r["obs"][lo:hi] for lo, hi in groups]
        ready, pending = list(range(len(groups))), {}
        with torch.no_grad():
            while ready or pending:
                if ready:
                    logits, _ = model(to_torch(pad_actions(enc.encode([o for gi in ready for o in cur[gi]])), device))
                    a, k = logits.argmax(-1).cpu().numpy(), 0
                    for gi in ready:
                        n = groups[gi][1] - groups[gi][0]
                        pending[gym.step_group(gi, a[k:k + n])] = gi
                        k += n
                    ready = []
                got, _ = concurrent.futures.wait(list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
                more, _ = concurrent.futures.wait(list(pending), timeout=0.004)
                for f in got | more:
                    gi = pending.pop(f)
                    rg = f.result()
                    lo, hi = groups[gi]
                    for j, info in enumerate(rg["info"]):
                        if info and res[lo + j] is None:
                            res[lo + j] = info
                    cur[gi] = rg["obs"]
                    if any(res[i] is None for i in range(lo, hi)):
                        ready.append(gi)
                ready.sort()
    else:
        gym = GymClient(mode=mode, envs=runs, extra=extra, fighter=fighter)
        r = gym.reset(seed)
        with torch.no_grad():
            while any(x is None for x in res):
                logits, _ = model(to_torch(pad_actions(enc.encode(r["obs"])), device))
                r = gym.step(logits.argmax(-1).cpu().numpy())
                for i, info in enumerate(r["info"]):
                    if info and res[i] is None:
                        res[i] = info
    gym.close()
    model.train()
    # Clear rate, rounds lived, and the top covenant's layers at the end (how far the stacking went).
    return (float(np.mean([x["cleared"] for x in res])), float(np.mean([x["rounds"] for x in res])),
            float(np.mean([(x.get("layers") or [0])[0] for x in res])))


class _PerfInfo(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def system_commit():
    """The whole machine's committed memory and its limit (GB). On this machine that limit, not RAM, runs out first."""
    try:
        pi = _PerfInfo()
        pi.cb = ctypes.sizeof(pi)
        ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
        return pi.CommitTotal * pi.PageSize / 2**30, pi.CommitLimit * pi.PageSize / 2**30
    except Exception:
        return float("nan"), float("nan")


def mem_str():
    """This process's committed memory (+ its children: the gyms), what torch holds on the GPU (on Windows that counts
    as committed memory too: allocated / reserved / peak reserved), and the machine's commit against its limit."""
    import psutil
    me = psutil.Process()
    priv = me.memory_info().private / 2**30
    kids = sum(c.memory_info().private for c in me.children(recursive=True)) / 2**30
    used, limit = system_commit()
    return (f"mem {priv:.1f}+{kids:.1f}G gpu {torch.cuda.memory_allocated() / 2**30:.1f}/{torch.cuda.memory_reserved() / 2**30:.1f}"
            f"/{torch.cuda.max_memory_reserved() / 2**30:.1f}G sys {used:.1f}/{limit:.1f}G")


def allocate(modes, recent, total, min_envs, opened, target=0.35, spread=0.25):
    """Envs per open level: at least min_envs each, the rest by how close the level's recent clear rate c is to the
    target (weight exp(-(c - target)^2 / 2 spread^2), at least 0.03): the hard-but-open levels get most of the play,
    the solved ones little (Stronghold-RL's auto1 aims the mix at an expected win of 0.35). Closed levels get none."""
    open_modes = [m for m in modes if m in opened]
    w = {}
    for m in open_modes:
        c = np.mean(recent[m]) if len(recent[m]) >= 30 else target
        w[m] = max(0.03, float(np.exp(-((c - target) ** 2) / (2 * spread ** 2))))
    free = total - min_envs * len(open_modes)
    tot = sum(w.values())
    alloc = {m: min_envs + int(free * w[m] / tot) for m in open_modes}
    left = total - sum(alloc.values())
    for m in sorted(open_modes, key=lambda m: -w[m])[:left]:
        alloc[m] += 1
    return alloc


def open_levels(modes, recent, opened, gate, n_min=100):
    """A level opens (for good) once the level before it clears at least `gate` of its last runs (n_min of them at
    least); the levels are in order of difficulty. Returns the newly opened ones."""
    new = []
    for k in range(1, len(modes)):
        prev, m = modes[k - 1], modes[k]
        if m not in opened and prev in opened and len(recent[prev]) >= n_min and np.mean(recent[prev]) >= gate:
            opened.add(m)
            new.append(m)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,8,11,14,16,20", help="in order of difficulty")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--min-envs", type=int, default=4)
    ap.add_argument("--mix-every", type=int, default=20)
    ap.add_argument("--open-first", type=int, default=2, help="levels open from the start (the rest open in order)")
    ap.add_argument("--open-gate", type=float, default=0.35, help="clear rate of the level before that opens the next")
    ap.add_argument("--target-clear", type=float, default=0.35)
    ap.add_argument("--spread", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=300, help="recent runs per level for the clear rates")
    ap.add_argument("--lr-warmup", type=int, default=20, help="updates of linear lr ramp after the value warm-up (fresh starts)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--value-warmup", type=int, default=5)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--accum", type=int, default=1, help="minibatches per optimizer step")
    ap.add_argument("--kl-anchor", type=float, default=0.0, help="x KL(starting policy || policy)")
    ap.add_argument("--plan", action="store_true", help="the target lineup as input (gym --plan)")
    ap.add_argument("--plan-lock", type=int, default=5, help="the round the target lineup is fixed from (1: once the core group is kept)")
    ap.add_argument("--plan-staged", action="store_true", help="the target follows the lineup's early / mid / finished members by round")
    ap.add_argument("--shape-plan", type=float, default=0.0, help="x progress toward the target lineup (potential)")
    ap.add_argument("--shape-plan-layers", type=float, default=1.0, help="x the target covenants' mean layers / 300 (capped at 2)")
    ap.add_argument("--shape-layers", type=float, default=1.0, help="x layers of all reached covenants / 1000 (capped at 3)")
    ap.add_argument("--r-layer", type=float, default=0.0,
                    help="x usable layers gained since the last fight / 180 (a finished elite lineup stacks about 180 a round): a real "
                         "reward, unlike the potentials above, which cancel out over a run; none from the deciding round on")
    ap.add_argument("--anchor-refresh", type=int, default=0,
                    help="every N updates the KL anchor becomes the current policy (a brake on fast drift, not a leash to the start); 0: the start")
    ap.add_argument("--r-plan", type=float, default=2.0,
                    help="run end: x target lineup completion x (0.5 + 0.5 x the clear round's boss share taken)")
    ap.add_argument("--r-stack", type=float, default=0.0,
                    help="run end: x (the four highest 盟约' layers / 500, / 300, / 200, / 200) / 3 (no cap: 1 at 500 + 300 + 200; a layer "
                         "of the third or fourth counts 2.5 times one of the first)")
    ap.add_argument("--r-plan-step", type=float, default=0.0,
                    help="x target lineup progress since the last fight (each member once when first held through a fight, once "
                         "when first 精锐; half its share of the target each): paid as it happens, where --r-plan pays at the run's end")
    ap.add_argument("--sim", default="asp", choices=["asp", "sp"],
                    help="sp: Stronghold-Protocol's simulator (gym_client.SPGymClient): every fight is the real one, no fight "
                         "model, no online correction; the in-run evaluation is the real one too")
    ap.add_argument("--sp-rehearsal", type=int, default=-1, help="--sim sp: layouts the game AI rehearses per prep (-1: its default)")
    ap.add_argument("--sp-boss-rehearsal", type=int, default=-1,
                    help="--sim sp: layouts rehearsed on the leader rounds whatever --sp-rehearsal says (-1: the same as the other rounds)")
    ap.add_argument("--async-steps", type=int, default=1,
                    help="--sim sp: the simulator processes step apart instead of waiting for the slowest each step (0: together)")
    ap.add_argument("--async-wait", type=float, default=0.02, help="--async-steps: seconds to gather further replies for one pass through the network")
    ap.add_argument("--sp-focus", default=None,
                    help="--sim sp: every match (training and evaluation) is given one of these core 盟约 ids (comma separated) as its "
                         "main, evenly: the plan the policy sees and the main the layout works for; --r-plan pays for following it")
    ap.add_argument("--sp-second", default=None,
                    help="--sim sp, with --sp-focus: every match is also given one of these 盟约 ids as its second")
    ap.add_argument("--sp-draw", type=int, default=0,
                    help="--sim sp: the given 盟约 (main and second) are drawn per leader in proportion to their recent win rate "
                         "against it (a 5 %% / 3 %% floor each), not evenly; the table is kept in runs/<tag>/draw.json")
    ap.add_argument("--sp-bands", default=None,
                    help="--sim sp: every match (training and evaluation) takes one of these 策略 ids (comma separated), evenly")
    ap.add_argument("--sp-fights", default=None,
                    help="--sim sp: every real fight of the training envs goes to <this prefix>_L<level>_<process>.jsonl (for the fight model)")
    ap.add_argument("--sp-picker", default=None,
                    help="--sim sp: a layout model's run directory (sp_layout.py); in training the candidate layouts are picked by "
                         "it instead of being simulated (--sp-rehearsal: how many candidates). The evaluations simulate them.")
    ap.add_argument("--init", default=None)
    ap.add_argument("--init-critic", default=None, help="a critic checkpoint (critic_last.pt of the run --init comes from)")
    ap.add_argument("--tag", default="mix")
    # Rewards (as train_ppo.py).
    ap.add_argument("--r-survive", type=float, default=0.1)
    ap.add_argument("--r-hp", type=float, default=2.0, help="x HP lost as a share of the start")
    ap.add_argument("--r-clear", type=float, default=5.0)
    ap.add_argument("--r-death", type=float, default=2.0)
    ap.add_argument("--r-boss", type=float, default=5.0)
    ap.add_argument("--r-kill", type=float, default=2.0)
    ap.add_argument("--r-fast", type=float, default=3.0)
    # Shaping.
    ap.add_argument("--shape-pot", type=float, default=5.0, help="x the fight model's potential (clear-round leader)")
    ap.add_argument("--shape-dps", type=float, default=1.0, help="x field DPS ÷ the leader's need (capped at --dps-cap)")
    ap.add_argument("--dps-cap", type=float, default=1.2)
    # Prior.
    ap.add_argument("--bc", type=float, default=0.0, help="imitation loss weight toward the scripted 跟阵容 action")
    ap.add_argument("--bc-updates", type=int, default=200, help="the imitation weight fades to 0 over these updates")
    ap.add_argument("--bc-pretrain", type=int, default=0, help="pure imitation updates first")
    ap.add_argument("--bc-beta", type=float, default=0.5, help="in the imitation start, chance the scripted policy acts")
    # Fight model.
    ap.add_argument("--surrogate-k", type=float, default=0.5)
    ap.add_argument("--refresh-every", type=int, default=25, help="0: no online correction")
    ap.add_argument("--refresh-n", type=int, default=1500)
    ap.add_argument("--refresh-steps", type=int, default=300)
    ap.add_argument("--replay", type=int, default=60000, help="original fights mixed into each correction")
    ap.add_argument("--resume", action="store_true", help="go on from runs/<tag>/state.pt (the anchor stays --init)")
    ap.add_argument("--max-mem-fails", type=int, default=30, help="updates in a row lost to a memory error before giving up")
    # Evaluation.
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-runs", type=int, default=64)
    ap.add_argument("--real-every", type=int, default=60)
    ap.add_argument("--real-levels", default="8,11,14,20")
    ap.add_argument("--sel-seed", type=int, default=9000, help="seeds of the in-run evaluations (checkpoint choice); "
                    "confirm with expert_eval.py on --seed 200000000 and 300000000")
    ap.add_argument("--real-runs", type=int, default=48)
    args = ap.parse_args()

    device = torch.device("cuda")
    modes = [f"规程等级{x}" for x in args.levels.split(",")]
    run_dir = os.path.join(ROOT, "python", "garrison_rl", "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    global DRAW
    if args.sim == "sp" and args.sp_draw and args.sp_focus:
        DRAW = DrawTable(args.sp_focus.split(","), [x for x in (args.sp_second or "").split(",") if x])
        if args.resume and os.path.exists(os.path.join(run_dir, "draw.json")):
            DRAW.load(os.path.join(run_dir, "draw.json"))
    log_path = os.path.join(ROOT, "reports", "playground", f"{args.tag}.md")

    def P(s):
        print(s, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    sp = args.sim == "sp"
    resume_fights = False
    if sp:
        os.environ["ASP_SIM"] = "sp"

        class _NoFighter:
            t_answer = 0.0
            record = False

        fighter = _NoFighter()
    else:
        from surrogate import Surrogate
        own_fights = os.path.join(run_dir, "fightnet")
        resume_fights = args.resume and os.path.isdir(own_fights) and any(f.startswith("fightnet_") for f in os.listdir(own_fights))
        fighter = Surrogate(path=own_fights, device=device, k=args.surrogate_k) if resume_fights else Surrogate(device=device, k=args.surrogate_k)
    refresher = None
    if args.refresh_every and not sp:
        from online import Refresher
        refresher = Refresher(fighter, args.tag, replay=args.replay, log=P)
    use_expert = args.bc > 0 or args.bc_pretrain > 0
    plan_flag = (["--plan", "--plan-lock", str(args.plan_lock)] + (["--plan-staged"] if args.plan_staged else [])) if args.plan else []
    gym_extra = ["--no-flying"] + (["--boards"] if args.shape_pot > 0 else []) + (["--expert"] if use_expert else []) + plan_flag
    if sp:
        gym_extra = [f"--sp:rehearsal={args.sp_rehearsal}"] if args.sp_rehearsal >= 0 else []
        if args.sp_boss_rehearsal >= 0:
            gym_extra.append(f"--sp:boss-rehearsal={args.sp_boss_rehearsal}")
        if args.sp_bands:
            gym_extra.append(f"--sp:bands={args.sp_bands}")
        if args.sp_focus:
            gym_extra.append(f"--sp:focus={args.sp_focus}")
        if args.sp_second:
            gym_extra.append(f"--sp:second={args.sp_second}")
        if args.sp_fights:
            fp = args.sp_fights if os.path.isabs(args.sp_fights) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.sp_fights)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            gym_extra.append(f"--sp:fights={fp}")
        if args.sp_picker:
            os.environ["ASP_SP_PICKER"] = os.path.join(run_dir, "..", args.sp_picker) if not os.path.isabs(args.sp_picker) else args.sp_picker
            gym_extra.append("--sp:picker")
    real_extra = MID_PLANNER + ["--no-flying", "--no-hidden-core"] + plan_flag

    recent = {m: collections.deque(maxlen=args.window) for m in modes}
    rets = {m: collections.deque(maxlen=args.window) for m in modes}     # each finished run's reward without the shaping
    plans = {m: collections.deque(maxlen=args.window) for m in modes}    # target lineup at the run's end: held, 精锐, layers
    stacks = {m: collections.deque(maxlen=args.window) for m in modes}   # each run: its top covenant's layers, layers gained in rounds 5-8 / 9-13
    opened = set(modes[:args.open_first])
    alloc = allocate(modes, recent, args.envs, args.min_envs, opened, args.target_clear, args.spread)
    gym = MultiGym(alloc, extra=gym_extra, fighter=fighter)
    enc = Encoder(gym.spec)
    model = Policy(enc).to(device)
    if args.init:
        init_state = torch.load(args.init, map_location=device)
        load_compatible(model, init_state)
    # A separate critic (same architecture, its own body): with a shared body the value regression rewrote the
    # policy's features (the policy drifted even with its own loss off). It predicts returns / VSCALE.
    critic = Policy(enc).to(device)
    if args.init:
        load_compatible(critic, init_state)
        del init_state
    if args.init_critic:
        critic.load_state_dict(torch.load(args.init_critic, map_location=device))
    VSCALE = 10.0
    anchor = None
    if args.kl_anchor > 0:
        anchor = Policy(enc).to(device)
        anchor.load_state_dict(model.state_dict())
        anchor.eval()
        for q in anchor.parameters():
            q.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    copt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
    P(f"\n# 混合难度 PPO（{args.tag}）\n\n参数：{vars(args)}\n")

    def reward_of(parts):
        if parts is None:
            return 0.0
        survived, lost, cleared, died, boss, killed, seconds, hidden, hidden_won = parts[:9]
        r = args.r_survive * survived - args.r_hp * lost + args.r_clear * cleared - args.r_death * died
        if len(parts) > 9:
            r += args.r_layer * parts[9] * 100 / 180
        if len(parts) > 10:
            r += args.r_plan_step * parts[10]
        if boss >= 0:
            r += args.r_boss * boss + killed * (args.r_kill + args.r_fast * max(0.0, 1 - seconds))
        return r

    def phi_of(r):
        dps = np.array([args.shape_dps * min(args.dps_cap, p[2]) + (args.shape_plan * p[3] if len(p) > 3 else 0.0)
                        + (args.shape_plan_layers * min(2.0, p[4]) + args.shape_layers * min(3.0, p[5]) if len(p) > 5 else 0.0)
                        for p in r["phi"]], np.float32)
        if args.shape_pot > 0:
            dps += args.shape_pot * fighter.potential([o.get("board") for o in r["obs"]])
        return dps

    best = -1.0
    seed_base = 10_000_000
    start = 1
    state_path = os.path.join(run_dir, "state.pt")

    def save_state(upd):
        torch.save(dict(model=model.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(), copt=copt.state_dict(),
                        upd=upd, best=best, seed_base=seed_base, alloc=alloc, opened=sorted(opened),
                        recent={m: list(v) for m, v in recent.items()}, rets={m: list(v) for m, v in rets.items()}),
                   state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    if args.resume:
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model"])
        critic.load_state_dict(st["critic"])
        opt.load_state_dict(st["opt"])
        copt.load_state_dict(st["copt"])
        start, best, seed_base = st["upd"] + 1, st["best"], st["seed_base"] + 1_000_000
        opened.update(st.get("opened", []))
        for m in modes:
            recent[m].extend(st["recent"].get(m, []))
            rets[m].extend(st["rets"].get(m, []))
        if st["alloc"] != alloc:
            alloc = st["alloc"]
            gym.close()
            gym = MultiGym(alloc, extra=gym_extra, fighter=fighter)
        if anchor is not None and args.anchor_refresh > 0:
            anchor.load_state_dict(model.state_dict())   # a moving anchor goes on from the resumed policy
        P(f"- 从第 {st['upd']} 次更新续跑（战斗模型：{'本轮校正过的' if resume_fights else '原始'}）")
        del st

    def new_gym():
        """A fresh set of gyms (after a memory error the pipes may be mid-message): retried until memory allows."""
        nonlocal gym, seed_base
        for k in range(30):
            try:
                try:
                    gym.close()
                except Exception:
                    pass
                gym = MultiGym(alloc, extra=gym_extra, fighter=fighter)
                seed_base += 1_000_000
                return gym.reset(seed_base)
            except (MemoryError, OSError) as e:
                P(f"  重建环境失败（{type(e).__name__}），60 秒后重试")
                gc.collect()
                time.sleep(60)
        raise RuntimeError("could not restart the gyms")

    r = gym.reset(seed_base)
    phi = phi_of(r)
    buf = None
    fails = 0
    for upd in range(start, args.updates + 1):
        try:
            t0 = time.time()
            pre = upd <= args.bc_pretrain
            bc_w = args.bc * max(0.0, 1 - (upd - args.bc_pretrain) / max(1, args.bc_updates)) if not pre else 1.0
            if upd > 1 and (upd - 1) % args.mix_every == 0:
                for m in open_levels(modes, recent, opened, args.open_gate):
                    P(f"- upd {upd} 开放 {m}（前一档近期通关率达到 {args.open_gate:.0%}）")
                new = allocate(modes, recent, args.envs, args.min_envs, opened, args.target_clear, args.spread)
                if new != alloc:
                    alloc = new
                    gym.close()
                    gym = MultiGym(alloc, extra=gym_extra, fighter=fighter)
                    seed_base += 1_000_000
                    r = gym.reset(seed_base)
                    phi = phi_of(r)
            # Linear lr ramp after the value warm-up on a fresh start (Adam starts empty); a resume has its moments.
            if anchor is not None and args.anchor_refresh > 0 and upd > start and (upd - 1) % args.anchor_refresh == 0:
                anchor.load_state_dict(model.state_dict())
            ramp = 1.0 if args.resume or args.lr_warmup <= 0 else min(1.0, max(0, upd - args.value_warmup) / args.lr_warmup)
            for gp in opt.param_groups:
                gp["lr"] = args.lr * max(ramp, 1e-3)
            t_enc = t_net = t_env = 0.0
            fighter.t_answer = 0.0
            N = gym.envs
            env_modes = gym.modes
            if len(getattr(gym, "_ep", [])) != N:
                gym._ep = np.zeros(N, np.float32)
            if DRAW is not None:
                DRAW.send(gym)
                if upd % 5 == 0:
                    DRAW.save(os.path.join(run_dir, "draw.json"))
            T = args.steps
            store = {k: np.zeros((T, N), np.int64 if k == "act" else np.float32) for k in ("act", "logp", "val", "rew", "done", "fight", "bound")}
            if len(getattr(gym, "_boss", [])) != N:
                gym._boss = np.zeros(N, np.float32)
            # The processes step apart (--async-steps): one that is simulating a fight holds up only its own envs, the
            # others go on. Each env still gets exactly T steps; only the order in which they are asked changes.
            groups = gym.groups() if args.async_steps and hasattr(gym, "groups") else None
            groups = groups or [(0, N)]
            cur = [r["obs"][lo:hi] for lo, hi in groups]      # each group's observations waiting for an action
            tcount = [0] * len(groups)
            ready = list(range(len(groups)))
            pending = {}
            n_fwd = 0

            def took(gi, rg):
                """A group's reply to its step tcount[gi]: rewards, bounds and run ends into the rollout."""
                lo, hi = groups[gi]
                t = tcount[gi]
                done_now = np.array(rg["done"], np.float32)
                bound = np.maximum(np.array([p is not None for p in rg["parts"]], np.float32), done_now)
                g_t = np.where(bound > 0, args.gamma, 1.0).astype(np.float32)
                phi_next = phi_of(rg)
                base = np.array([reward_of(p) for p in rg["parts"]], np.float32)
                # The clear round's boss share taken (the run's own, not the 隐秘核心's), for the plan bonus.
                for j, p in enumerate(rg["parts"]):
                    if p is not None and p[4] >= 0 and p[7] == 0:
                        gym._boss[lo + j] = p[4]
                for j, info in enumerate(rg["info"]):
                    if info and info.get("plan") and args.r_plan > 0:
                        pl = info["plan"]
                        base[j] += args.r_plan * (0.5 * pl["held"] + 0.5 * pl["elite"] + 0.5 * pl.get("held2", 0.0)) * (0.5 + 0.5 * float(gym._boss[lo + j]))
                    if info and args.r_stack > 0:
                        # The run's highest 盟约 against hand-played 规程 20 runs at the 隐秘核心 (the user's own: about
                        # 500 / 300 / 200, = 1; a strong player's: 600 / 300 / 200 / 200, = 1.4).
                        ls = (list(info.get("layers") or []) + [0, 0, 0, 0])[:4]
                        base[j] += args.r_stack * sum(l / t for l, t in zip(ls, STACK_TARGET)) / 3
                gym._ep[lo:hi] += base
                store["rew"][t, lo:hi] = base + g_t * phi_next * (1 - done_now) - phi[lo:hi]
                phi[lo:hi] = phi_next
                store["bound"][t, lo:hi] = bound
                store["fight"][t, lo:hi] = np.array(rg["fight"], np.float32)
                store["done"][t, lo:hi] = done_now
                for j, info in enumerate(rg["info"]):
                    if info:
                        i = lo + j
                        recent[env_modes[i]].append(float(info["cleared"]))
                        if DRAW is not None:
                            DRAW.add(re.sub(r"\D", "", env_modes[i]) or 0, info)
                        rets[env_modes[i]].append(float(gym._ep[i]))
                        gym._ep[i] = 0
                        gym._boss[i] = 0
                        if info.get("plan"):
                            pl = info["plan"]
                            plans[env_modes[i]].append((pl["held"], pl["elite"], pl["layers"]))
                        gn = info.get("gains") or []
                        stacks[env_modes[i]].append(((info.get("layers") or [0])[0], float(sum(gn[4:8])), float(sum(gn[8:13])), float(info.get("elites", 0))))
                cur[gi] = rg["obs"]
                tcount[gi] = t + 1

            while ready or pending:
                if ready:
                    idx = np.concatenate([np.arange(*groups[gi]) for gi in ready])
                    rows = np.concatenate([tcount[gi] * N + np.arange(*groups[gi]) for gi in ready])
                    t1 = time.time()
                    batch = pad_actions(enc.encode([o for gi in ready for o in cur[gi]]))
                    t_enc += time.time() - t1
                    # The rollout goes into arrays allocated once (not a list of per-step arrays concatenated every update).
                    if buf is None or buf["g"].shape[0] != T * N or any(buf[k].shape[1:] != v.shape[1:] for k, v in batch.items()):
                        buf = None
                        gc.collect()
                        buf = {k: np.empty((T * N,) + v.shape[1:], v.dtype) for k, v in batch.items()}
                    for k, v in batch.items():
                        buf[k][rows] = v
                    ex = batch["expert"]
                    ex[ex >= batch["kind"].shape[1]] = -1
                    t1 = time.time()
                    with torch.no_grad():
                        tb = to_torch(batch, device)
                        logits, _ = model(tb)
                        _, v = critic(tb)
                        v = v * VSCALE
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                        logp = dist.log_prob(a)
                    a_np = a.cpu().numpy()
                    logp_np, v_np = logp.cpu().numpy(), v.cpu().numpy()
                    t_net += time.time() - t1
                    n_fwd += 1
                    send = a_np.copy()
                    if pre:
                        use = np.random.rand(len(idx)) < args.bc_beta
                        send = np.where(use, np.where(ex >= 0, ex, -2), a_np)
                        a_np = np.where(use & (ex >= 0), ex, a_np)
                    k = 0
                    for gi in ready:
                        lo, hi = groups[gi]
                        t, n = tcount[gi], hi - lo
                        store["act"][t, lo:hi] = a_np[k:k + n]
                        store["logp"][t, lo:hi] = logp_np[k:k + n]
                        store["val"][t, lo:hi] = v_np[k:k + n]
                        if len(groups) == 1:
                            t1 = time.time()
                            took(gi, gym.step(send))
                            t_env += time.time() - t1
                        else:
                            pending[gym.step_group(gi, send[k:k + n])] = gi
                        k += n
                    ready = [gi for gi in ready if len(groups) == 1 and tcount[gi] < T]
                if pending:
                    t1 = time.time()
                    got, _ = concurrent.futures.wait(list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
                    # A moment for the other quick replies, so that they share the next pass through the network.
                    more, _ = concurrent.futures.wait(list(pending), timeout=args.async_wait)
                    t_env += time.time() - t1
                    for f in got | more:
                        gi = pending.pop(f)
                        took(gi, f.result())
                        if tcount[gi] < T:
                            ready.append(gi)
                    ready.sort()
            r = {"obs": [o for c in cur for o in c]}
            with torch.no_grad():
                _, last_v = critic(to_torch(pad_actions(enc.encode(r["obs"])), device))
            last_v = last_v.cpu().numpy() * VSCALE
            vals = store["val"]; rews = store["rew"]; dones = store["done"]; bounds = store["bound"]
            adv = np.zeros((T, N), np.float32)
            gae = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                nxt = last_v if t == T - 1 else vals[t + 1]
                nonterm = 1 - dones[t]
                g = np.where(bounds[t] > 0, args.gamma, 1.0)
                l = np.where(bounds[t] > 0, args.lam, 1.0)
                delta = rews[t] + g * nxt * nonterm - vals[t]
                gae = delta + g * l * nonterm * gae
                adv[t] = gae
            ret = adv + vals
            fights = store["fight"]
            aux_t = np.full((T, N), -1, np.float32)
            nxt_f = np.full(N, -1, np.float32)
            for t in reversed(range(T)):
                nxt_f = np.where(fights[t] >= 0, fights[t], np.where(dones[t] > 0, -1, nxt_f))
                aux_t[t] = nxt_f
            aux_f = aux_t.reshape(-1)
            flat = buf
            acts = store["act"].reshape(-1); old_logp = store["logp"].reshape(-1)
            adv_f = adv.reshape(-1); ret_f = ret.reshape(-1)
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
            M = T * N
            t_learn0 = time.time()
            stop = False
            kls, bcs, vls, ents, anc = [], [], [], [], []
            vwarm = upd <= args.value_warmup and not pre
            warm = upd <= args.value_warmup or pre
            # No imitation while the value warms up (it would move the policy unchecked).
            big = args.minibatch * args.accum
            for _ in range(args.epochs):
                perm = np.random.permutation(M)
                for s0 in range(0, M, big):
                    chunk = perm[s0:s0 + big]
                    act_on = not (stop or vwarm)
                    opt.zero_grad()
                    copt.zero_grad()
                    kl_c = 0.0
                    for s1 in range(0, len(chunk), args.minibatch):
                        idx = chunk[s1:s1 + args.minibatch]
                        share = len(idx) / len(chunk)
                        mb = to_torch({k: v[idx] for k, v in flat.items()}, device)
                        _, v = critic(mb)
                        vl = F.mse_loss(v, torch.as_tensor(ret_f[idx] / VSCALE, device=device))
                        (vl * share).backward()
                        vls.append(vl.item())
                        if not act_on:
                            continue
                        logits, _ = model(mb)
                        dist = torch.distributions.Categorical(logits=logits)
                        a_t = torch.as_tensor(acts[idx], device=device)
                        lp = dist.log_prob(a_t)
                        ratio = torch.exp(lp - torch.as_tensor(old_logp[idx], device=device))
                        A_t = torch.as_tensor(adv_f[idx], device=device)
                        pg = -torch.min(ratio * A_t, ratio.clamp(1 - args.clip, 1 + args.clip) * A_t).mean()
                        ent = dist.entropy().mean()
                        at = torch.as_tensor(aux_f[idx], device=device)
                        am = at >= 0
                        al = F.mse_loss(model.last_aux[am], at[am]) if am.any() else torch.zeros((), device=device)
                        ex = mb["expert"]
                        em = ex >= 0
                        use_bc = bc_w > 0 and em.any() and (pre or upd > args.value_warmup)
                        bl = F.cross_entropy(logits[em], ex[em]) if use_bc else torch.zeros((), device=device)
                        ka = torch.zeros((), device=device)
                        if anchor is not None:
                            with torch.no_grad():
                                la = F.log_softmax(anchor(mb)[0], -1)
                            lq = F.log_softmax(logits, -1)
                            ka = (la.exp() * (la - lq)).sum(-1).mean()
                        loss = (0 if warm else 1) * (pg - args.entropy * ent + args.kl_anchor * ka) + args.aux * al + bc_w * bl
                        (loss * share).backward()
                        with torch.no_grad():
                            kl_c += share * ((ratio - 1) - torch.log(ratio)).mean().item()
                        bcs.append(bl.item())
                        ents.append(ent.item())
                        anc.append(ka.item())
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    copt.step()
                    if not act_on:
                        continue   # the critic keeps learning on this rollout; the policy stops at the KL limit / waits for the warm-up
                    kls.append(kl_c)
                    if args.target_kl > 0 and not pre and kl_c > 1.5 * args.target_kl:
                        stop = True
                        continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
            dt = time.time() - t0
            tr = " ".join(f"L{m.replace('规程等级', '')}:{np.mean(recent[m]) if recent[m] else float('nan'):.2f}/{np.mean(rets[m]) if rets[m] else float('nan'):.1f}/{alloc.get(m, 0)}" for m in modes)

            print(f"upd {upd} {'BC ' if pre else ''}train {tr} bc {bc_w:.2f}/{np.mean(bcs) if bcs else 0:.2f} anc {np.mean(anc) if anc else 0:.3f} vl {np.mean(vls):.3f} ent {np.mean(ents) if ents else 0:.3f} kl {np.mean(kls) if kls else 0:.3f} {dt:.0f}s [env {t_env:.0f} / {n_fwd} passes (fight model {fighter.t_answer:.0f}) enc {t_enc:.0f} net {t_net:.0f} learn {time.time() - t_learn0:.0f}] {mem_str()}", flush=True)

            if refresher and upd % args.refresh_every == 0:
                try:
                    gap = refresher.refresh(n=args.refresh_n, steps=args.refresh_steps)
                except Exception as e:   # a failed correction must not end the run
                    gap = None
                    P(f"- upd {upd} 校正战斗模型失败，跳过：{type(e).__name__}: {e}")
                if gap:
                    P(f"- upd {upd} 校正战斗模型：{gap}")
                print(f"  after refresh: {mem_str()}", flush=True)
            if upd % args.eval_every == 0:
                fighter.record = False
                res = {m: evaluate(model, enc, m, device, args.eval_runs, args.sel_seed, ([f"--sp:bands={args.sp_bands}"] if args.sp_bands else []) + ([f"--sp:focus={args.sp_focus}"] if args.sp_focus else []) + ([f"--sp:second={args.sp_second}"] if args.sp_second else []) if sp else ["--no-flying"] + plan_flag, fighter) for m in modes}
                fighter.record = True
                mean_clear = float(np.mean([c for c, *_ in res.values()]))
                if DRAW is not None:
                    DRAW.save(os.path.join(run_dir, "draw.json"))
                    P(f"- upd {upd} 规程 16+ 各首领下各盟约的近期通关率（括号内为局数）：\n  - " + "\n  - ".join(DRAW.report("hi")))
                torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
                torch.save(critic.state_dict(), os.path.join(run_dir, "critic_last.pt"))
                save_state(upd)
                pl = "，".join(f"L{m.replace('规程等级', '')} {np.mean([x[0] for x in plans[m]]):.0%}/{np.mean([x[1] for x in plans[m]]):.0%}/{np.mean([x[2] for x in plans[m]]):.0f}层"
                              for m in modes if plans[m])
                sk = "，".join(f"L{m.replace('规程等级', '')} {np.mean([x[0] for x in stacks[m]]):.0f}/{np.mean([x[1] for x in stacks[m]]):.0f}/{np.mean([x[2] for x in stacks[m]]):.0f}/{np.mean([x[3] for x in stacks[m]]):.1f}"
                              for m in modes if stacks[m])
                if sp and mean_clear > best:
                    best = mean_clear
                    torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
                P(f"- upd {upd} 评估（{'真实模拟' if sp else '战斗模型'}）：" + "，".join(f"{m.replace('规程等级', 'L')} {c:.0%}" for m, (c, *_) in res.items())
                  + f"；平均 {mean_clear:.0%}；训练近况 {tr}；目标阵容（拿到/精锐/目标盟约层数）{pl}；"
                  + f"叠层（终局最高盟约层数 / 第5-8回合合计+层 / 第9-13回合合计+层 / 精锐数）{sk}；开放 {len(opened)}/{len(modes)} 档")
                print(f"  after eval: {mem_str()}", flush=True)
            if args.real_every and upd % args.real_every == 0:
                torch.save(model.state_dict(), os.path.join(run_dir, f"upd{upd}.pt"))
                res = {f"规程等级{x}": evaluate(model, enc, f"规程等级{x}", device, args.real_runs, args.sel_seed, real_extra)
                       for x in args.real_levels.split(",")}
                real_mean = float(np.mean([c for c, *_ in res.values()]))
                # The checkpoint is chosen by the real simulation (not the fight model); confirm on held-out seeds.
                if real_mean > best:
                    best = real_mean
                    torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
                P(f"- upd {upd} **真实模拟**：" + "，".join(f"{m.replace('规程等级', 'L')} {c:.0%}（{rd:.1f} 回合，最高盟约 {ly:.0f} 层）" for m, (c, rd, ly) in res.items())
                  + f"；平均 {real_mean:.0%}{'（新最好）' if real_mean >= best else ''}")
                print(f"  after real eval: {mem_str()}", flush=True)
            if upd % 10 == 0 and upd % args.eval_every != 0:
                # A restart goes on from here (the evaluations save too).
                torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
                torch.save(critic.state_dict(), os.path.join(run_dir, "critic_last.pt"))
                save_state(upd)
            fails = 0
        except RuntimeError as e:
            if "gym process ended" not in str(e):
                raise
            # A simulator process died: this update is lost, the gyms are rebuilt, the run goes on.
            fails += 1
            P(f"- upd {upd} 模拟器进程退出，跳过这次更新并重建环境（连续第 {fails} 次）")
            if fails > 10:
                save_state(upd - 1)
                raise
            store = flat = None
            r = new_gym()
            phi = phi_of(r)
            gym._ep = np.zeros(gym.envs, np.float32)
        except (MemoryError, torch.OutOfMemoryError) as e:
            # Out of committed memory (other programs grew): drop this update, free what can be freed, wait, and
            # go on with fresh gyms. The update's data is lost; nothing else is.
            fails += 1
            P(f"- upd {upd} 内存不足，跳过这次更新（连续第 {fails} 次）：{type(e).__name__}: {str(e)[:160]}；{mem_str()}")
            del e
            store = flat = mb = logits = None
            opt.zero_grad(set_to_none=True)
            copt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if fails > args.max_mem_fails:
                save_state(upd - 1)
                P("- 连续内存不足太多次，保存状态后退出（--resume 可续跑）")
                raise
            time.sleep(60)
            r = new_gym()
            phi = phi_of(r)
            gym._ep = np.zeros(gym.envs, np.float32)
    gym.close()


if __name__ == "__main__":
    main()
