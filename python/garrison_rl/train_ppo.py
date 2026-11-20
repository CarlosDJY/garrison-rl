"""
PPO for the whole 卫戍协议 run (drafting, leveling, offers, equipment) against src/ASPAssistant.Gym.

  python train_ppo.py --modes 规程等级1,规程等级4,规程等级8,规程等级11,规程等级14 --promote 0.7

Each mode trains until the greedy policy clears --promote of the 48 held-out runs (seeds 9000-9047, the same ones the
linear policy was scored on), then the next mode starts from the same weights. Logs to reports/playground/ppo.md,
checkpoints to python/garrison_rl/runs/.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from gym_client import GymClient, ROOT
from model import Encoder, Policy, to_torch

A_MAX = 200


def pad_actions(batch):
    A = batch["kind"].shape[1]
    if A >= A_MAX:
        for k in ("kind", "ptr", "amask"):
            batch[k] = batch[k][:, :A_MAX]
        return batch
    B = batch["kind"].shape[0]
    batch["kind"] = np.concatenate([batch["kind"], np.zeros((B, A_MAX - A), np.int64)], 1)
    batch["ptr"] = np.concatenate([batch["ptr"], np.zeros((B, A_MAX - A, 2), np.int64)], 1)
    batch["amask"] = np.concatenate([batch["amask"], np.zeros((B, A_MAX - A), bool)], 1)
    return batch


def evaluate(model, enc, mode, device, runs=48, seed=9000, extra=(), fighter=None):
    gym = GymClient(mode=mode, envs=runs, extra=extra, fighter=fighter)
    r = gym.reset(seed)
    results = [None] * runs
    model.eval()
    with torch.no_grad():
        while any(x is None for x in results):
            t = to_torch(pad_actions(enc.encode(r["obs"])), device)
            logits, _ = model(t)
            r = gym.step(logits.argmax(-1).cpu().numpy())
            for i, info in enumerate(r["info"]):
                if info and results[i] is None:
                    results[i] = info
    gym.close()
    model.train()
    cleared = np.mean([x["cleared"] for x in results])
    rounds = np.mean([x["rounds"] for x in results])
    hidden = [x["hiddenCore"] for x in results if x.get("hiddenCore") is not None]
    evaluate.hidden = (np.mean(hidden) * len(hidden) / len(results)) if hidden else None   # share of all runs
    return cleared, rounds, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="规程等级1")
    ap.add_argument("--promote", type=float, default=0.7)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--updates", type=int, default=400, help="most updates per mode")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99, help="discount per round (shop actions within a round are not discounted)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--entropy", type=float, default=0.03)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="ppo")
    ap.add_argument("--extra-start", type=int, default=0,
                    help="widened-shop curriculum: extra shop slots the first mode starts with (e.g. 20); each later mode starts "
                         "--extra-drop fewer; within a mode the slots shrink 100%%/60%%/40%%/20%%/0 of its start, each step kept "
                         "until the evaluation clears --promote with it; the last step is always the real shop")
    ap.add_argument("--extra-drop", type=int, default=4, help="how many fewer extra slots each harder mode starts with")
    ap.add_argument("--planner", default="quick", choices=["quick", "mid", "full"],
                    help="placement search: quick (600 steps, no simulation), mid (2000 x2, 200 simulated), full (the app's 5000 x5, 700 simulated)")
    ap.add_argument("--no-flying", action="store_true")
    ap.add_argument("--shape-layers", type=float, default=0.0, help="potential-based shaping weight of layer strength")
    ap.add_argument("--shape-elites", type=float, default=0.0, help="potential-based shaping weight per elite (by tier)")
    ap.add_argument("--shape-dps", type=float, default=3.0, help="potential-based shaping weight of field DPS ÷ the leader's need")
    ap.add_argument("--dps-cap", type=float, default=1.2, help="the DPS ratio stops counting past this (no glass cannons for points)")
    ap.add_argument("--aux", type=float, default=0.5, help="weight of the next-fight HP-loss prediction")
    # Reward per fight (HP lost always costs its share of the starting HP).
    ap.add_argument("--r-survive", type=float, default=0.3, help="for living through a round")
    ap.add_argument("--r-hp", type=float, default=0.3, help="x the HP lost as a share of the start (HP matters mostly as the boss-round budget)")
    ap.add_argument("--r-clear", type=float, default=25.0, help="for clearing the run")
    ap.add_argument("--r-death", type=float, default=5.0, help="taken away when the run is lost")
    ap.add_argument("--r-boss", type=float, default=2.0, help="x the share of the bosses' HP taken on a boss round")
    ap.add_argument("--r-kill", type=float, default=2.0, help="for killing every boss of the round")
    ap.add_argument("--r-fast", type=float, default=3.0, help="x how much of the fixed 115 s was left when the bosses fell")
    ap.add_argument("--r-hidden", type=float, default=10.0, help="bonus for beating the 隐秘核心 (round 15, 规程等级5+)")
    ap.add_argument("--promote-hidden", type=float, default=0.3,
                    help="also needed to move up where the 隐秘核心 exists: share of runs that beat it (it is about a stronger level's normal leader)")
    ap.add_argument("--surrogate", action="store_true", help="fights answered by the learned fight model (runs/fightnet), no simulation")
    ap.add_argument("--surrogate-k", type=float, default=0.5, help="pessimism: leaks and overtime at mean + k x the ensemble spread")
    ap.add_argument("--real-eval-every", type=int, default=0, help="with --surrogate: also evaluate in the real simulation every N updates")
    ap.add_argument("--target-kl", type=float, default=0.0, help="stop an update's epochs once the approximate KL passes this (0: off)")
    ap.add_argument("--value-warmup", type=int, default=0, help="first N updates train only the value (a fresh value head on a loaded policy)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra = {"quick": [], "mid": ["--iterations", "2000", "--restarts", "2", "--sim", "200"],
             "full": ["--iterations", "5000", "--restarts", "5", "--sim", "700"]}[args.planner]
    if args.no_flying:
        extra = extra + ["--no-flying"]

    def reward_of(parts):
        if parts is None:
            return 0.0
        survived, lost, cleared, died, boss, killed, seconds, hidden, hidden_won = parts[:9]
        r = args.r_survive * survived - args.r_hp * lost + args.r_clear * cleared - args.r_death * died
        if boss >= 0:
            r += args.r_boss * boss + killed * (args.r_kill + args.r_fast * max(0.0, 1 - seconds))
        if hidden:
            r += args.r_hidden * hidden_won
        return r

    def phi_of(r):
        return np.array([args.shape_layers * p[0] + args.shape_elites * p[1] + args.shape_dps * min(args.dps_cap, p[2])
                         for p in r["phi"]], np.float32)
    modes = args.modes.split(",")
    run_dir = os.path.join(ROOT, "python", "garrison_rl", "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(ROOT, "reports", "playground", f"{args.tag}.md")

    fighter = None
    if args.surrogate:
        from surrogate import Surrogate
        fighter = Surrogate(device=device, k=args.surrogate_k)
    gym = GymClient(mode=modes[0], envs=args.envs, extra=extra, fighter=fighter)
    enc = Encoder(gym.spec)
    model = Policy(enc).to(device)
    if args.init:
        own = model.state_dict()
        loaded = {k: v for k, v in torch.load(args.init, map_location=device).items() if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(loaded, strict=False)
        print("init: new layers", missing, "dropped", unexpected, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n# PPO（实体 Transformer，{n_params / 1e6:.2f}M 参数，{args.envs} 个环境 × {args.steps} 步/更新，起点 {args.init or '随机'}）\n\n")
        f.write("| 难度 | 更新 | 局数 | 训练通关率（近 200 局） | 平均回合 | 评估通关（48 局） | 评估平均回合 | 每更新耗时 |\n|---|---|---|---|---|---|---|---|\n")

    # Stages: every mode from its widened shop down to the real one (e.g. 规程等级4: 20,12,8,4,0; 规程等级6: 16,10,6,3,0).
    stages = []
    for k, m in enumerate(modes):
        start = max(0, args.extra_start - k * args.extra_drop)
        steps = sorted({round(start * f) for f in (1.0, 0.6, 0.4, 0.2)} - {0}, reverse=True) + [0]
        stages += [(m, x) for x in steps]
    print("stages:", ", ".join(f"{m}+{x}" for m, x in stages), flush=True)
    base_extra = extra
    for mi, (mode, xs) in enumerate(stages):
        extra = base_extra + (["--extra-slots", str(xs)] if xs else [])
        label = mode if xs == 0 else f"{mode}+{xs}格"
        if mi > 0 or xs:
            gym.close()
            gym = GymClient(mode=mode, envs=args.envs, extra=extra, fighter=fighter)
        r = gym.reset(1000 * (mi + 1))
        phi = phi_of(r)
        recent = []
        episodes = 0
        best_eval = -1
        for upd in range(1, args.updates + 1):
            t0 = time.time()
            store = {k: [] for k in ("batch", "act", "logp", "val", "rew", "done", "fight", "bound")}
            for _ in range(args.steps):
                batch = pad_actions(enc.encode(r["obs"]))
                with torch.no_grad():
                    logits, v = model(to_torch(batch, device))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                r = gym.step(a.cpu().numpy())
                done_now = np.array(r["done"], np.float32)
                # Time passes only when a round is fought: shop actions are not discounted (else every buy or refresh
                # that leaves the potential unchanged costs (1 - gamma) * phi, a tax on spending).
                bound = np.maximum(np.array([p is not None for p in r["parts"]], np.float32), done_now)
                g_t = np.where(bound > 0, args.gamma, 1.0).astype(np.float32)
                # Potential-based shaping: + gamma_t * phi(next) - phi(now); a finished run's next potential is 0.
                phi_next = phi_of(r)
                base = np.array([reward_of(p) for p in r["parts"]], np.float32)
                shaped = base + g_t * phi_next * (1 - done_now) - phi
                phi = phi_next
                store["bound"].append(bound)
                store["fight"].append(np.array(r["fight"], np.float32))
                store["batch"].append(batch)
                store["act"].append(a.cpu().numpy())
                store["logp"].append(logp.cpu().numpy())
                store["val"].append(v.cpu().numpy())
                store["rew"].append(shaped)
                store["done"].append(np.array(r["done"], np.float32))
                for info in r["info"]:
                    if info:
                        recent.append(info)
                        episodes += 1
            recent = recent[-200:]
            with torch.no_grad():
                _, last_v = model(to_torch(pad_actions(enc.encode(r["obs"])), device))
            last_v = last_v.cpu().numpy()
            # GAE.
            T, N = args.steps, args.envs
            vals = np.stack(store["val"]); rews = np.stack(store["rew"]); dones = np.stack(store["done"])
            adv = np.zeros((T, N), np.float32)
            gae = np.zeros(N, np.float32)
            bounds = np.stack(store["bound"])
            for t in reversed(range(T)):
                nxt = last_v if t == T - 1 else vals[t + 1]
                nonterm = 1 - dones[t]
                # Discount and GAE decay only across rounds; within the shop phase credit flows undiminished.
                g_t = np.where(bounds[t] > 0, args.gamma, 1.0)
                l_t = np.where(bounds[t] > 0, args.lam, 1.0)
                delta = rews[t] + g_t * nxt * nonterm - vals[t]
                gae = delta + g_t * l_t * nonterm * gae
                adv[t] = gae
            ret = adv + vals
            # The next fight's HP loss from each step (within this rollout; -1 when none came).
            fights = np.stack(store["fight"])
            aux_t = np.full((T, N), -1, np.float32)
            nxt_f = np.full(N, -1, np.float32)
            for t in reversed(range(T)):
                nxt_f = np.where(fights[t] >= 0, fights[t], np.where(dones[t] > 0, -1, nxt_f))
                aux_t[t] = nxt_f
            aux_f = aux_t.reshape(-1)
            flat = {k: np.concatenate([b[k] for b in store["batch"]], 0) for k in store["batch"][0]}
            acts = np.concatenate(store["act"]); old_logp = np.concatenate(store["logp"])
            adv_f = adv.reshape(-1); ret_f = ret.reshape(-1)
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
            M = T * N
            stop = False
            for _ in range(args.epochs):
                if stop:
                    break
                perm = np.random.permutation(M)
                for s in range(0, M, args.minibatch):
                    idx = perm[s:s + args.minibatch]
                    mb = to_torch({k: v[idx] for k, v in flat.items()}, device)
                    logits, v = model(mb)
                    dist = torch.distributions.Categorical(logits=logits)
                    a_t = torch.as_tensor(acts[idx], device=device)
                    lp = dist.log_prob(a_t)
                    ratio = torch.exp(lp - torch.as_tensor(old_logp[idx], device=device))
                    A_t = torch.as_tensor(adv_f[idx], device=device)
                    pg = -torch.min(ratio * A_t, ratio.clamp(1 - args.clip, 1 + args.clip) * A_t).mean()
                    vl = F.mse_loss(v, torch.as_tensor(ret_f[idx], device=device))
                    ent = dist.entropy().mean()
                    at = torch.as_tensor(aux_f[idx], device=device)
                    am = at >= 0
                    al = F.mse_loss(model.last_aux[am], at[am]) if am.any() else torch.zeros((), device=device)
                    warm = upd <= args.value_warmup
                    loss = (0 if warm else 1) * (pg - args.entropy * ent) + 0.5 * vl + args.aux * al
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    if args.target_kl > 0 and not warm and kl > 1.5 * args.target_kl:
                        stop = True
                        break
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
            dt = time.time() - t0
            train_clear = np.mean([x["cleared"] for x in recent]) if recent else 0
            train_rounds = np.mean([x["rounds"] for x in recent]) if recent else 0
            ev_clear = ev_rounds = None
            if upd % args.eval_every == 0:
                ev_clear, ev_rounds, _ = evaluate(model, enc, mode, device, extra=extra, fighter=fighter)
                torch.save(model.state_dict(), os.path.join(run_dir, f"last_{label}.pt"))
                if ev_clear > best_eval:
                    best_eval = ev_clear
                    torch.save(model.state_dict(), os.path.join(run_dir, f"best_{label}.pt"))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"| {label} | {upd} | {episodes} | {train_clear:.0%} | {train_rounds:.1f} | "
                        f"{'' if ev_clear is None else f'{ev_clear:.0%}'} | {'' if ev_rounds is None else f'{ev_rounds:.1f}'} | {dt:.0f}s |\n")
            mem = ""
            if ev_clear is not None:
                try:
                    import psutil
                    mem = f" gymMB {psutil.Process(gym.proc.pid).memory_info().rss / 2**20:.0f} pyMB {psutil.Process().memory_info().rss / 2**20:.0f}"
                except Exception:
                    pass
            lv = np.mean([x["level"] for x in recent]) if recent else 0
            el = np.mean([x["elites"] for x in recent]) if recent else 0
            dr = np.mean([x.get("dpsRatio", 0) for x in recent]) if recent else 0
            hc = [x["hiddenCore"] for x in recent if x.get("hiddenCore") is not None]
            hc_s = f" hidden {np.mean(hc):.2f}" if hc else ""
            print(f"{label} upd {upd} eps {episodes} train {train_clear:.2f}{hc_s} rounds {train_rounds:.1f} level {lv:.1f} elites {el:.1f} dps {dr:.2f}"
                  + ("" if ev_clear is None else f" eval {ev_clear:.2f}")
                  + ("" if ev_clear is None or evaluate.__dict__.get("hidden") is None else f" evalHidden {evaluate.hidden:.2f}")
                  + f" {dt:.0f}s" + mem, flush=True)
            if fighter is not None and args.real_eval_every and upd % args.real_eval_every == 0:
                rc, rr, _ = evaluate(model, enc, mode, device, extra=extra + ["--no-hidden-core"])
                print(f"{label} upd {upd} REAL eval {rc:.2f} rounds {rr:.1f}", flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"| {label} | {upd} | 真实模拟评估 | | | {rc:.0%} | {rr:.1f} | |\n")
            hidden_ok = evaluate.__dict__.get("hidden") is None or evaluate.hidden >= args.promote_hidden
            if ev_clear is not None and ev_clear >= args.promote and hidden_ok:
                break
    gym.close()


if __name__ == "__main__":
    main()
