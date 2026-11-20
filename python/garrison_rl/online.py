"""
Keeps the fight model honest while the policy trains on it (Dyna / DAgger style): a sample of the boards the policy
actually brings (fights the surrogate answered, and clear-round what-ifs it rated) is fought in the real simulation
(the gym's --relabel, the placement search the fight data was made with), the models' error on them is logged (how much
the policy is exploiting the model), and every ensemble member is fine-tuned on all the online fights so far mixed
with a replay of the original data.
"""
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from fight_model import DATA, ROOT, Vocab, featurize, load_rows, losses, to_t
from gym_client import GYM_EXE, MID_PLANNER

ONLINE = os.path.join(ROOT, "python", "garrison_rl", "data", "fights_online")


def _job(b, seed, src):
    keys = ("mode", "round", "hp", "shopLevel", "fieldBonus", "map", "boss", "types", "units", "layers")
    return dict({k: b[k] for k in keys}, seed=seed, src=b.get("_src") or src)


def _lineup_files():
    """Boards toward the saved lineups (gen_lineup_boards.py), the training part."""
    return sorted(glob.glob(os.path.join(ROOT, "python", "garrison_rl", "data", "fights_lineup", "lineup_train_*.jsonl")))


def replay_rows(n=60000, seed=0, max_round=14):
    """The original fight data as train_fight.py --boss-v2 --max-round 14 uses it, a random n of it."""
    rows = [r for r in load_rows() if r["sim"]["Bosses"] == 0]
    rows += load_rows(sorted(glob.glob(os.path.join(ROOT, "python", "garrison_rl", "data", "fights_v2", "boss_*.jsonl"))))
    rows = [r for r in rows if r["round"] <= max_round]
    random.Random(seed).shuffle(rows)
    # Up to half of the replay is the lineup boards (what the policy's own boards never show the models), the rest the
    # original data.
    lineup = [r for r in load_rows(_lineup_files()) if r["round"] <= max_round]
    random.Random(seed + 1).shuffle(lineup)
    lineup = lineup[:n // 2]
    return lineup + rows[:n - len(lineup)]


def _replay_files():
    return sorted(glob.glob(os.path.join(DATA, "*.jsonl"))) + sorted(glob.glob(os.path.join(ROOT, "python", "garrison_rl", "data", "fights_v2", "boss_*.jsonl"))) + _lineup_files()


def replay_arrays(n, vocab_path, seed=0, max_round=14):
    """replay_rows(n) featurized, cached in an .npz. It is built in a child process: reading every original fight as
    Python objects leaves the parent's heap ~1 GB larger for good, and on this machine memory is the tight resource."""
    h = hashlib.sha1(open(vocab_path, "rb").read())
    for f in _replay_files():
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(f"{n}|{seed}|{max_round}".encode())
    path = os.path.join(ONLINE, f"_replay_{h.hexdigest()[:12]}.npz")
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.abspath(__file__), "--build-replay", str(n), vocab_path, path, str(seed), str(max_round)], check=True)
    with np.load(path) as z:
        x = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        y = {k[2:]: z[k] for k in z.files if k.startswith("y_")}
    return x, y


def _cat(a, b, keep):
    return b if a is None else {k: np.concatenate([a[k], b[k]])[-keep:] for k in a}


class Refresher:
    def __init__(self, fighter, tag, threads=18, replay=60000, lr=1e-4, log=print, keep_online=30000):
        self.f = fighter
        self.tag = tag
        self.threads = threads
        self.lr = lr
        self.log = log
        self.k = 0
        # The online fights so far, featurized (the newest keep_online of them).
        self.ox = self.oy = None
        self.keep_online = keep_online
        self.n_online = 0
        os.makedirs(ONLINE, exist_ok=True)
        t0 = time.time()
        self.rx, self.ry = replay_arrays(replay, os.path.join(fighter.path, "vocab.json"))
        self.n_replay = len(self.ry["leak"])
        log(f"online: replay of {self.n_replay} fights loaded in {time.time() - t0:.0f}s")
        self.opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01) for m in fighter.models]
        self.out_dir = os.path.join(ROOT, "python", "garrison_rl", "runs", tag, "fightnet")
        os.makedirs(self.out_dir, exist_ok=True)

    def _fight(self, boards):
        self.k += 1
        rng = random.Random(1000 + self.k)
        src = f"online_{self.tag}_{self.k}"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        inp = os.path.join(ONLINE, f"_in_{self.tag}_{stamp}.jsonl")
        out = os.path.join(ONLINE, f"{self.tag}_{stamp}.jsonl")
        with open(inp, "w", encoding="utf-8") as fh:
            for b in boards:
                fh.write(json.dumps(_job(b, rng.randrange(1_000_000), src), ensure_ascii=False) + "\n")
        cmd = [GYM_EXE, "--data", os.path.join(ROOT, "data"), "--envs", str(self.threads), "--relabel", inp, "--out", out] + MID_PLANNER + ["--no-flying"]
        err = out.replace(".jsonl", ".err.log")
        for attempt in range(2):
            if os.path.exists(out):
                os.remove(out)
            with open(err, "a", encoding="utf-8") as fe:
                code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=fe).returncode
            if code == 0:
                break
            self.log(f"online: relabel exited {code} (attempt {attempt + 1}), see {err}")
        os.remove(inp)
        # A failed run still leaves the boards it finished; use those rather than nothing.
        return load_rows([out]) if os.path.exists(out) and os.path.getsize(out) > 0 else []

    @torch.no_grad()
    def _gap(self, rows):
        """The models (as the policy sees them) against the real fights of the policy's own boards."""
        x, y = featurize(rows, self.f.vocab)
        idx = np.arange(len(rows))
        preds = []
        for m in self.f.models:
            p = m(to_t(x, idx, self.f.device))
            p["kill"] = torch.sigmoid(p["kill"])
            preds.append({k: v.float().cpu().numpy() for k, v in p.items()})
        mean = {k: np.mean([p[k] for p in preds], 0) for k in preds[0]}
        boss = y["boss"] > 0
        out = {}
        if (~boss).any():
            out["leak_pred"], out["leak_real"] = float(mean["leak"][~boss].mean()), float(y["leak"][~boss].mean())
        if boss.any():
            out["kill_pred"], out["kill_real"] = float(mean["kill"][boss].mean()), float(y["kill"][boss].mean())
            out["prog_pred"], out["prog_real"] = float(mean["prog"][boss].mean()), float(y["prog"][boss].mean())
        # Layers gained in the round, every covenant together.
        out["layer_pred"], out["layer_real"] = float(mean["layer"].sum(1).mean()), float(y["layer"].sum(1).mean())
        out["n_boss"], out["n_normal"] = int(boss.sum()), int((~boss).sum())
        return out

    def _aspire(self, boards, rng, n):
        """n of the policy's boards moved toward a saved lineup (gen_lineup_boards.aspire_board), marked by their src."""
        import gen_lineup_boards as glb
        if getattr(self, "_tables", None) is None:
            self._tables = glb.Tables()
            self._lineups = self._tables.lineups()
        if not self._lineups or not boards or n <= 0:
            return []
        out = []
        for b in rng.sample(boards, min(len(boards), n)):
            if b.get("round", 0) < 4 or not b.get("units"):
                continue
            a = glb.aspire_board(self._tables, rng, self._lineups, b)
            a["_src"] = f"aspire_{self.tag}"
            out.append(a)
        return out

    def refresh(self, n=1500, steps=300, batch=256, whatif_share=0.4, aspire_share=0.25):
        f = self.f
        t0 = time.time()
        rng = random.Random(self.k)
        fights, whatifs = list(f.fights), list(f.whatifs)
        # A share of the sample is not the policy's own boards but those boards moved toward a saved lineup: the models
        # are corrected where the policy should go as well as where it is.
        na = int(n * aspire_share)
        n -= na
        nw = min(len(whatifs), int(n * whatif_share))
        boards = rng.sample(fights, min(len(fights), n - nw)) + rng.sample(whatifs, nw)
        if not boards:
            return None
        boards += self._aspire(fights, rng, na)
        rows = self._fight(boards)
        if len(rows) < 50:
            self.log(f"online: only {len(rows)} boards came back; refresh skipped")
            return None
        t_sim = time.time() - t0
        # The gap that tells how far the policy exploits the models is on its own boards; the lineup-ward ones apart.
        own = [r for r in rows if not str(r.get("src", "")).startswith("aspire_")]
        asp = [r for r in rows if str(r.get("src", "")).startswith("aspire_")]
        gap = self._gap(own or rows)
        if asp:
            gap.update({"aspire_" + k: v for k, v in self._gap(asp).items()})
        nx, ny = featurize(rows, f.vocab)
        del rows
        self.ox = _cat(self.ox, nx, self.keep_online)
        self.oy = _cat(self.oy, ny, self.keep_online)
        self.n_online += len(ny["leak"])
        ox, oy = self.ox, self.oy
        oyt = {k: torch.as_tensor(v, device=f.device) for k, v in oy.items()}
        ryt = {k: torch.as_tensor(v, device=f.device) for k, v in self.ry.items()}
        nr, no = self.n_replay, len(oy["leak"])
        for m, opt in zip(f.models, self.opts):
            m.train()
            for _ in range(steps):
                bo = np.random.randint(0, no, batch)
                br = np.random.randint(0, nr, batch)
                lo, _ = losses(m(to_t(ox, bo, f.device)), {k: v[torch.as_tensor(bo, device=f.device)] for k, v in oyt.items()})
                lr_, _ = losses(m(to_t(self.rx, br, f.device)), {k: v[torch.as_tensor(br, device=f.device)] for k, v in ryt.items()})
                loss = lo + lr_
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
        for j, m in enumerate(f.models):
            torch.save(m.state_dict(), os.path.join(self.out_dir, f"fightnet_{j}.pt"))
        f.vocab.save(os.path.join(self.out_dir, "vocab.json"))
        gap.update(n=len(ny["leak"]), online=self.n_online, sim_s=round(t_sim), total_s=round(time.time() - t0))
        return gap


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--build-replay":
    _, _, n, vocab_path, out, seed, max_round = sys.argv
    x, y = featurize(replay_rows(int(n), int(seed), int(max_round)), Vocab(path=vocab_path))
    tmp = out + ".tmp.npz"
    np.savez(tmp, **{"x_" + k: v for k, v in x.items()}, **{"y_" + k: v for k, v in y.items()})
    os.replace(tmp, out)
