"""
The learned fight model answering the gym's fights (src/ASPAssistant.Gym --surrogate): the ensemble in runs/fightnet
(train_fight.py) predicts each board, a little pessimistic (mean + k x the ensemble spread on leaks and boss overtime,
so the policy is not paid for boards the models disagree on), and a boss kill is drawn from the predicted chance.

It also rates a board against the clear round's leader (potential(): the shaping signal) and keeps the boards it was
asked about (recent fights and clear-round what-ifs) so online.py can fight a sample of them in the real simulation
and fine-tune the models on the policy's own boards.
"""
import collections
import json
import os

import numpy as np
import torch

from fight_model import ROOT, FightNet, Vocab, featurize, to_t

MODELS = os.environ.get("ASP_FIGHTNET") or os.path.join(ROOT, "python", "garrison_rl", "runs", "fightnet")


class Surrogate:
    def __init__(self, path=MODELS, device=None, k=0.5, seed=0, keep=40000):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.vocab = Vocab(path=os.path.join(path, "vocab.json"))
        self.models = []
        for f in sorted(os.listdir(path)):
            if f.startswith("fightnet_") and f.endswith(".pt"):
                m = FightNet(self.vocab).to(self.device)
                m.load_state_dict(torch.load(os.path.join(path, f), map_location=self.device))
                m.eval()
                self.models.append(m)
        if not self.models:
            raise RuntimeError(f"no fight models in {path}")
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.covs = {i: c for c, i in self.vocab.tables["cov"].items()}
        self.calls = 0
        # Boards seen lately: fights asked by the gym, and boards rated for the clear round (from round 9 on).
        self.fights = collections.deque(maxlen=keep)
        self.whatifs = collections.deque(maxlen=keep)
        self.record = True

    @torch.no_grad()
    def predict(self, boards, batch=2048):
        """Ensemble mean and spread for a list of boards (the gym's query dicts)."""
        rows = []
        for b in boards:
            r = dict(b)
            r["sim"] = {"Bosses": 1 if b.get("bossRound") else 0, "BossesKilled": 0, "Leaked": 0, "Alive": 0,
                        "bossProgress": 0.0, "seconds": 0.0, "Finished": False}
            r["handled"], r["worth"] = 0.0, 1.0
            rows.append(r)
        x, _ = featurize(rows, self.vocab)
        outs = []
        for m in self.models:
            parts = []
            for s in range(0, len(rows), batch):
                p = m(to_t(x, np.arange(s, min(len(rows), s + batch)), self.device))
                p["kill"] = torch.sigmoid(p["kill"])
                parts.append({k: v.float().cpu().numpy() for k, v in p.items()})
            outs.append({k: np.concatenate([q[k] for q in parts]) for k in parts[0]})
        stack = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
        return {k: v.mean(0) for k, v in stack.items()}, {k: v.std(0) for k, v in stack.items()}

    def _one(self, board, mean, std, j):
        leak = float(mean["leak"][j] + self.k * std["leak"][j])
        over = float(mean["over"][j] + self.k * std["over"][j])
        kill = bool(self.rng.random() < float(mean["kill"][j])) if board.get("bossRound") else False
        if kill:
            over = min(over, 60.0)   # a kill takes its own time: past the fixed time only a little
        layers = {self.covs[i]: round(float(g), 2) for i, g in enumerate(mean["layer"][j]) if i in self.covs and g >= 0.05}
        return {"chk": f"{board.get('round')}|{len(board.get('units') or [])}|{board.get('hp')}", "leak": max(0.0, leak), "kill": kill, "prog": float(mean["prog"][j]), "over": max(0.0, over),
                "hand": float(mean["hand"][j]), "layers": layers}

    def answer_batch(self, boards):
        import time
        t0 = time.time()
        out = self._answer_batch(boards)
        self.t_answer = getattr(self, "t_answer", 0.0) + time.time() - t0
        return out

    def _answer_batch(self, boards):
        self.calls += len(boards)
        if self.record:
            self.fights.extend(boards)
        mean, std = self.predict(boards)
        return [self._one(b, mean, std, j) for j, b in enumerate(boards)]

    def answer(self, board):
        return self.answer_batch([board])[0]

    def potential(self, boards):
        """How close each board is to beating the clear round's leader: (boss HP share taken + kill chance) / 2, from the
        ensemble mean less k x its spread (no credit for boards the models disagree on). An empty field rates 0."""
        idx = [j for j, b in enumerate(boards) if b and b.get("units")]
        out = np.zeros(len(boards), np.float32)
        if not idx:
            return out
        sub = [boards[j] for j in idx]
        if self.record:
            self.whatifs.extend(b for b in sub if b.get("now", 0) >= 9)
        mean, std = self.predict(sub)
        v = 0.5 * (mean["prog"] - self.k * std["prog"]) + 0.5 * (mean["kill"] - self.k * std["kill"])
        out[idx] = np.clip(v, 0, 1)
        return out


if __name__ == "__main__":
    s = Surrogate()
    board = json.loads(open(os.path.join(ROOT, "python", "garrison_rl", "runs", "lineup_try.jsonl"), encoding="utf-8").readline())
    board["bossRound"] = True
    print(s.answer(board), s.potential([board]))
