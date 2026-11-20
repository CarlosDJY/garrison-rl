"""Client for src/ASPAssistant.Gym: a batch of whole-run environments in a .NET process, one JSON line per message."""
import concurrent.futures
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# ASP_GYM_EXE: another build (e.g. while a training run holds the default one).
GYM_EXE = os.environ.get("ASP_GYM_EXE") or os.path.join(ROOT, "src", "ASPAssistant.Gym", "bin", "Release", "net8.0-windows", "ASPAssistant.Gym.exe")
DATA = os.path.join(ROOT, "data")
# The placement search the fight data was made with (gen_fights.py): real-simulation checks use it too.
MID_PLANNER = ["--iterations", "2000", "--restarts", "2", "--sim", "200"]


def level_of(mode):
    try:
        return int(mode.replace("规程等级", ""))
    except ValueError:
        return 0


class NetGymClient:
    def __init__(self, mode="规程等级1", envs=64, exe=GYM_EXE, data=DATA, extra=(), fighter=None):
        """fighter: a surrogate.Surrogate answering the fights (the gym runs with --surrogate)."""
        self.fighter = fighter
        self.mode = mode
        self.level = level_of(mode)
        args = [exe, "--data", data, "--mode", mode, "--envs", str(envs), *extra]
        if fighter is not None:
            args += ["--surrogate", "--no-hidden-core"]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=1 << 20)
        self.spec = self._read()["spec"]
        self.envs = envs

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("gym process ended")
            msg = json.loads(line.decode("utf-8"))
            if "fightQueries" in msg:
                # Fights for the surrogate (a batch): answer them and keep reading.
                ans = self.fighter.answer_batch(msg["fightQueries"])
                self.proc.stdin.write((json.dumps(ans, ensure_ascii=False) + "\n").encode("utf-8"))
                self.proc.stdin.flush()
                continue
            for o in msg.get("obs") or ():
                o["level"] = self.level
            return msg

    def _write(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _send(self, msg):
        self._write(msg)
        return self._read()

    def reset(self, seed=0):
        return self._send({"cmd": "reset", "seed": seed})

    def step(self, actions):
        return self._send({"cmd": "step", "actions": [int(a) for a in actions]})

    def close(self):
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass
        self.proc.kill()


# The other simulator: Stronghold-Protocol's (the fan remake's engine; D:/Code/Games/Arknights/sp-gym/spgym.mjs speaks this
# protocol). ASP_SIM=sp picks it; its fights are always the real ones (no fight model in the loop).
SP_NODE = os.environ.get("ASP_SP_NODE") or r"D:/Code/Games/Arknights/_tools/node-v24.22.3-win-x64/node.exe"
SP_GYM = os.environ.get("ASP_SP_GYM") or r"D:/Code/Games/Arknights/sp-gym/spgym.mjs"
SP_VERSION = "act3autochess@V3_1"


_PICKER = None


def sp_picker():
    """The layout picker (sp_layout.LayoutPicker) of ASP_SP_PICKER (a run directory), loaded once."""
    global _PICKER
    if _PICKER is None:
        from sp_layout import LayoutPicker
        _PICKER = LayoutPicker(os.environ["ASP_SP_PICKER"], margin=float(os.environ.get("ASP_SP_PICKER_MARGIN", "0")))
    return _PICKER


class _SPProc:
    def __init__(self, mode, envs, extra=()):
        args = [SP_NODE, SP_GYM, "--version", SP_VERSION, "--mode", mode, "--envs", str(envs), *extra]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=1 << 20,
                                     cwd=os.path.dirname(SP_GYM))
        self.envs = envs
        self.spec = self.read()["spec"]

    def read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("sp gym process ended")
            msg = json.loads(line.decode("utf-8"))
            if "layoutQueries" in msg:
                # Candidate layouts of the preps that just ended (gym --picker): the layout model names one each.
                self.write({"picks": sp_picker().answer(msg["layoutQueries"])})
                continue
            return msg

    def write(self, msg):
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + chr(10)).encode("utf-8"))
        self.proc.stdin.flush()


class SPGymClient:
    """One difficulty on Stronghold-Protocol's simulator, its envs spread over several node processes (one core each)."""

    KEYS = ("obs", "reward", "done", "info", "phi", "fight", "parts")

    def __init__(self, mode="规程等级8", envs=64, extra=(), per_proc=None):
        per_proc = per_proc or int(os.environ.get("ASP_SP_PER_PROC", "8"))
        self.mode, self.level, self.envs, self.fighter = mode, level_of(mode), envs, None
        sizes = [per_proc] * (envs // per_proc) + ([envs % per_proc] if envs % per_proc else [])
        # Only the options meant for this simulator: "--sp:rehearsal=0" -> --rehearsal 0.
        own = [a for x in extra if x.startswith("--sp:") for a in ("--" + x[5:]).split("=", 1)]

        def own_of(j):
            # --fights names a prefix: each process appends to its own file (prefix_L<level>_<j>.jsonl).
            out = list(own)
            if "--fights" in out:
                k = out.index("--fights") + 1
                out[k] = f"{out[k]}_L{self.level}_{j}.jsonl"
            return out

        self.procs = [_SPProc(mode, n, own_of(j)) for j, n in enumerate(sizes)]
        self.spec = self.procs[0].spec

    def _write(self, msg):
        if msg.get("cmd") == "step":
            k = 0
            for p in self.procs:
                p.write({"cmd": "step", "actions": msg["actions"][k:k + p.envs]})
                k += p.envs
        elif msg.get("cmd") == "reset":
            for j, p in enumerate(self.procs):
                p.write({"cmd": "reset", "seed": int(msg.get("seed", 0)) + 5000 * j + 1})
        else:
            for p in self.procs:
                p.write(msg)

    def _read(self):
        msgs = [p.read() for p in self.procs]
        if "obs" not in msgs[0]:
            return msgs[0]
        out = {k: [x for m in msgs for x in m[k]] for k in self.KEYS if k in msgs[0]}
        for o in out["obs"]:
            o["level"] = self.level
        return out

    def _send(self, msg):
        self._write(msg)
        return self._read()

    def reset(self, seed=0):
        return self._send({"cmd": "reset", "seed": seed})

    def step(self, actions):
        return self._send({"cmd": "step", "actions": [int(a) for a in actions]})

    def close(self):
        for p in self.procs:
            try:
                p.write({"cmd": "close"})
            except Exception:
                pass
        for p in self.procs:
            try:
                p.proc.wait(timeout=5)
            except Exception:
                p.proc.kill()


def GymClient(mode="规程等级1", envs=64, exe=GYM_EXE, data=DATA, extra=(), fighter=None):
    if os.environ.get("ASP_SIM") == "sp":
        return SPGymClient(mode=mode, envs=envs, extra=extra)
    return NetGymClient(mode=mode, envs=envs, exe=exe, data=data, extra=extra, fighter=fighter)


class MultiGym:
    """Several gyms (one per difficulty) stepped as one batch: alloc = {mode: envs}. Each env's mode is in .modes."""

    KEYS = ("obs", "reward", "done", "info", "phi", "fight", "parts")

    def __init__(self, alloc, extra=(), fighter=None):
        self.clients = [GymClient(mode=m, envs=n, extra=extra, fighter=fighter) for m, n in alloc.items() if n > 0]
        self.spec = self.clients[0].spec
        self.modes = [c.mode for c in self.clients for _ in range(c.envs)]
        self.envs = len(self.modes)

    def _merge(self, msgs):
        return {k: [x for m in msgs for x in m[k]] for k in self.KEYS if k in msgs[0]}

    def reset(self, seed=0):
        return self._merge([c.reset(seed + 100_000 * k) for k, c in enumerate(self.clients)])

    def step(self, actions):
        k = 0
        for c in self.clients:
            c._write({"cmd": "step", "actions": [int(a) for a in actions[k:k + c.envs]]})
            k += c.envs
        return self._merge([c._read() for c in self.clients])

    def groups(self):
        """The simulator processes as (first env, one past the last env), or None when the gyms cannot be stepped apart.
        A group is stepped on its own with step_group: a process that is simulating a fight then holds up only its own envs."""
        if getattr(self, "_groups", None) is None:
            out, k = [], 0
            for c in self.clients:
                if not hasattr(c, "procs"):
                    return None
                for p in c.procs:
                    out.append((c, p, k, k + p.envs))
                    k += p.envs
            self._groups = out
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(out))
        return [(lo, hi) for _, _, lo, hi in self._groups]

    def step_group(self, gi, actions):
        """Steps one group; a future of its reply (the keys of step(), for that group's envs)."""
        c, p, _, _ = self._groups[gi]

        def run():
            p.write({"cmd": "step", "actions": [int(a) for a in actions]})
            msg = p.read()
            for o in msg["obs"]:
                o["level"] = c.level
            return msg

        return self._pool.submit(run)

    def close(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        for c in self.clients:
            c.close()


if __name__ == "__main__":
    import random
    import time

    g = GymClient(envs=8)
    print(len(g.spec["operators"]), "operators,", len(g.spec["covenants"]), "covenants,", len(g.spec["items"]), "items")
    r = g.reset(1)
    t0, steps, ends = time.time(), 0, []
    while len(ends) < 16:
        acts = [random.randrange(len(o["acts"])) for o in r["obs"]]
        r = g.step(acts)
        steps += 1
        ends += [i for i in r["info"] if i]
    print(f"{steps} steps, {len(ends)} runs in {time.time() - t0:.1f}s; mean rounds {sum(e['rounds'] for e in ends) / len(ends):.1f}")
    g.close()
