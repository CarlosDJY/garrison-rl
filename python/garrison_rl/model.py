"""
Entity-transformer policy for the 卫戍协议 run.

Tokens (fixed slots, padding masked): 1 global, 20 held units, 6 shop slots, 3 晋升奖励 slots, 5 hand items,
1 item-shop slot, one per covenant. Operators carry a learned id embedding plus their static features (tier,
covenants, profession, layer potential); items and covenants have id embeddings. A transformer encoder mixes them;
each legal action is scored from its kind and the encoded token(s) it points at (buy -> that shop slot, sell -> that
unit, equip -> that hand item and that unit, ...) together with the global token. A value head reads the global token.
"""
import numpy as np
import torch
import torch.nn as nn

# Shop slots: up to 26 (the widened-shop curriculum adds up to 20 to the real 5); unused ones are masked.
MAX_UNITS, SHOP, OFFER, HAND, ITEMSHOP = 20, 26, 3, 5, 1
G_FEATS, U_FEATS, COV_FEATS = 17, 6, 5   # g: 16 + the shop frozen flag
PLAN_G = 4   # locked, share held, share held as 精锐, target covenants' layers / 300


class Layout:
    def __init__(self, n_cov):
        self.unit0 = 1
        self.shop0 = self.unit0 + MAX_UNITS
        self.offer0 = self.shop0 + SHOP
        self.hand0 = self.offer0 + OFFER
        self.item0 = self.hand0 + HAND
        self.cov0 = self.item0 + ITEMSHOP
        self.length = self.cov0 + n_cov


class Encoder:
    """Turns the gym's JSON observations into padded numpy arrays (and each env's legal actions into pointers)."""

    def __init__(self, spec):
        self.spec = spec
        self.kinds = {k: i for i, k in enumerate(spec["kinds"])}
        self.n_ops = len(spec["operators"])
        self.n_cov = len(spec["covenants"])
        self.n_items = len(spec["items"])
        self.layout = Layout(self.n_cov)
        # Run features: enemy types, boss, map (one-hot), the last 3 fights (2 each), elites, layer strength.
        # + field DPS against the deciding leader ÷ what it needs (capped), and whether it is reached.
        self.r_feats = len(spec["enemyTypes"]) + len(spec["bosses"]) + len(spec["maps"]) + 10
        # The next fight's enemy group as the game shows it in the 休整期 (gym spec previewLen; 0: older gyms).
        self.p_len = spec.get("previewLen", 0)
        profs = sorted({o["profession"] for o in spec["operators"]})
        feats = []
        for o in spec["operators"]:
            tier = np.eye(6)[o["tier"] - 1]
            cov = np.zeros(self.n_cov)
            cov[o["covenants"]] = 1
            prof = np.zeros(len(profs))
            prof[profs.index(o["profession"])] = 1
            feats.append(np.concatenate([tier, cov, prof, np.array(o["layersNormal"]) / 10, np.array(o["layersElite"]) / 10]))
        self.op_static = np.stack(feats).astype(np.float32)

    def encode(self, obs_list):
        B, L = len(obs_list), self.layout
        g = np.zeros((B, G_FEATS), np.float32)
        run = np.zeros((B, self.r_feats), np.float32)
        unit_op = np.full((B, MAX_UNITS), -1, np.int64)
        unit_f = np.zeros((B, MAX_UNITS, U_FEATS), np.float32)
        unit_items = np.full((B, MAX_UNITS, 2), -1, np.int64)
        shop = np.full((B, SHOP), -1, np.int64)
        offer = np.full((B, OFFER), -1, np.int64)
        hand = np.full((B, HAND), -1, np.int64)
        hand_adv = np.zeros((B, HAND), np.float32)
        item = np.full((B, ITEMSHOP), -1, np.int64)
        cov = np.zeros((B, self.n_cov, COV_FEATS), np.float32)
        level = np.zeros((B, 1), np.float32)
        expert = np.full(B, -1, np.int64)
        # The target lineup (gym --plan): per operator its wanted weight and whether it is a finished one of the locked
        # lineup (row n_ops = padding, stays 0); locked / share held / share held as 精锐; the target's covenants.
        want = np.zeros((B, self.n_ops + 1, 2), np.float32)
        plan_g = np.zeros((B, PLAN_G), np.float32)
        preview = np.zeros((B, max(1, self.p_len)), np.float32)
        plan_cov = np.zeros((B, self.n_cov, 1), np.float32)
        acts = []
        for b, o in enumerate(obs_list):
            g[b] = o["g"]
            run[b] = o["run"]
            for i, u in enumerate(o["units"][:MAX_UNITS]):
                unit_op[b, i] = u[0]
                unit_f[b, i] = [u[1], u[2] / 2, u[5] / 3, 1, u[6] if len(u) > 6 else 0, 0]   # elite, items, copies, held, fought last round
                unit_items[b, i] = [u[3], u[4]]
            for i, x in enumerate(o["shop"][:SHOP]):
                shop[b, i] = x
            for i, x in enumerate(o["offer"][:OFFER]):
                offer[b, i] = x
            for i, h in enumerate(o["hand"][:HAND]):
                hand[b, i] = h[0]
                hand_adv[b, i] = h[1]
            for i, x in enumerate(o["itemShop"][:ITEMSHOP]):
                item[b, i] = x
            cov[b] = np.array(o["cov"], np.float32)
            level[b, 0] = o.get("level", 0) / 20
            expert[b] = o.get("expert", -1)
            if self.p_len and o.get("preview"):
                preview[b] = o["preview"]
            pl = o.get("plan")
            if pl:
                for op, w, fin in pl["want"]:
                    want[b, int(op)] = (w, fin)
                plan_g[b, :len(pl["g"])] = pl["g"]
                plan_cov[b, :, 0] = pl["cov"]
            acts.append(self._actions(o["acts"]))
        A = max(len(a) for a in acts)
        kind = np.zeros((B, A), np.int64)
        ptr = np.zeros((B, A, 2), np.int64)
        amask = np.zeros((B, A), bool)
        for b, a in enumerate(acts):
            for j, (k, p0, p1) in enumerate(a):
                kind[b, j], ptr[b, j] = k, (p0, p1)
                amask[b, j] = True
        return dict(g=g, run=run, unit_op=unit_op, unit_f=unit_f, unit_items=unit_items, shop=shop, offer=offer, hand=hand,
                    hand_adv=hand_adv, item=item, cov=cov, kind=kind, ptr=ptr, amask=amask, level=level, expert=expert,
                    want=want, plan_g=plan_g, plan_cov=plan_cov, preview=preview)

    def _actions(self, acts):
        K, L = self.kinds, self.layout
        out = []
        for k, a, b in acts:
            name = self.spec["kinds"][k]
            p0 = p1 = 0  # 0 = the global token
            if name == "Buy":
                p0 = L.shop0 + a
            elif name == "Sell":
                p0 = L.unit0 + a
            elif name == "TakeOffer":
                p0 = L.offer0 + a
            elif name == "BuyItem":
                p0 = L.item0 + a
            elif name == "Equip":
                p0, p1 = L.hand0 + a, L.unit0 + b
            elif name == "ChooseGroup":
                k = len(self.spec["kinds"]) + a  # its own kind per group
            out.append((k, p0, p1))
        return out


class Policy(nn.Module):
    def __init__(self, enc: Encoder, d=192, layers=4, heads=6):
        super().__init__()
        self.enc = enc
        L = enc.layout
        self.register_buffer("op_static", torch.tensor(enc.op_static))
        self.op_id = nn.Embedding(enc.n_ops + 1, d)            # +1: padding / unknown
        self.op_proj = nn.Linear(enc.op_static.shape[1], d)
        self.item_id = nn.Embedding(enc.n_items + 1, d)
        self.cov_id = nn.Embedding(enc.n_cov, d)
        self.type_emb = nn.Embedding(7, d)                     # global, unit, shop, offer, hand, item, covenant
        self.g_proj = nn.Linear(G_FEATS, d)
        self.r_proj = nn.Sequential(nn.Linear(enc.r_feats, d), nn.GELU(), nn.Linear(d, d))
        self.u_proj = nn.Linear(U_FEATS, d)
        self.c_proj = nn.Linear(COV_FEATS, d)
        self.adv_proj = nn.Linear(1, d)
        # The difficulty (规程等级 / 20), added to the global token; starts at zero so older checkpoints load unchanged.
        self.level_proj = nn.Linear(1, d)
        nn.init.zeros_(self.level_proj.weight)
        nn.init.zeros_(self.level_proj.bias)
        # The target lineup (all zero at the start, so older checkpoints load unchanged).
        self.plan_op = nn.Linear(2, d)
        self.plan_g = nn.Linear(PLAN_G, d)
        self.plan_c = nn.Linear(1, d)
        # The enemy preview (zero at the start too).
        self.prev_proj = nn.Linear(max(1, enc.p_len), d)
        for m in (self.plan_op, self.plan_g, self.plan_c, self.prev_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        n_kinds = len(enc.spec["kinds"]) + 2
        self.kind_emb = nn.Embedding(n_kinds, d)
        self.act_mlp = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        # Auxiliary: the HP the next fight will cost (share of the start), so the body learns what makes leaks.
        self.aux = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d
        self.L = L

    def _op(self, ids):
        safe = ids.clamp(min=0)
        x = self.op_id(torch.where(ids < 0, torch.full_like(ids, self.enc.n_ops), ids)) + self.op_proj(self.op_static[safe])
        return x

    def _item(self, ids):
        return self.item_id(torch.where(ids < 0, torch.full_like(ids, self.enc.n_items), ids))

    def forward(self, t):
        B = t["g"].shape[0]
        dev = t["g"].device
        T = lambda i: self.type_emb.weight[i]
        glob = self.g_proj(t["g"]) + self.r_proj(t["run"]) + T(0)
        if "level" in t:
            glob = glob + self.level_proj(t["level"])
        glob = glob.unsqueeze(1)
        units = self._op(t["unit_op"]) + self.u_proj(t["unit_f"]) + self._item(t["unit_items"]).sum(2) + T(1)
        shop = self._op(t["shop"]) + T(2)
        offer = self._op(t["offer"]) + T(3)
        hand = self._item(t["hand"]) + self.adv_proj(t["hand_adv"].unsqueeze(-1)) + T(4)
        item = self._item(t["item"]) + T(5)
        cov = self.cov_id.weight.unsqueeze(0).expand(B, -1, -1) + self.c_proj(t["cov"]) + T(6)
        if "want" in t:
            def wf(ids):
                i = torch.where(ids < 0, torch.full_like(ids, self.enc.n_ops), ids)
                return self.plan_op(torch.gather(t["want"], 1, i.unsqueeze(-1).expand(-1, -1, 2)))
            units = units + wf(t["unit_op"])
            shop = shop + wf(t["shop"])
            offer = offer + wf(t["offer"])
            glob = glob + self.plan_g(t["plan_g"]).unsqueeze(1)
            cov = cov + self.plan_c(t["plan_cov"])
        if "preview" in t:
            glob = glob + self.prev_proj(t["preview"]).unsqueeze(1)
        x = torch.cat([glob, units, shop, offer, hand, item, cov], 1)
        pad = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=dev), t["unit_op"] < 0, t["shop"] < 0, t["offer"] < 0,
            t["hand"] < 0, t["item"] < 0, torch.zeros(B, self.enc.n_cov, dtype=torch.bool, device=dev)], 1)
        h = self.norm(self.body(x, src_key_padding_mask=pad))
        g = h[:, 0]
        pooled = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True)
        # Actions: kind, the token(s) they point at, the global token.
        A = t["kind"].shape[1]
        idx0 = t["ptr"][..., 0].unsqueeze(-1).expand(-1, -1, self.d)
        idx1 = t["ptr"][..., 1].unsqueeze(-1).expand(-1, -1, self.d)
        p0 = torch.gather(h, 1, idx0)
        p1 = torch.gather(h, 1, idx1)
        a = torch.cat([self.kind_emb(t["kind"]), p0, p1, g.unsqueeze(1).expand(-1, A, -1)], -1)
        logits = self.act_mlp(a).squeeze(-1).masked_fill(~t["amask"], -1e9)
        gp = torch.cat([g, pooled], -1)
        # detach_value: the value head learns alone (a warm-up on new rewards must not move the shared body / policy).
        value = self.value(gp.detach() if getattr(self, "detach_value", False) else gp).squeeze(-1)
        self.last_aux = self.aux(gp).squeeze(-1)
        return logits, value


def load_compatible(model, state):
    """Loads what fits: tensors of the same shape as they are; a weight whose input side grew (new features appended,
    e.g. the plan's global inputs) gets its old columns and zeros for the new ones; anything else stays as initialised."""
    own = model.state_dict()
    out = {}
    for k, v in state.items():
        if k not in own:
            continue
        o = own[k]
        if o.shape == v.shape:
            out[k] = v
        elif o.dim() == 2 and v.dim() == 2 and o.shape[0] == v.shape[0] and o.shape[1] > v.shape[1]:
            w = torch.zeros_like(o)
            w[:, :v.shape[1]] = v
            out[k] = w
        elif k == "kind_emb.weight" and o.dim() == 2 and o.shape[1] == v.shape[1] and o.shape[0] > v.shape[0]:
            # New action kinds (appended to the game's list): the old kinds keep their rows; the last two rows are the
            # two core-group choices, which come after the game's kinds; a new kind starts as the mean of the old ones.
            w = v.mean(0, keepdim=True).repeat(o.shape[0], 1)
            w[:v.shape[0] - 2] = v[:-2]
            w[-2:] = v[-2:]
            out[k] = w
    model.load_state_dict(out, strict=False)
    return out


def to_torch(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
