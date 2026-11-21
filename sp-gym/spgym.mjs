// spgym.mjs — a batch of solo 规程 matches on Stronghold-Protocol (virtual time, the real simulation), speaking the JSON
// line protocol of ASP Assistant's garrison_rl gym (python/garrison_rl/gym_client.py).
//
//   node spgym.mjs --version act3autochess@V3_1 --mode 规程等级8 --envs 16 [--rehearsal 3] [--max-actions 40]
//
// The policy owns the prep economy of the seat — buy (operator / equipment), sell, refresh, level up, freeze, take a
// 晋升奖励 — and ends it; the game's own AI ("legacy", server/match/bot.js) then lays the field out, equips, uses the 指令
// and readies, and also picks the 指令 at the start and the 机变 cards. One observation per decision.
//
//   {"cmd":"reset","seed":S}            -> {obs, reward, done, info, phi, fight, parts}
//   {"cmd":"step","actions":[...]}      -> the same; an action is an index into that env's obs.acts
//   {"cmd":"record","matches":N,"seed":S,"out":file,"fights":file}
//                                       -> legacy plays N whole matches; every shop call it makes is written as the
//                                          observation before it with "expert" = the action's index (behaviour cloning)
//   {"cmd":"close"}
import '../Stronghold-Protocol/tools/versions/runtime.mjs';
import { appendFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { Match } from '../Stronghold-Protocol/server/match/Match.js';
import { VirtualScheduler } from '../Stronghold-Protocol/server/match/scheduler.js';
import { getData } from '../Stronghold-Protocol/server/data.js';
import { defineBotStrategy, LEGACY_BOT } from '../Stronghold-Protocol/server/match/botStrategyContract.js';
import { RehearsalScore } from '../Stronghold-Protocol/server/sim/rehearsal.js';
import { createBotPrepSteps, createDelegatedRehearsalJob, arrangeSteps, botUseCommand, botIntent, _spSelfPlan, _spPagerUse } from '../Stronghold-Protocol/server/match/bot.js';

const argv = process.argv.slice(2);
const arg = (name, dflt = null) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt; };
const MODE = arg('--mode', '规程等级8');
const LEVEL = Number(String(MODE).replace(/\D+/g, '')) || 8;
const MODE_ID = `mode_single_${LEVEL}`;
const ENVS = Number(arg('--envs', 8));
// --bands a,b,c: every match takes one of these 策略 (by its seed, evenly) instead of the game AI's own pick.
const bandHash = (x) => { x = Math.imul(x ^ (x >>> 16), 0x85ebca6b); x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35); return (x ^ (x >>> 16)) >>> 0; };
const BANDS = arg('--bands') ? String(arg('--bands')).split(',').filter(Boolean) : null;
// --focus a,b,c: every match is given one of these core 盟约 as its main (by its seed, evenly, among those the match has
// on): the plan the policy sees (obs.plan) and the main the game AI's layout / equipment work for, instead of the AI's
// own pick (which carries rules: a prior per 盟约 — 萨尔贡 +7, 谢拉格 +5 ... 炎 -6 —, the 策略's own 盟约, 谢拉格 on 杰西卡).
// Each 盟约 gets the same share of the matches, so what each is worth shows in its own results.
const FOCUS = arg('--focus') ? String(arg('--focus')).split(',').filter(Boolean) : null;
// --second a,b,c: with --focus, every match is also given one of these 盟约 as its second (the plan shows it at half
// weight, the layout favours its members after the main's). {"cmd": "weights", "main": {leader: {盟约: w}}, "second":
// {...}} (no reply) sets how the two are drawn from then on: for the match's leader, in proportion to the weights
// given (the trainer's: each 盟约's recent win rate against that leader, with a floor); without weights, evenly.
const SECOND = arg('--second') ? String(arg('--second')).split(',').filter(Boolean) : null;
// --band-focus band=盟约,...: a match played with that 策略 is always given that 盟约 as its main (when the match has it
// on) instead of a drawn one — for a 策略 tied to one 盟约 (克莱门莎: 阿戈尔). Its info carries fixed: 1.
const BAND_FOCUS = new Map((arg('--band-focus') ? String(arg('--band-focus')).split(',') : []).filter(Boolean).map((x) => x.split('=')));
let WEIGHTS = null;
function drawBond(list, table, bossId, u) {
  if (!list.length) return null;
  const w = table && table[bossId];
  if (!w) return list[Math.min(list.length - 1, Math.floor(u * list.length))];
  const ws = list.map((id) => Math.max(0, Number(w[id]) || 0));
  const total = ws.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return list[Math.min(list.length - 1, Math.floor(u * list.length))];
  let x = u * total;
  for (let i = 0; i < list.length; i++) { x -= ws[i]; if (x < 0) return list[i]; }
  return list[list.length - 1];
}
const REHEARSAL = arg('--rehearsal') == null ? undefined : Number(arg('--rehearsal'));
// --boss-rehearsal N: the leader rounds (the 最终攻势 and the 隐秘核心) rehearse N layouts whatever --rehearsal says: that
// is where the rehearsal earns most (its leader plans and damage lineups), and it is one prep in fourteen.
const BOSS_REHEARSAL = arg('--boss-rehearsal') == null ? null : Number(arg('--boss-rehearsal'));
function setRehearsal(slot, m) {
  if (BOSS_REHEARSAL == null) return;
  if (slot.baseRehearsal == null) slot.baseRehearsal = m.botRehearsal;
  m.botRehearsal = m.round >= m.gd.bossRound ? BOSS_REHEARSAL : slot.baseRehearsal;
}
const MAX_ACTIONS = Number(arg('--max-actions', 40));
const MAX_UNITS = 20, OFFER = 3, HAND = 5;

const quiet = { warn() {}, error() {}, info() {}, debug() {} };
const out = (o) => process.stdout.write(`${JSON.stringify(o)}${String.fromCharCode(10)}`);
const data = getData({ log: quiet });
if (!data.config.modes[MODE_ID]) throw new RangeError(`no mode ${MODE_ID} in this version`);
const DIFFICULTY = data.config.modes[MODE_ID].difficulty;

// ---- catalogs (the spec) ----
const KINDS = ['Buy', 'Sell', 'Refresh', 'Upgrade', 'EndRound', 'TakeOffer', 'SkipOffer', 'BuyItem', 'UseItem', 'Equip', 'Freeze'];
const K = Object.fromEntries(KINDS.map((k, i) => [k, i]));
const chessBase = Object.values(data.chess).filter((c) => !c.isGolden).sort((a, b) => (a.chessId < b.chessId ? -1 : 1));
const OP = new Map(chessBase.map((c, i) => [c.chessId, i]));
const bondIds = Object.keys(data.bonds);
const COV = new Map(bondIds.map((b, i) => [b, i]));
const itemBase = Object.values(data.items).filter((c) => !c.isGolden).sort((a, b) => (a.id < b.id ? -1 : 1));
const ITEM = new Map(itemBase.map((c, i) => [c.id, i]));
const bandIds = Object.keys(data.bands);
const bossIds = Object.keys(data.bosses);
const stageIds = Object.keys(data.stages);
const PREVIEW = 12;
const zeros = (n) => new Array(n).fill(0);
const opOf = (id) => { const c = data.chess[id]; return c ? (OP.get(c.baseId) ?? -1) : -1; };
const itemOf = (id) => { const c = data.items[id]; return c ? (ITEM.get(c.baseId) ?? -1) : -1; };
const spec = {
  operators: chessBase.map((c) => ({ name: c.name, id: c.chessId, tier: c.tier, covenants: (c.bonds || []).map((b) => COV.get(b)).filter((x) => x != null),
    profession: c.profession || '', layersNormal: zeros(bondIds.length), layersElite: zeros(bondIds.length) })),
  covenants: bondIds.map((b) => ({ name: data.bonds[b].name, id: b, needed: data.bonds[b].thresholds?.[0] ?? data.bonds[b].activeCount ?? 1 })),
  items: itemBase.map((c) => ({ name: c.name, id: c.id, combat: c.itemType === 'EQUIP' })),
  kinds: KINDS,
  // The run's one-hots: the 指令 (band) taken, the leader, the stage.
  enemyTypes: bandIds, bosses: bossIds, maps: stageIds,
  rounds: 15, maxShopSlots: 5, maxActionsPerRound: MAX_ACTIONS, previewLen: PREVIEW, sim: 'stronghold-protocol',
};

// ---- the legacy prep, split ----
const noop = () => {};
// eslint-disable-next-line require-yield
const none = function* () { return null; };
// The layout half: no procurement (the policy did it); a 晋升奖励 the policy left is still taken.
const LAYOUT_ONLY = createBotPrepSteps({ sellJunk: noop, buyLoopSteps: none, levelUp: noop, planSpendSteps: none, spendDown: noop, maybeFreeze: noop });
function designate(slot, m, ps) {
  if (slot.focus === undefined) {
    const gd = ps.gd || m.gd;
    const live = (id) => !gd.modeInactiveBonds.has(id) && (!gd.regulationBondDraw || m.bondPlayable(id));
    const on = FOCUS.filter((id) => gd.bond(id)?.isCore && live(id));
    const tied = BAND_FOCUS.get(ps.bandId);
    slot.fixed = tied && gd.bond(tied)?.isCore && live(tied) ? 1 : 0;
    if (WEIGHTS || SECOND) {
      slot.focus = slot.fixed ? tied : drawBond(on, WEIGHTS?.main, m.bossId, bandHash(slot.seed ^ 0x5bd1e995) / 4294967296);
      slot.second = SECOND ? drawBond(SECOND.filter((id) => gd.bond(id) && live(id) && id !== slot.focus), WEIGHTS?.second, m.bossId,
        bandHash(slot.seed ^ 0x2545f491) / 4294967296) : null;
    } else {
      slot.focus = slot.fixed ? tied : on.length ? on[bandHash(slot.seed ^ 0x5bd1e995) % on.length] : null;
      slot.second = null;
    }
  }
  const id = slot.focus;
  const bonds = id ? [Object.freeze({ id, priority: 5 })] : [];
  if (id && slot.second) bonds.push(Object.freeze({ id: slot.second, priority: 3 }));
  ps._botSelf = id ? { round: m.round, plan: Object.freeze({ bonds: Object.freeze(bonds) }) } : null;
  if (!id) _spSelfPlan(m, ps);
}
function preamble(m, ps, slot) {
  ps.autoPickCommands();
  botUseCommand(m, ps, false);
  if (FOCUS && slot) designate(slot, m, ps); else _spSelfPlan(m, ps);
  _spPagerUse(m, ps);
}

class Slot {
  constructor(index) {
    this.index = index;
    this.m = null; this.sched = null; this.ps = null;
    this.pending = null; this.ended = false; this.summary = null;
    this.seed = 0;
    this.reset();
  }
  reset() {
    this.round = 0; this.lp = 0; this.startLp = 1; this.layerMark = 0; this.history = [];
    this.gains = []; this.losses = []; this.lastLayers = 0; this.bossShare = -1; this.teach = null;
    this.uids = []; this.shopMap = []; this.itemMap = []; this.acts = []; this.actions = 0; this.pendingParts = null;
    this.fightRows = null; this.errors = [];
  }
}

function usableLayers(ps) {
  let sum = 0;
  for (const b of ps.privateView().bonds || []) if (b.active && !b.off && !data.bonds[b.bondId]?.noStack) sum += b.layers || 0;
  return sum;
}

function newMatch(slot, seed, teacher) {
  slot.reset();
  slot.seed = seed; slot.pending = null; slot.ended = false; slot.summary = null; slot.focus = undefined; slot.second = null;
  const sched = new VirtualScheduler();
  const TEACH = teacher ? createBotPrepSteps({
    *arrangeSteps(m, ps, o) {
      // Legacy's procurement is over at its first arrangement: that is where the policy ends its prep.
      if (slot.teach && slot.teach.round === m.round && !slot.teach.ended) { slot.teach.ended = true; teachSample(slot, K.EndRound, 0); unwrap(slot); }
      return yield* LEGACY_ARRANGE(m, ps, o);
    },
  }) : null;
  const strategy = defineBotStrategy({
    id: 'sp-gym',
    decide(m, ps, d) {
      if (d.kind !== 'prep' || teacher) return false;
      onPrepStart(slot, m, ps);
      preamble(m, ps, slot);
      slot.preambleRound = m.round;
      slot.pending = { d, ps };
      slot.actions = 0;
      return true;
    },
    *prepBegin(m, ps, o) {
      if (teacher) {
        onPrepStart(slot, m, ps);
        setRehearsal(slot, m);
        slot.teach = { round: m.round, ended: false };
        slot.actions = 0;
        wrap(slot, ps);
        preamble(m, ps, slot);
        return hookJob(slot, m, ps, yield* TEACH.prepBegin(m, ps, o));
      }
      setRehearsal(slot, m);
      if (slot.preambleRound !== m.round) preamble(m, ps, slot);
      return hookJob(slot, m, ps, yield* LAYOUT_ONLY.prepBegin(m, ps, PICKER ? { ...o, delegate: true } : o));
    },
    *prepEnd(m, ps, job) {
      if (teacher) unwrap(slot);
      const r = LEGACY_BOT.prepEnd(m, ps, job);
      if (r && typeof r.next === 'function') yield* r;
      captureBoard(slot);
    },
  });
  const seats = [{ seat: 0, playerId: 'ai_0', name: 'AI-1', isBot: true, connected: true }];
  const log = { info() {}, debug() {}, warn() {}, error: (...a) => { if (slot.errors.length < 5) slot.errors.push(a.map(String).join(' ').slice(0, 300)); } };
  slot.sched = sched;
  slot.m = new Match({ roomCode: 'GYM', mode: 'solo', difficulty: DIFFICULTY, modeId: MODE_ID, seats, seed, data, log, scheduler: sched,
    botRehearsal: REHEARSAL, botStrategy: strategy, botStrategySeats: {}, send: () => true, broadcast: () => {},
    onEnd: (s) => { slot.summary = s; slot.ended = true; } });
  if (BANDS) {
    // a 策略 tied to one 盟约 (--band-focus) is only played when the match has that 盟约 on; otherwise one of the others
    const m0 = slot.m, gd0 = m0.gd, h = bandHash(seed);
    // (solo 规程: the core 盟约 come in two groups and the game AI takes the first — settled only in INFO_CHECK, after this)
    const on0 = (id) => !!gd0.bond(id) && !gd0.modeInactiveBonds.has(id)
      && (m0.bondChoice?.groups ? m0.bondChoice.groups[0].includes(id) : (!gd0.regulationBondDraw || m0.bondPlayable(id)));
    let band = BANDS[h % BANDS.length];
    if (BAND_FOCUS.has(band) && !on0(BAND_FOCUS.get(band))) {
      const free = BANDS.filter((x) => !BAND_FOCUS.has(x));
      if (free.length) band = free[bandHash(h ^ 0x9e3779b9) % free.length];
    }
    m0.botBands.set('ai_0', band);
  }
  slot.ps = slot.m.players.get('ai_0');
  if (PICKER && !teacher) {
    const m = slot.m;
    m.delegateRehearsal = ({ ps, chosen, plans, inputs, cap = null, limits = null }) => {
      const job = createDelegatedRehearsalJob(m, ps, chosen, plans, inputs, cap, limits);
      let row = null;
      try { row = layoutRow(slot, m, ps, chosen, plans); } catch { row = null; }
      if (!row || row.cands.length < 2) return null;   // nothing to choose: the default plan, simulated as usual
      slot.query = { job, row };
      return job;
    };
  }
  slot.m.start();
}

const LEGACY_ARRANGE = arrangeSteps;

/** A new round's prep begins: what the last fight did (for the reward), the marks of the new one. */
function onPrepStart(slot, m, ps) {
  const lp = Math.max(0, ps.lp || 0);
  if (slot.round === 0) { slot.startLp = Math.max(1, lp); slot.lp = lp; slot.layerMark = 0; slot.lastLayers = 0; }
  else if (m.round !== slot.round) slot.pendingParts = fightParts(slot, m, ps, false);
  slot.round = m.round;
}

/** survived, HP lost (share of the start), cleared, died, leader share taken (-1: no leader), leader killed, time,
 *  hidden core round, hidden core beaten, usable layers gained / 100 (above the run's best), plan progress (0). */
function fightParts(slot, m, ps, ended) {
  const s = slot.summary;
  const round = slot.round;
  const lpNow = ended ? Math.max(0, s?.teamLp ?? ps.lp ?? 0) : Math.max(0, ps.lp || 0);
  const died = ended && !s?.victory && !(s?.roundsPassed >= m.gd.bossRound);
  const lost = Math.max(0, slot.lp - (died ? 0 : lpNow)) / slot.startLp;
  slot.lp = lpNow;
  const bossRound = round === m.gd.bossRound, hiddenRound = round === m.gd.hiddenRound;
  let share = -1, killed = 0;
  if (bossRound || hiddenRound) {
    const pool = m.bossPool;
    share = pool && pool.maxHp > 0 ? 1 - Math.max(0, pool.hp) / pool.maxHp : 0;
    if (bossRound) { killed = (ended ? !!s?.victory : true) ? 1 : 0; if (killed) share = 1; }
    else { killed = ended && s?.hiddenCleared ? 1 : 0; if (killed) share = 1; }
  }
  const cleared = bossRound && killed ? 1 : 0;
  const usable = usableLayers(ps);
  const gain = round < m.gd.bossRound ? Math.max(0, usable - slot.layerMark) : 0;
  slot.layerMark = Math.max(slot.layerMark, usable);
  slot.gains.push(Math.round((usable - slot.lastLayers) * 10) / 10);
  slot.lastLayers = usable;
  slot.losses.push(Math.round(lost * slot.startLp * 10) / 10);
  slot.history.push({ lost, leaks: ps.lastResult?.leaks ?? 0 });
  const parts = [died ? 0 : 1, lost, cleared, died ? 1 : 0, share, killed, 1, hiddenRound ? 1 : 0, hiddenRound && killed ? 1 : 0, gain / 100, 0];
  writeFight(slot, ps, parts, ended);
  return parts;
}

// ---- observation ----
function previewOf(m, v) {
  const out = zeros(PREVIEW);
  const list = v.nextEnemies || [];
  let n = 0, fly = 0, elite = 0, boss = 0, upper = 0, lower = 0, last = 0, hp = 0, atk = 0, def = 0, res = 0;
  const keys = new Set();
  for (const e of list) {
    const c = e.count || 1;
    n += c; if (e.fly) fly += c; if (e.elite) elite += c; if (e.boss) boss = 1;
    if (e.gate === 'upper') upper += c; else lower += c;
    last = Math.max(last, e.t || 0);
    keys.add(e.enemyKey);
    const st = data.enemies[e.enemyKey]?.stats || data.enemies[e.enemyKey]?.attributes;
    if (st) { hp += c * (st.maxHp || 0); atk = Math.max(atk, st.atk || 0); def += c * (st.def || 0); res += c * (st.res ?? st.magicResistance ?? 0); }
  }
  out[0] = n / 60; out[1] = fly / 20; out[2] = elite / 10; out[3] = boss; out[4] = upper / 40; out[5] = lower / 40; out[6] = last / 120;
  out[7] = keys.size / 10; out[8] = Math.log10(1 + hp) / 7; out[9] = Math.log10(1 + atk) / 4; out[10] = n ? def / n / 1000 : 0; out[11] = n ? res / n / 100 : 0;
  return out;
}

function planOf(slot, m, ps, held) {
  let intent = null;
  try { intent = botIntent(m, ps, 'prep'); } catch { intent = null; }
  const cov = zeros(bondIds.length);
  const want = [];
  const given = FOCUS ? slot.focus : null;
  if (!given && (!intent || !intent.focus)) return { want, g: [0, 0, 0, 0], cov };
  const focus = given || intent.focus, second = given && SECOND ? slot.second : (intent && intent.second !== focus ? intent.second : null);
  if (COV.has(focus)) cov[COV.get(focus)] = 1;
  if (second && COV.has(second)) cov[COV.get(second)] = 0.5;
  let members = 0, have = 0, elite = 0;
  for (const c of chessBase) {
    const inF = (c.bonds || []).includes(focus), inS = second && (c.bonds || []).includes(second);
    if (!inF && !inS) continue;
    const i = OP.get(c.chessId);
    want.push([i, inF ? 1 : 0.5, inF ? 1 : 0]);
    if (inF) { members++; const h = held.get(c.chessId); if (h) { have++; if (h.golden) elite++; } }
  }
  const th = data.bonds[focus]?.thresholds || [1];
  const top = th[th.length - 1] || 1;
  const layers = (ps.layers?.[focus] ?? 0);
  return { want, g: [1, Math.min(1, have / top), Math.min(1, elite / top), Math.min(3, (Number(layers) || 0) / 300)], cov };
}

function observe(slot) {
  const m = slot.m, ps = slot.ps;
  const v = ps.privateView();
  const shop = v.shop;
  const pieces = ps.allChess();
  const onBoard = new Set((v.board || []).filter(Boolean).map((p) => p.uid));
  const baseCount = new Map(), held = new Map();
  for (const p of pieces) {
    const base = m.gd.baseIdOf(p.id);
    baseCount.set(base, (baseCount.get(base) || 0) + 1);
    const h = held.get(base) || { golden: false };
    if (m.gd.isGolden(p.id)) h.golden = true;
    held.set(base, h);
  }
  slot.uids = pieces.slice(0, MAX_UNITS).map((p) => p.uid);
  const units = pieces.slice(0, MAX_UNITS).map((p) => {
    const its = (p.items || []).map((x) => itemOf(x.id ?? x));
    return [opOf(p.id), m.gd.isGolden(p.id) ? 1 : 0, its.length, its[0] ?? -1, its[1] ?? -1, baseCount.get(m.gd.baseIdOf(p.id)) || 1, onBoard.has(p.uid) ? 1 : 0];
  });
  const items = ps.itemCards().slice(0, HAND);
  const funds = ps.funds || 0;
  const handFullChess = ps.equipHand ? ps.handSlot('chess') < 0 : ps.hand.findIndex((p) => p == null) < 0;
  const handFullItem = ps.equipHand ? ps.handSlot('item') < 0 : handFullChess;
  // shop: operator slots in their order, the equipment slot apart
  slot.shopMap = []; slot.itemMap = [];
  const shopOps = [], shopItems = [];
  const acts = [];
  const capped = slot.actions >= MAX_ACTIONS;
  shop.slots.forEach((s, i) => {
    if (!s) return;
    if (s.kind === 'chess') {
      const j = shopOps.length;
      slot.shopMap.push(i);
      shopOps.push(s.sold ? -1 : opOf(s.id));
      if (!capped && !s.sold && funds >= s.price && (!handFullChess)) acts.push([K.Buy, j, 0]);
    } else if (s.kind === 'item' && shopItems.length < 1) {
      slot.itemMap.push(i);
      shopItems.push(s.sold ? -1 : itemOf(s.id));
      if (!capped && !s.sold && funds >= s.price && !handFullItem) acts.push([K.BuyItem, 0, 0]);
    }
  });
  const offer = [];
  const off = ps.offers?.[0];
  if (off) off.slots.slice(0, OFFER).forEach((s, i) => {
    const isChess = s && s.kind === 'chess' && !s.sold;
    offer.push(isChess ? opOf(s.id) : -1);
    if (!capped && isChess && !handFullChess) acts.push([K.TakeOffer, i, 0]);
  });
  if (!capped) {
    units.forEach((u, i) => acts.push([K.Sell, i, 0]));
    if (shop.freeRefreshes > 0 || funds >= shop.refreshPrice) acts.push([K.Refresh, 0, 0]);
    if (shop.level < shop.maxLevel && funds >= Math.max(0, shop.upgradePrice)) acts.push([K.Upgrade, 0, 0]);
    if (!shop.frozen && shop.slots.some((s) => s && !s.sold)) acts.push([K.Freeze, 0, 0]);
  }
  acts.push([K.EndRound, 0, 0]);
  slot.acts = acts;
  const bondsById = new Map((v.bonds || []).map((b) => [b.bondId, b]));
  const off2 = new Set(m.disabledBonds || []);
  let activeLayers = 0;
  const cov = bondIds.map((id) => {
    const b = bondsById.get(id), rec = data.bonds[id];
    const need = rec.thresholds?.[0] ?? 1, tiers = rec.thresholds?.length || 1;
    if (b?.active) activeLayers += b.layers || 0;
    return [b ? Math.min(3, (b.count || 0) / need) : 0, ((b?.layers ?? ps.layers?.[id]) || 0) / 100, b?.active ? 1 : 0, b ? (b.tier || 0) / tiers : 0, off2.has(id) || b?.off ? 1 : 0];
  });
  const run = [...bandIds.map((b) => (ps.bandId === b ? 1 : 0)), ...bossIds.map((b) => (m.bossId === b ? 1 : 0)), ...stageIds.map((s) => (m.stageId === s ? 1 : 0))];
  for (let k = 1; k <= 3; k++) { const h = slot.history[slot.history.length - k]; run.push(h ? h.lost : 0, h ? Math.min(1, (h.leaks || 0) / 20) : 0); }
  run.push(pieces.filter((p) => m.gd.isGolden(p.id)).length / 8, Math.min(2, activeLayers / 500), slot.startLp / 45, Math.min(2, activeLayers / 350));
  const g = [m.round / 15, Math.max(0, ps.lp || 0) / slot.startLp, funds / 10, shop.level / 6, Math.max(0, shop.upgradePrice) / 10, (v.deployCap || 8) - 8,
    pieces.length / 18, items.length / 5, slot.actions / 20, (ps.round?.buys || 0) / 5, (ps.round?.refreshes || 0) / 5, Math.min(3, shop.freeRefreshes || 0) / 3,
    (v.deployCount || 0) / 8, Math.min(10, ps.pendingFunds || 0) / 5, v.command?.usable ? 1 : 0, off ? 1 : 0, shop.frozen ? 1 : 0];
  return {
    g, run, preview: previewOf(m, v), units, shop: shopOps, offer,
    hand: items.map((p) => [itemOf(p.id), data.items[p.id]?.isGolden ? 1 : 0]), itemShop: shopItems, cov, acts, expert: -1,
    plan: planOf(slot, m, ps, held),
  };
}

// ---- acting ----
function intentOf(slot, a) {
  const [kind, i] = a;
  if (kind === K.Buy) return { t: 'g.buy', slot: slot.shopMap[i] };
  if (kind === K.BuyItem) return { t: 'g.buy', slot: slot.itemMap[i] };
  if (kind === K.Sell) return { t: 'g.sell', uid: slot.uids[i] };
  if (kind === K.Refresh) return { t: 'g.refresh' };
  if (kind === K.Upgrade) return { t: 'g.levelUp' };
  if (kind === K.Freeze) return { t: 'g.freeze' };
  if (kind === K.TakeOffer) return { t: 'g.reward', idx: i };
  return null;
}

function advance(slot) {
  let guard = 0;
  while (!slot.pending && !slot.ended && !slot.query) {
    if (!slot.sched.runNext()) { if (slot.query) break; slot.ended = true; slot.stuck = true; break; }
    if (++guard > 5e6) { slot.ended = true; slot.stuck = true; break; }
  }
}

function infoOf(slot) {
  const s = slot.summary, p = s?.players?.[0];
  const m = slot.m, ps = slot.ps;
  const bonds = (p?.bonds || []).filter((b) => b.active).map((b) => b.layers || 0).sort((a, b) => b - a);
  const lineup = p?.lineup || [];
  return { cleared: !!s?.victory, hiddenCore: s ? (s.hiddenReached ? !!s.hiddenCleared : null) : null, rounds: s?.roundsPassed ?? slot.round, hp: s?.teamLp ?? 0,
    level: ps?.shop?.level ?? 0, dpsRatio: 0, elites: ps ? ps.allChess().filter((q) => m.gd.isGolden(q.id)).length : lineup.filter((x) => x.golden).length,
    layers: bonds.slice(0, 4), gains: slot.gains, losses: slot.losses, band: p?.bandId ?? null, boss: s?.bossId ?? null, seed: slot.seed, focus: slot.focus ?? null, second: slot.second ?? null, fixed: slot.fixed || 0,
    stuck: !!slot.stuck, errors: slot.errors, plan: planInfo(slot, m, ps) };
}
/** How far the run got with the 盟约 it was given (--focus): its different members held and those 精锐, each as a share
 *  of the 盟约's top threshold, and its layers / 300. */
function planInfo(slot, m, ps) {
  const id = FOCUS ? slot.focus : null;
  if (!id || !ps) return null;
  const count = (bond) => {
    const all = new Set(), elite = new Set();
    for (const q of ps.allChess()) {
      const c = data.chess[q.id];
      if (!c || !(c.bonds || []).includes(bond)) continue;
      all.add(c.baseId ?? q.id);
      if (m.gd.isGolden(q.id)) elite.add(c.baseId ?? q.id);
    }
    const th = data.bonds[bond]?.thresholds || [1];
    const top = th[th.length - 1] || 1;
    return [Math.min(1, all.size / top), Math.min(1, elite.size / top)];
  };
  const [held, elite] = count(id);
  const out = { held, elite, layers: Math.min(3, (Number(ps.layers?.[id]) || 0) / 300) };
  if (slot.second) { out.held2 = count(slot.second)[0]; out.layers2 = Math.min(3, (Number(ps.layers?.[slot.second]) || 0) / 300); }
  return out;
}

const FIGHTS = arg('--fights');
// --layouts file: every layout rehearsal's candidates (each one's tiles and facings) with what its simulated fight did
// — counted leaks, leader damage, layers gained, kills — and the one the game AI took. With it every candidate is
// played out (no early stop), so each has its full result.
const LAYOUTS = arg('--layouts');
// --picker: a prep's candidate layouts are not simulated here; they go out as {"layoutQueries": [...]} and the reply
// {"picks": [...]} names the one to use (a model that learned the rehearsal, python sp_layout.py). --rehearsal sets
// how many candidates are offered.
const PICKER = argv.includes('--picker');
let layoutCapture = null;
if (LAYOUTS) {
  const note = RehearsalScore.prototype.note;
  RehearsalScore.prototype.note = function (battle, index) {
    try {
      const r = battle && typeof battle.result === 'function' ? battle.result() : null;
      const pp = r && !r.synthetic && r.perPlayer && r.perPlayer[this.playerId];
      if (pp && layoutCapture) {
        layoutCapture.push({ i: index, leaks: (pp.leaked || []).filter((l) => l && l.counted !== false).length,
          boss: Math.max(0, Number(pp.bossDamage) || 0), killed: pp.killed || 0,
          gain: Object.values(pp.layerGains || {}).reduce((a, n) => a + (Number(n) || 0), 0), leader: battle.kind === 'boss' || battle.kind === 'hidden' ? 1 : 0 });
      }
    } catch { /* a label less */ }
    return note.call(this, battle, index);
  };
  RehearsalScore.prototype.alive = () => true;
}
function hookJob(slot, m, ps, job) {
  if (!LAYOUTS || !job || typeof job.run !== 'function' || job._sp) return job;
  job._sp = true;
  const run = job.run;
  const cap = [];
  job.run = function (...a) {
    const prev = layoutCapture;
    layoutCapture = cap;
    let done;
    try { done = run.apply(job, a); } finally { layoutCapture = prev; }
    if (done && !job._spWritten) { job._spWritten = true; try { writeLayouts(slot, m, ps, job, cap); } catch { /* a row less */ } }
    return done;
  };
  return job;
}
function stateRow(slot, m, ps, v) {
  const enemies = {};
  for (const e of v.nextEnemies || []) enemies[e.enemyKey] = (enemies[e.enemyKey] || 0) + (e.count || 1);
  return { level: LEVEL, seed: slot.seed, round: m.round, stage: m.stageId, band: ps.bandId, boss: m.bossId,
    lp: Math.max(0, ps.lp || 0), startLp: slot.startLp, shopLevel: ps.shop.level,
    bench: ps.allChess().length - (v.deployCount || 0),
    bonds: (v.bonds || []).map((b) => [COV.get(b.bondId), b.count || 0, b.tier || 0, b.layers || 0, b.active ? 1 : 0]),
    enemies, preview: previewOf(m, v).map((x) => Math.round(x * 1e4) / 1e4) };
}
function layoutRow(slot, m, ps, chosen, plans) {
  const byUid = new Map(chosen.map((q) => [q.uid, q]));
  const cands = plans.map((plan) => [...plan.entries()].map(([uid, key]) => {
    const q = byUid.get(uid);
    if (!q || q.kind !== 'chess') return null;
    const [row, col] = String(key).split(',').map(Number);
    return [opOf(q.id), m.gd.isGolden(q.id) ? 1 : 0, row, col, typeof plan.dirOf === 'function' ? plan.dirOf(uid) : 'RIGHT', (q.items || []).map((x) => itemOf(x.id ?? x))];
  }).filter(Boolean));
  const row = stateRow(slot, m, ps, ps.privateView());
  row.cands = cands;
  return row;
}
function writeLayouts(slot, m, ps, job, cap) {
  if (cap.length < 2) return;
  const row = layoutRow(slot, m, ps, job.chosen, job.plans);
  row.scores = cap; row.choice = job.stats?.choice ?? job.plans.indexOf(job.best);
  appendFileSync(LAYOUTS, `${JSON.stringify(row)}${String.fromCharCode(10)}`);
}
let nextSeed = 1;
const slots = Array.from({ length: ENVS }, (_, i) => new Slot(i));

function reply(parts, done, info, fight) {
  const obs = slots.map((s) => observe(s));
  return { obs, reward: parts.map((p) => (p ? p[0] - p[1] + 10 * p[2] : 0)), done, info, phi: slots.map(() => [0, 0, 0, 0, 0, 0]), fight, parts };
}

async function resolveQueries() {
  for (let guard = 0; guard < 64; guard++) {
    const asking = slots.filter((s) => s.query);
    if (!asking.length) return;
    out({ layoutQueries: asking.map((s) => s.query.row) });
    const ans = JSON.parse(await nextLine());
    asking.forEach((s, k) => {
      const { job, row } = s.query;
      s.query = null;
      const pick = Number.isInteger(ans.picks?.[k]) && ans.picks[k] >= 0 && ans.picks[k] < row.cands.length ? ans.picks[k] : 0;
      try { job.accept(pick); job.onSettled?.(); } catch (e) { if (s.errors.length < 5) s.errors.push(String(e?.message || e).slice(0, 200)); }
      advance(s);
    });
  }
}

async function resetAll(seed) {
  nextSeed = seed;
  for (const s of slots) { newMatch(s, nextSeed++, false); advance(s); settleEnd(s, null); }
  await resolveQueries();
  return reply(slots.map(() => null), slots.map(() => false), slots.map(() => null), slots.map(() => -1));
}

/** A match that ended (or one that could not start): its info, then a fresh one up to its first decision. */
function settleEnd(slot, out) {
  let tries = 0;
  while (slot.ended && tries++ < 20) {
    if (out) {
      out.parts = slot.round > 0 ? fightParts(slot, slot.m, slot.ps, true) : null;
      out.info = infoOf(slot);
      out.done = true;
    }
    try { slot.m.dispose?.(); } catch { /* already gone */ }
    newMatch(slot, nextSeed++, false);
    advance(slot);
    if (out && out.done) out = null; // only the first end is reported
  }
}

async function stepAll(actions) {
  const n = slots.length;
  const parts = new Array(n).fill(null), done = new Array(n).fill(false), info = new Array(n).fill(null), fight = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const slot = slots[i];
    let a = slot.acts[actions[i]] || slot.acts[slot.acts.length - 1];
    const dec = slot.pending;
    if (!dec) continue;
    const { d, ps } = dec;
    const m = slot.m;
    if (!d.valid()) { slot.pending = null; }
    else if (a[0] === K.EndRound) {
      slot.pending = null;
      if (!d.complete()) d.fallback();
    } else {
      slot.actions++;
      const intent = intentOf(slot, a);
      try { if (intent) m._handleSeatIntent(ps, intent); } catch (e) { if (slot.errors.length < 5) slot.errors.push(String(e?.message || e).slice(0, 200)); }
      if (!d.valid() || d.complete()) slot.pending = null;
    }
    if (!slot.pending) {
      slot.pendingParts = null;
      slot.moved = true;
      advance(slot);
    }
  }
  await resolveQueries();
  for (let i = 0; i < n; i++) {
    const slot = slots[i];
    if (!slot.moved) continue;
    slot.moved = false;
    const res = { parts: slot.pendingParts, info: null, done: false };
    slot.pendingParts = null;
    if (slot.ended) settleEnd(slot, res);
    parts[i] = res.parts; info[i] = res.info; done[i] = res.done;
    if (res.parts) fight[i] = res.parts[1];
  }
  return reply(parts, done, info, fight);
}

// ---- the teacher: legacy's own shop calls, recorded ----
let recordOut = null, recorded = 0, unrepresented = 0;
function teachSample(slot, kind, index) {
  const o = observe(slot);
  const j = o.acts.findIndex((a) => a[0] === kind && a[1] === index);
  if (j < 0) { unrepresented++; return false; }
  o.expert = j; o.level = LEVEL; o.seed = slot.seed; o.round = slot.m.round;
  appendFileSync(recordOut, JSON.stringify(o) + '\n');
  recorded++;
  return true;
}
function wrap(slot, ps) {
  if (slot.wrapped) return;
  const saved = {};
  const hook = (name, toAct) => {
    saved[name] = Object.getOwnPropertyDescriptor(ps, name);
    const base = ps[name];
    ps[name] = function (...args) {
      if (slot.teach && !slot.teach.ended) {
        // the observation before the call; its mapping needs the maps observe() builds
        const act = toAct(...args);
        const o = act ? observe(slot) : null;
        const mapped = act ? act(o) : null;
        const j = mapped ? o.acts.findIndex((a) => a[0] === mapped[0] && a[1] === mapped[1]) : -1;
        const r = base.apply(this, args);
        if (!(r && r.error)) {
          if (j >= 0) { o.expert = j; o.level = LEVEL; o.seed = slot.seed; o.round = slot.m.round; appendFileSync(recordOut, JSON.stringify(o) + '\n'); recorded++; slot.actions++; }
          else unrepresented++;
        }
        return r;
      }
      return base.apply(this, args);
    };
  };
  hook('buy', (i) => () => { const s = ps.shop.slots[i]; if (!s) return null; return s.kind === 'chess' ? [K.Buy, slot.shopMap.indexOf(i)] : [K.BuyItem, slot.itemMap.indexOf(i)]; });
  hook('sell', (uid) => () => [K.Sell, slot.uids.indexOf(uid)]);
  hook('refresh', () => () => [K.Refresh, 0]);
  hook('freeze', () => () => [K.Freeze, 0]);
  hook('levelUp', () => () => [K.Upgrade, 0]);
  hook('pickReward', (i) => () => [K.TakeOffer, i]);
  slot.wrapped = saved;
}
function unwrap(slot) {
  const ps = slot.ps;
  if (!slot.wrapped || !ps) return;
  for (const [name, desc] of Object.entries(slot.wrapped)) { if (desc) Object.defineProperty(ps, name, desc); else delete ps[name]; }
  slot.wrapped = null;
}

// ---- fights (for the fight model): the field each real battle was fought with, and what the battle did ----
function captureBoard(slot) {
  const m = slot.m, ps = slot.ps;
  if (!FIGHTS || !ps || !ps.alive) return;
  const v = ps.privateView();
  const enemies = {};
  for (const e of v.nextEnemies || []) enemies[e.enemyKey] = (enemies[e.enemyKey] || 0) + (e.count || 1);
  slot.fightRow = { level: LEVEL, seed: slot.seed, round: m.round, stage: m.stageId, band: ps.bandId, boss: m.bossId,
    lp: Math.max(0, ps.lp || 0), startLp: slot.startLp, shopLevel: ps.shop.level, funds: ps.funds ?? null,
    units: (v.board || []).filter((p) => p && p.kind === 'chess').map((p) => [opOf(p.id), p.golden ? 1 : 0, p.row, p.col, p.dir, (p.items || []).map((x) => itemOf(x.id ?? x))]),
    bench: ps.allChess().length - (v.deployCount || 0),
    bonds: (v.bonds || []).map((b) => [COV.get(b.bondId), b.count || 0, b.tier || 0, b.layers || 0, b.active ? 1 : 0]),
    enemies, preview: previewOf(m, v).map((x) => Math.round(x * 1e4) / 1e4) };
}
function writeFight(slot, ps, parts, ended) {
  const row = slot.fightRow;
  slot.fightRow = null;
  if (!FIGHTS || !row) return;
  const after = ps.privateView().bonds || [];
  row.out = { lost: Math.round(parts[1] * slot.startLp * 100) / 100, died: parts[3], share: parts[4], killed: parts[5], ended: ended ? 1 : 0,
    leaks: ps.lastResult?.leaks ?? null, kills: ps.lastResult?.kills ?? null, funds: ps.funds ?? null,
    bonds: after.map((b) => [COV.get(b.bondId), b.layers || 0]) };
  appendFileSync(FIGHTS, `${JSON.stringify(row)}\n`);
}

function record(msg) {
  recordOut = msg.out; recorded = 0; unrepresented = 0;
  const slot = slots[0];
  const infos = [];
  for (let k = 0; k < msg.matches; k++) {
    newMatch(slot, msg.seed + k, true);
    let guard = 0;
    while (!slot.ended && guard++ < 5e6) if (!slot.sched.runNext()) break;
    unwrap(slot);
    if (slot.round > 0) fightParts(slot, slot.m, slot.ps, true);
    infos.push(infoOf(slot));
    try { slot.m.dispose?.(); } catch { /* gone */ }
  }
  return { recorded, unrepresented, matches: infos.length, wins: infos.filter((x) => x.cleared).length, info: infos };
}

// ---- protocol ----
process.on('uncaughtException', (e) => { process.stderr.write(`[spgym ${MODE} pid ${process.pid}] uncaught: ${e?.stack || e}
`); process.exit(3); });
process.on('unhandledRejection', (e) => { process.stderr.write(`[spgym ${MODE} pid ${process.pid}] rejection: ${e?.stack || e}
`); process.exit(4); });
process.on('exit', (code) => { if (code) process.stderr.write(`[spgym ${MODE} pid ${process.pid}] exit ${code}
`); });
out({ spec });
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
const lines = rl[Symbol.asyncIterator]();
async function nextLine() {
  for (;;) {
    const r = await lines.next();
    if (r.done) process.exit(0);
    if (r.value.trim()) return r.value;
  }
}
for (;;) {
  const msg = JSON.parse(await nextLine());
  if (msg.cmd === 'reset') out(await resetAll(msg.seed || 1));
  else if (msg.cmd === 'step') out(await stepAll(msg.actions));
  else if (msg.cmd === 'record') out(record(msg));
  else if (msg.cmd === 'weights') WEIGHTS = { main: msg.main || null, second: msg.second || null };
  else if (msg.cmd === 'close') { out({ closed: true }); break; }
}
process.exit(0);
