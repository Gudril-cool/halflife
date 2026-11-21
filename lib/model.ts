// HalfLife — an expected-value token launch radar.
//
// The name is the model. A pre-token protocol decays toward a launch the way an
// unstable isotope decays toward a stable state: you cannot say when, but you
// can measure how much it resembles things that already decayed. HalfLife fits
// that resemblance on the snapshot itself, then prices it honestly:
//
//   Part A — P(tokenized). A logistic regression trained on ALL 758 protocols.
//     The label is 1 when a protocol records a token (mcap or symbol present),
//     0 when it records none. Features, all from the snapshot's own fields:
//       x1 = log10(tvl_usd)
//       x2 = log10(age_days + 30)          (age unknown -> 365 days)
//       x3 = log10(max(1, chain count))
//       x4 = the protocol category's tokenized base rate (share of that
//            category's protocols that record a token)
//     x1..x3 are standardised with the snapshot's own mean and spread, then a
//     fixed-schedule full-batch gradient descent fits the weights: 900 passes,
//     learning rate 0.12, a small L2 pull of 0.002 on the weights only. No
//     randomness anywhere, so two runs fit the identical model.
//     P is a resemblance score — how much a pre-token protocol looks like the
//     protocols that already record tokens. It is NOT a promised probability
//     of an airdrop or a launch date, and the dashboard says exactly that.
//
//   Part B — value if it launches like its peers. Each category's median
//     FDV/TVL multiple among its tokenized peers (mcap / TVL), winsorised at
//     the global 90th percentile, then shrunk toward the global median by
//     peer count: weight n / (n + 4). Conditional value is that shrunk
//     multiple times the protocol's own TVL.
//
//   Blend — implied = round(conditional x momentum x (0.25 + 0.75 x P)).
//     The 0.25 floor is the residual option value of a protocol that never
//     launches a token of its own: its positions, points and governance weight
//     are not worth zero. Momentum is bounded: the 7-day TVL change, clamped
//     to [-50, +100] percent, moves the value by at most -11% / +22%.
//
// The board ranks every rankable protocol that records no token: all 40
// watchlist slugs plus the earlier finds below the floor. Products OF live
// tokens are set aside, never ranked: wrapped, staked or bridged variants,
// CEX and bridge categories, exchange-prefixed names and versioned pools.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export const MODEL_VERSION = 'halflife-1';
export const GD_ITERATIONS = 900;
export const GD_LEARNING_RATE = 0.12;
export const GD_L2 = 0.002;
export const RESIDUAL_FLOOR = 0.25;
export const UNKNOWN_AGE_DAYS = 365;
export const LIVE_ROWS = 24;

const DAY_MS = 86_400_000;

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Records a token when DefiLlama's record carries a market cap or a symbol. */
export function recordsToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null || p.symbol !== null;
}

/** Records no token at all: no market cap, no CoinGecko id, no symbol. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
const EXCHANGE_PREFIX = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];
const PRODUCT_CATEGORIES = new Set(['CEX', 'Bridge', 'Canonical Bridge', 'Cross Chain Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC']);

/** Rankable on the radar board: pre-token and not a product of a live token. */
export function isRankable(p: SnapshotProtocol): boolean {
  if (!isPreToken(p)) return false;
  if (p.category !== null && PRODUCT_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name) || WRAPPED_WORD.test(p.name)) return false;
  if (EXCHANGE_PREFIX.some((x) => p.name.startsWith(x))) return false;
  return true;
}

// ---------------------------------------------------------------- logistic fit

export interface FeatureStats {
  mean_tvl: number;
  std_tvl: number;
  mean_age: number;
  std_age: number;
  mean_chains: number;
  std_chains: number;
}

export interface LogisticModel {
  bias: number;
  w_tvl: number;
  w_age: number;
  w_chains: number;
  w_catrate: number;
  stats: FeatureStats;
  category_rates: Record<string, number>;
  global_rate: number;
  trained_on: number;
  positives: number;
}

interface RawFeatures {
  f_tvl: number;
  f_age: number;
  f_chains: number;
  cat_rate: number;
}

function rawFeatures(p: SnapshotProtocol, asOf: string, rates: Map<string, number>, globalRate: number): RawFeatures {
  const age = ageDays(asOf, p.listed_at) ?? UNKNOWN_AGE_DAYS;
  return {
    f_tvl: Math.log10(Math.max(1, p.tvl_usd)),
    f_age: Math.log10(age + 30),
    f_chains: Math.log10(Math.max(1, p.chains.length)),
    cat_rate: rates.get(p.category ?? '') ?? globalRate,
  };
}

function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

function meanStd(values: number[]): { mean: number; std: number } {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) * (v - mean), 0) / values.length;
  const std = Math.sqrt(variance);
  return { mean, std: std > 1e-9 ? std : 1 };
}

/** Fit Part A on the whole snapshot with a fixed, deterministic schedule. */
export function fitLogistic(snapshot: Snapshot): LogisticModel {
  const totals = new Map<string, number>();
  const positivesByCat = new Map<string, number>();
  let positives = 0;
  for (const p of snapshot.protocols) {
    const key = p.category ?? '';
    totals.set(key, (totals.get(key) ?? 0) + 1);
    if (recordsToken(p)) {
      positivesByCat.set(key, (positivesByCat.get(key) ?? 0) + 1);
      positives++;
    }
  }
  const globalRate = positives / snapshot.protocols.length;
  const rates = new Map<string, number>();
  for (const [key, n] of totals) rates.set(key, (positivesByCat.get(key) ?? 0) / n);

  const raw = snapshot.protocols.map((p) => rawFeatures(p, snapshot.as_of, rates, globalRate));
  const sTvl = meanStd(raw.map((r) => r.f_tvl));
  const sAge = meanStd(raw.map((r) => r.f_age));
  const sChains = meanStd(raw.map((r) => r.f_chains));
  const stats: FeatureStats = {
    mean_tvl: sTvl.mean, std_tvl: sTvl.std,
    mean_age: sAge.mean, std_age: sAge.std,
    mean_chains: sChains.mean, std_chains: sChains.std,
  };
  const xs = raw.map((r) => [
    (r.f_tvl - stats.mean_tvl) / stats.std_tvl,
    (r.f_age - stats.mean_age) / stats.std_age,
    (r.f_chains - stats.mean_chains) / stats.std_chains,
    r.cat_rate,
  ]);
  const ys = snapshot.protocols.map((p) => (recordsToken(p) ? 1 : 0));

  let w = [0, 0, 0, 0];
  let bias = 0;
  const n = xs.length;
  for (let iter = 0; iter < GD_ITERATIONS; iter++) {
    const grad = [0, 0, 0, 0];
    let gradBias = 0;
    for (let i = 0; i < n; i++) {
      const x = xs[i]!;
      const z = bias + w[0]! * x[0]! + w[1]! * x[1]! + w[2]! * x[2]! + w[3]! * x[3]!;
      const err = sigmoid(z) - ys[i]!;
      for (let k = 0; k < 4; k++) grad[k] = grad[k]! + err * x[k]!;
      gradBias += err;
    }
    for (let k = 0; k < 4; k++) w[k] = w[k]! - GD_LEARNING_RATE * (grad[k]! / n + GD_L2 * w[k]!);
    bias -= GD_LEARNING_RATE * (gradBias / n);
  }

  const categoryRates: Record<string, number> = {};
  for (const key of [...rates.keys()].sort()) categoryRates[key === '' ? '(none)' : key] = round4(rates.get(key)!);
  return {
    bias: round6(bias),
    w_tvl: round6(w[0]!), w_age: round6(w[1]!), w_chains: round6(w[2]!), w_catrate: round6(w[3]!),
    stats: {
      mean_tvl: round6(stats.mean_tvl), std_tvl: round6(stats.std_tvl),
      mean_age: round6(stats.mean_age), std_age: round6(stats.std_age),
      mean_chains: round6(stats.mean_chains), std_chains: round6(stats.std_chains),
    },
    category_rates: categoryRates,
    global_rate: round4(globalRate),
    trained_on: n,
    positives,
  };
}

const round4 = (v: number): number => Math.round(v * 1e4) / 1e4;
const round6 = (v: number): number => Math.round(v * 1e6) / 1e6;

export interface Contribution {
  value: number;
  push: number;
}

export interface RowScore {
  p_launch: number;
  contributions: { tvl: Contribution; age: Contribution; chains: Contribution; cat_rate: Contribution; bias: number };
  conditional_value_usd: number;
  multiple: number;
  momentum_factor: number;
  blend_factor: number;
  implied: number;
}

/** Score one pre-token protocol under a fitted model and the peer multiples. */
export function scoreRow(
  p: SnapshotProtocol,
  asOf: string,
  model: LogisticModel,
  multiples: Map<string, number>,
  globalMultiple: number,
): RowScore {
  const rates = new Map(Object.entries(model.category_rates).map(([k, v]) => [k === '(none)' ? '' : k, v]));
  const raw = rawFeatures(p, asOf, rates, model.global_rate);
  const zTvl = (raw.f_tvl - model.stats.mean_tvl) / model.stats.std_tvl;
  const zAge = (raw.f_age - model.stats.mean_age) / model.stats.std_age;
  const zChains = (raw.f_chains - model.stats.mean_chains) / model.stats.std_chains;
  const pushTvl = model.w_tvl * zTvl;
  const pushAge = model.w_age * zAge;
  const pushChains = model.w_chains * zChains;
  const pushCat = model.w_catrate * raw.cat_rate;
  const pLaunch = sigmoid(model.bias + pushTvl + pushAge + pushChains + pushCat);

  const multiple = multiples.get(p.category ?? '') ?? globalMultiple;
  const conditional = multiple * p.tvl_usd;
  const change = p.change_7d === null ? 0 : Math.min(100, Math.max(-50, p.change_7d));
  const momentum = 1 + (change / 100) * 0.22;
  const blend = RESIDUAL_FLOOR + (1 - RESIDUAL_FLOOR) * pLaunch;
  const implied = Math.max(1, Math.round(conditional * momentum * blend));
  return {
    p_launch: round4(pLaunch),
    contributions: {
      tvl: { value: round4(raw.f_tvl), push: round4(pushTvl) },
      age: { value: round4(raw.f_age), push: round4(pushAge) },
      chains: { value: round4(raw.f_chains), push: round4(pushChains) },
      cat_rate: { value: round4(raw.cat_rate), push: round4(pushCat) },
      bias: model.bias,
    },
    conditional_value_usd: Math.round(conditional),
    multiple: round4(multiple),
    momentum_factor: round4(momentum),
    blend_factor: round4(blend),
    implied,
  };
}

// ---------------------------------------------------------------- peer multiples

export interface CategoryMultiple {
  category: string;
  peers: number;
  median_multiple: number;
  shrunk_multiple: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** Part B's multiples: per-category medians shrunk toward the global median. */
export function peerMultiples(snapshot: Snapshot): { byCategory: Map<string, number>; global: number; table: CategoryMultiple[] } {
  // Winsorise first (revision after the first build): one peer with a
  // 400x FDV/TVL ratio — a prediction market whose DefiLlama TVL is a
  // fraction of its real float — dragged its whole category's shrunk
  // multiple to 80x. Every peer ratio is capped at the global 90th
  // percentile before any median is taken, so a single broken record can
  // bend a category but never own it.
  const rawRatios: { key: string; ratio: number }[] = [];
  for (const p of snapshot.protocols) {
    if (p.mcap_usd === null || p.tvl_usd <= 0) continue;
    rawRatios.push({ key: p.category ?? '', ratio: p.mcap_usd / p.tvl_usd });
  }
  const sortedRatios = rawRatios.map((r) => r.ratio).sort((a, b) => a - b);
  const cap = sortedRatios[Math.floor(0.9 * (sortedRatios.length - 1))] ?? Infinity;
  const ratiosByCat = new Map<string, number[]>();
  const all: number[] = [];
  for (const { key, ratio } of rawRatios) {
    const capped = Math.min(ratio, cap);
    all.push(capped);
    if (!ratiosByCat.has(key)) ratiosByCat.set(key, []);
    ratiosByCat.get(key)!.push(capped);
  }
  const global = median(all);
  const byCategory = new Map<string, number>();
  const table: CategoryMultiple[] = [];
  for (const key of [...ratiosByCat.keys()].sort()) {
    const ratios = ratiosByCat.get(key)!;
    const med = median(ratios);
    const weight = ratios.length / (ratios.length + 4);
    const shrunk = weight * med + (1 - weight) * global;
    byCategory.set(key, shrunk);
    table.push({
      category: key === '' ? '(none)' : key,
      peers: ratios.length,
      median_multiple: round4(med),
      shrunk_multiple: round4(shrunk),
    });
  }
  return { byCategory, global, table };
}

// ---------------------------------------------------------------- farm plan

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 300;
  if (p.tvl_usd >= 1_000_000_000) return 200;
  if (p.tvl_usd >= 100_000_000) return 120;
  return 60;
}

function playbook(p: SnapshotProtocol, capital: number, chain: string): { action: string; verb: string } {
  const c = (p.category ?? '').toLowerCase();
  const n = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return { action: `supply $${capital} of USDC as collateral to ${n} on ${chain} and borrow a small amount against it`, verb: 'supply or repay' };
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm') || c.includes('farm')) return { action: `add $${capital} of liquidity to ${n}'s deepest pool on ${chain} and keep the LP position open`, verb: 'swap or rebalance' };
  if (c.includes('staking') || c.includes('restak')) return { action: `stake $${capital} of the native asset through ${n} on ${chain} and hold the staking receipt`, verb: 'stake top-up' };
  if (c.includes('derivative') || c.includes('perpetual')) return { action: `post $${capital} of USDC margin on ${n} and hold one small hedged position`, verb: 'hedge roll' };
  if (c.includes('curator') || c.includes('allocator') || c.includes('risk')) return { action: `deposit $${capital} into a ${n} curated vault on ${chain} and keep the vault shares`, verb: 'vault deposit' };
  if (c.includes('stable')) return { action: `mint or deposit $${capital} of the stable asset through ${n} on ${chain} and hold it in the farming wallet`, verb: 'mint or redeem check' };
  if (c.includes('rwa')) return { action: `mint $${capital} of ${n}'s yield asset on ${chain} and hold it to maturity of the farm window`, verb: 'holdings check' };
  if (c.includes('launchpad')) return { action: `register the wallet on ${n}, deposit $${capital} of USDC on ${chain} and join one live sale or waitlist`, verb: 'sale registration' };
  if (c.includes('insurance')) return { action: `buy $${capital} of cover from ${n} on ${chain} and keep the policy active`, verb: 'cover renewal check' };
  if (c.includes('prediction')) return { action: `fund $${capital} on ${n} and hold one position in a liquid market`, verb: 'market position' };
  return { action: `deposit $${capital} of USDC into ${n} on ${chain} and keep the position open`, verb: 'deposit' };
}

/**
 * Seven milestone steps per row. The voice is the calendar, not a checklist:
 * first month, while waiting, monthly, if points appear, pre-launch, launch.
 * Every step names its capital, gas and time so a reader can budget the farm.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): string[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const gas = capital >= 200 ? 8 : 4;
  const { action, verb } = playbook(p, capital, chain);
  const window = age === null ? 'a full quarter' : age < 180 ? 'at least 60 days' : age < 365 ? 'at least 45 days' : 'at least 30 days';
  const steps = [
    `First month: open a fresh wallet for ${p.name}, fund it with $${capital} plus gas on ${chain}, and keep the seed phrase offline. Cost: $${capital} capital, about $3 gas. Time: 20 minutes in week one.`,
    `First month: ${action}. Cost: $${capital} stays in the position, about $${gas} gas. Time: 15 minutes, before day 30.`,
    `While waiting: make one qualifying ${verb} each week on ${chain} so the wallet ages with steady history over ${window}, not a burst. Cost: about $2 gas a week. Time: 10 minutes a week.`,
    `Monthly: on the first of the month re-check ${p.name}'s TVL, docs and rewards page; add $25 only if TVL held or grew that month. Cost: $0 to check, $25 optional. Time: 15 minutes a month.`,
    `If a points program appears: join in its first week, screenshot your balance, and log the date — the first recorded cohort is usually the one a launch snapshot rewards. Cost: under $2 gas. Time: 10 minutes once.`,
    `Pre-launch: if ${p.name} announces a launch window, raise activity to twice a week and keep the full position open through any snapshot date. Cost: about $4 extra gas. Time: 20 minutes a week for two weeks.`,
    `At launch: claim only from the link ${p.name} publishes itself, never pay a claim fee, and write the claim transaction in your farm log. Cost: claim gas only. Time: 30 minutes on launch day.`,
  ];
  return steps.map((s) => (s.length > 600 ? `${s.slice(0, 596)}...` : s));
}

// ---------------------------------------------------------------- board

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  p_launch: number;
  conditional_value_usd: number;
  multiple: number;
  momentum_factor: number;
  blend_factor: number;
  farm_steps: string[];
}

export interface RadarRow extends BoardRow {
  contributions?: RowScore['contributions'];
}

export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  model: LogisticModel;
  multiples: CategoryMultiple[];
  global_multiple: number;
  residual_floor: number;
  watchlist_size: number;
  watchlist: string[];
  board_size: number;
  early_size: number;
  live_size: number;
  rows: RadarRow[];
  live: LiveRow[];
}

export function buildBoard(snapshot: Snapshot): { board: BoardRow[]; model: LogisticModel; multiples: CategoryMultiple[]; globalMultiple: number } {
  const model = fitLogistic(snapshot);
  const { byCategory, global, table } = peerMultiples(snapshot);
  const chosen = snapshot.protocols.filter(isRankable);
  const scored = chosen.map((p) => ({ p, score: scoreRow(p, snapshot.as_of, model, byCategory, global), age: ageDays(snapshot.as_of, p.listed_at) }));
  scored.sort((a, b) => b.score.implied - a.score.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token' as const,
    age_days: s.age,
    implied_value_usd: s.score.implied,
    p_launch: s.score.p_launch,
    conditional_value_usd: s.score.conditional_value_usd,
    multiple: s.score.multiple,
    momentum_factor: s.score.momentum_factor,
    blend_factor: s.score.blend_factor,
    farm_steps: farmSteps(s.p, s.age),
  }));
  // Contributions ride in data.js only (the dashboard detail view), not board.json.
  void model;
  return { board, model, multiples: table, globalMultiple: global };
}

export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const model = fitLogistic(snapshot);
  const { byCategory, global, table } = peerMultiples(snapshot);
  const chosen = snapshot.protocols.filter(isRankable);
  const scored = chosen.map((p) => ({ p, score: scoreRow(p, snapshot.as_of, model, byCategory, global), age: ageDays(snapshot.as_of, p.listed_at) }));
  scored.sort((a, b) => b.score.implied - a.score.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token' as const,
    age_days: s.age,
    implied_value_usd: s.score.implied,
    p_launch: s.score.p_launch,
    conditional_value_usd: s.score.conditional_value_usd,
    multiple: s.score.multiple,
    momentum_factor: s.score.momentum_factor,
    blend_factor: s.score.blend_factor,
    farm_steps: farmSteps(s.p, s.age),
  }));
  const rows: RadarRow[] = scored.map((s, i) => ({
    ...board[i]!,
    // The detail view's feature pushes: attached here so the page shows the
    // fitted model's own arithmetic for this row, not a re-derivation.
    contributions: s.score.contributions,
  } as RadarRow));
  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    model,
    multiples: table,
    global_multiple: round4(global),
    residual_floor: RESIDUAL_FLOOR,
    watchlist_size: snapshot.watchlist.length,
    watchlist: snapshot.watchlist,
    board_size: board.length,
    early_size: board.filter((r) => !snapshot.watchlist.includes(r.slug)).length,
    live_size: live.length,
    rows,
    live,
  };
  return { board, radar };
}

export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  if (board.length < 100) return `the board holds ${board.length} rows, under the 100-row floor`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return 'all rows share one implied value';
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600) return `${r.slug} has a farm step of ${t.length} characters`;
    }
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

export function dataScript(radar: RadarData): string {
  return `window.HALFLIFE=${JSON.stringify(radar)};\n`;
}
