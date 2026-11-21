// Local replica of the token-radar-01 seven-row gate, for HalfLife.
// Usage: node validate.mjs <output-dir> <snapshot.json>
// Checks, in the gate's own order: dashboard_files, board_file,
// sorted_by_value, coverage, farm_steps, screenshots, dossier_matches.
// The dossier check rebuilds the dossier from the output with the vendored
// library and compares bytes — exactly what the scorer does.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { buildDossier } from './vendor/makefun-board.ts';

const out = process.argv[2] ?? 'out';
const snapFile = process.argv[3];
if (!snapFile) { console.error('usage: node validate.mjs <output-dir> <snapshot.json>'); process.exit(2); }
const snap = JSON.parse(readFileSync(snapFile, 'utf8'));
const results = [];
const check = (name, ok, detail) => results.push({ name, ok, detail });

// 1 dashboard_files
let dashOk = false, dashDetail = '';
try {
  const dir = join(out, 'dashboard');
  const files = readdirSync(dir, { recursive: true }).filter((f) => statSync(join(dir, f)).isFile());
  const total = files.reduce((s, f) => s + statSync(join(dir, f)).size, 0);
  const hasHtml = files.some((f) => f.endsWith('index.html'));
  const hasCss = files.some((f) => f.endsWith('.css'));
  const hasJs = files.some((f) => f.endsWith('.js'));
  let network = '';
  for (const f of files) {
    const t = readFileSync(join(dir, f), 'utf8');
    if (/\b(?:href|src)\s*=\s*["']https?:\/\//i.test(t) || /url\(\s*['"]?https?:\/\//i.test(t) || /(?:import|require|from)\s*\(?\s*['"]https?:\/\//.test(t)) network = f;
  }
  dashOk = hasHtml && hasCss && hasJs && files.length <= 400 && total <= 4 * 1024 * 1024 && !network;
  dashDetail = `${files.length} files, ${total} bytes, html=${hasHtml} css=${hasCss} js=${hasJs}${network ? ` network in ${network}` : ''}, source total with data note`;
  // dossier source cap is about dashboard/ only
  check('dashboard_files', dashOk, dashDetail + `; dashboard bytes=${total} (<=65536 for full dossier text: ${total <= 65536})`);
} catch (e) { check('dashboard_files', false, String(e.message ?? e)); }

// board
let board = null;
try { board = JSON.parse(readFileSync(join(out, 'board.json'), 'utf8')); } catch {}
const rows = board ? (Array.isArray(board) ? board : board.rows) : null;

// 2 board_file
if (!rows) check('board_file', false, 'board.json missing or unparsable');
else {
  const slugs = new Set(snap.protocols.map((p) => p.slug));
  const bad = rows.filter((r) => !slugs.has(r.slug));
  const size = statSync(join(out, 'board.json')).size;
  const ok = rows.length >= 100 && rows.length <= 4000 && size <= 8 * 1024 * 1024 && bad.length === 0 &&
    rows.every((r) => typeof r.name === 'string' && r.name.length > 0 && Number.isSafeInteger(r.rank) && r.rank > 0 &&
      Number.isSafeInteger(r.implied_value_usd) && r.implied_value_usd > 0 && r.implied_value_usd <= 1e14 && Array.isArray(r.farm_steps));
  check('board_file', ok, `${rows.length} rows, ${size} bytes, invented slugs=${bad.length}`);
}

// 3 sorted_by_value
if (!rows) check('sorted_by_value', false, 'no board');
else {
  const ranks = rows.map((r) => r.rank).sort((a, b) => a - b);
  const ranksOk = ranks.every((v, i) => v === i + 1);
  const byRank = [...rows].sort((a, b) => a.rank - b.rank);
  const orderOk = byRank.every((r, i) => i === 0 || r.implied_value_usd <= byRank[i - 1].implied_value_usd);
  const allEqual = rows.length >= 2 && rows[0].implied_value_usd === rows[rows.length - 1].implied_value_usd;
  check('sorted_by_value', ranksOk && orderOk && !allEqual, `ranksOk=${ranksOk} orderOk=${orderOk} allEqual=${allEqual}`);
}

// 4 coverage
if (!rows) check('coverage', false, 'no board');
else {
  const have = new Set(rows.map((r) => r.slug));
  const missing = snap.watchlist.filter((s) => !have.has(s));
  const outside = rows.filter((r) => !snap.watchlist.includes(r.slug)).length;
  check('coverage', missing.length === 0 && outside >= 60, `missing=${missing.length}, outside watchlist=${outside}`);
}

// 5 farm_steps
if (!rows) check('farm_steps', false, 'no board');
else {
  let bad = 0, first = '';
  for (const r of rows) {
    if (!Array.isArray(r.farm_steps) || r.farm_steps.length < 3 || r.farm_steps.length > 12) { bad++; if (!first) first = r.slug; continue; }
    for (const s of r.farm_steps) { const t = String(s).trim(); if (t.length < 20 || t.length > 600) { bad++; if (!first) first = `${r.slug}:${t.length}`; break; } }
  }
  check('farm_steps', bad === 0, bad === 0 ? `all ${rows.length} rows within 3-12 x 20-600` : `bad rows=${bad}, first=${first}`);
}

// image dimensions: PNG IHDR and JPEG SOF parsers, no dependencies
function imageSize(path) {
  const b = readFileSync(path);
  if (b.length > 8 && b[0] === 0x89 && b[1] === 0x50) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), type: 'png' };
  if (b.length > 2 && b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) { i++; continue; }
      const marker = b[i + 1];
      if (marker >= 0xc0 && marker <= 0xc3) return { w: b.readUInt16BE(i + 7), h: b.readUInt16BE(i + 5), type: 'jpeg' };
      const len = b.readUInt16BE(i + 2);
      if (len < 2) break;
      i += 2 + len;
    }
  }
  return null;
}

// 6 screenshots
try {
  const dir = join(out, 'screenshots');
  const files = readdirSync(dir).filter((f) => !f.startsWith('.'));
  let ok = files.length >= 3 && files.length <= 4;
  const details = [];
  for (const f of files) {
    const size = statSync(join(dir, f)).size;
    const dim = imageSize(join(dir, f));
    const long = dim ? Math.max(dim.w, dim.h) : 0;
    const short = dim ? Math.min(dim.w, dim.h) : 0;
    const good = dim && size <= 524288 && long <= 1440 && short >= 800;
    if (!good) ok = false;
    details.push(`${f}:${size}B ${dim ? dim.w + 'x' + dim.h : '?'}`);
  }
  check('screenshots', ok, details.join(', '));
} catch (e) { check('screenshots', false, String(e.message ?? e)); }

// 7 dossier_matches
try {
  const expected = Buffer.from(buildDossier(out, { text_max_bytes: 65536, screenshots: { max: 4, max_bytes: 524288 }, dashboard: { max_files: 400, max_bytes: 4194304 } }));
  const actual = readFileSync(join(out, 'dossier.json'));
  check('dossier_matches', expected.equals(actual) && actual.length <= 1.5 * 1024 * 1024, `dossier ${actual.length} bytes, identical=${expected.equals(actual)}`);
} catch (e) { check('dossier_matches', false, String(e.message ?? e)); }

let pass = 0;
for (const r of results) { console.log(`${r.ok ? 'PASS' : 'FAIL'} ${r.name} — ${r.detail}`); if (r.ok) pass++; }
console.log(`${pass}/7 gate checks pass`);
process.exit(pass === 7 ? 0 : 1);
