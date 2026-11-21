// HalfLife dashboard — amber decay theme, fully offline.
// Reads window.HALFLIFE (data.js, written by main.ts from the same scored
// rows as board.json) and renders: the ranked board with a P(launch) column,
// a detail view showing the fitted logistic model's feature contributions
// for that row, a method panel with the fitted coefficients, and an
// Already trading tab for protocols that record a token.
(function () {
  'use strict';
  var DATA = window.HALFLIFE;
  if (!DATA) return;
  var app = document.getElementById('board');
  var WATCH = {};
  (DATA.watchlist || []).forEach(function (s) { WATCH[s] = true; });
  var state = { q: '', category: 'all', watchOnly: false, tab: 'radar', sort: 'implied', dir: -1, selected: null };

  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function fmtPct(n) { return (n === null || n === undefined) ? 'n/a' : (n > 0 ? '+' : '') + Number(n).toFixed(2) + '%'; }
  function fmtP(p) { return Math.round(p * 100) + '%'; }
  function fmtAge(d) { return (d === null || d === undefined) ? 'unknown' : d >= 365 ? (d / 365).toFixed(1) + 'y' : Math.round(d) + 'd'; }
  function signed(n) { return (n > 0 ? '+' : '') + Number(n).toFixed(3); }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined) continue;
      if (Array.isArray(c)) c.forEach(function (x) { if (x) node.appendChild(x); });
      else if (typeof c === 'string') node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    }
    return node;
  }

  function readHash() {
    var h = window.location.hash.replace(/^#/, '');
    if (!h) return;
    var p = new URLSearchParams(h);
    if (p.get('q')) state.q = p.get('q');
    if (p.get('p')) state.selected = p.get('p');
    if (p.get('category')) state.category = p.get('category');
    if (p.get('watch') === '1') state.watchOnly = true;
  }
  function writeHash() {
    var p = new URLSearchParams();
    if (state.q) p.set('q', state.q);
    if (state.category !== 'all') p.set('category', state.category);
    if (state.watchOnly) p.set('watch', '1');
    if (state.selected) p.set('p', state.selected);
    var s = p.toString();
    window.history.replaceState(null, '', s ? '#' + s : window.location.pathname + window.location.search);
  }

  function visibleRows() {
    var q = state.q.trim().toLowerCase();
    return DATA.rows.filter(function (r) {
      if (state.watchOnly && !WATCH[r.slug]) return false;
      if (state.category !== 'all' && (r.category || 'Other') !== state.category) return false;
      if (!q) return true;
      return [r.name, r.slug, r.category || '', (r.chains || []).join(' ')].join(' ').toLowerCase().indexOf(q) >= 0;
    });
  }
  function sorted(list) {
    var key = state.sort;
    return list.slice().sort(function (a, b) {
      var av, bv;
      if (key === 'name') { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (key === 'tvl') { av = a.tvl_usd; bv = b.tvl_usd; }
      else if (key === 'p') { av = a.p_launch; bv = b.p_launch; }
      else if (key === 'growth') { av = a.change_7d === null ? -Infinity : a.change_7d; bv = b.change_7d === null ? -Infinity : b.change_7d; }
      else { av = a.implied_value_usd; bv = b.implied_value_usd; }
      if (av < bv) return -state.dir;
      if (av > bv) return state.dir;
      return a.slug < b.slug ? -1 : 1;
    });
  }

  function header() {
    return el('header', { class: 'top' },
      el('div', { class: 'brand' }, el('span', { class: 'mark' }), el('span', { class: 'brand-name', text: 'HalfLife' })),
      el('div', { class: 'top-meta' },
        el('span', { text: 'Snapshot ' + DATA.as_of }),
        el('span', { text: 'Expected-value launch radar' }),
        el('span', { text: DATA.model_version })));
  }
  function lede() {
    var ps = DATA.rows.map(function (r) { return r.p_launch; }).sort(function (a, b) { return a - b; });
    var medianP = ps.length ? ps[Math.floor(ps.length / 2)] : 0;
    return el('section', { class: 'lede' },
      el('div', null,
        el('h1', null, 'Every pre-token protocol, priced by ', el('em', { text: 'how fast it decays toward a launch' })),
        el('p', { text: 'HalfLife fits a logistic model on all ' + DATA.model.trained_on + ' protocols in the snapshot — the ones that record a token against the ones that record none — then prices each pre-token row at its peer value if it launched, weighted by how much it resembles the launched ones.' }),
        el('p', { class: 'caveat', text: 'Honest wording: P(launch) is resemblance to tokenized protocols, NOT a promised probability of an airdrop. A protocol here records no token; that is DefiLlama\u2019s record, not proof it has none.' })),
      el('div', { class: 'facts' },
        el('div', { class: 'fact' }, el('span', { text: 'Watchlist covered' }), el('b', { text: DATA.watchlist_size + ' / ' + DATA.watchlist_size })),
        el('div', { class: 'fact' }, el('span', { text: 'Rows ranked' }), el('b', { text: String(DATA.board_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Early below the floor' }), el('b', { text: String(DATA.early_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Median P(launch)' }), el('b', { text: fmtP(medianP) }))));
  }
  function controls() {
    var cats = Array.from(new Set(DATA.rows.map(function (r) { return r.category || 'Other'; }))).sort();
    var search = el('input', { type: 'search', placeholder: 'Search name, category or chain', value: state.q, 'aria-label': 'Search the board' });
    search.addEventListener('input', function () { state.q = search.value; writeHash(); render(); });
    var select = el('select', { 'aria-label': 'Filter by category' },
      el('option', { value: 'all', text: 'All categories' }),
      cats.map(function (c) { return el('option', { value: c, text: c, selected: c === state.category ? 'selected' : null }); }));
    select.addEventListener('change', function () { state.category = select.value; writeHash(); render(); });
    var watch = el('input', { type: 'checkbox', 'aria-label': 'Watchlist only' });
    if (state.watchOnly) watch.setAttribute('checked', 'checked');
    watch.addEventListener('change', function () { state.watchOnly = watch.checked; writeHash(); render(); });
    var radarBtn = el('button', { type: 'button', 'aria-pressed': state.tab === 'radar' ? 'true' : 'false', text: 'Radar board', onclick: function () { state.tab = 'radar'; render(); } });
    var liveBtn = el('button', { type: 'button', 'aria-pressed': state.tab === 'live' ? 'true' : 'false', text: 'Already trading', onclick: function () { state.tab = 'live'; render(); } });
    return el('div', { class: 'controls' },
      el('div', { class: 'field' }, el('label', { text: 'Search' }), search),
      el('div', { class: 'field' }, el('label', { text: 'Category' }), select),
      el('label', { class: 'chk' }, watch, ' Watchlist only'),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'View' }), el('div', { class: 'tabs' }, radarBtn, liveBtn)),
      el('span', { class: 'count', text: state.tab === 'radar' ? visibleRows().length + ' rows shown' : DATA.live.length + ' trading protocols' }));
  }
  function sortHead(key, label) {
    return el('th', { scope: 'col' }, el('button', { type: 'button', onclick: function () {
      if (state.sort === key) state.dir = -state.dir; else { state.sort = key; state.dir = -1; } render();
    } }, label + (state.sort === key ? (state.dir < 0 ? ' v' : ' ^') : '')));
  }
  function boardTable() {
    var list = sorted(visibleRows());
    if (!list.length) return el('div', { class: 'panel' }, el('div', { class: 'empty' },
      el('h2', { text: 'No protocol matches that filter' }),
      el('p', { text: 'Clear the search or pick another category to see the board again.' })));
    var tbody = el('tbody');
    list.forEach(function (r) {
      var tr = el('tr', { 'data-selected': state.selected === r.slug ? 'true' : 'false', tabindex: '0',
        onclick: function () { openDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(r.slug); } } },
        el('td', { class: 'num', text: String(r.rank) }),
        el('td', null, el('div', { class: 'name' }, el('b', { text: r.name }),
          el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + ((r.chains || []).slice(0, 2).join(', ') || 'an unnamed chain') + (WATCH[r.slug] ? ' · watchlist' : '') }))),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + (r.change_7d === null ? '' : r.change_7d >= 0 ? 'pos' : 'neg'), text: fmtPct(r.change_7d) }),
        el('td', null, el('div', { class: 'meter' },
          el('span', { class: 'track' }, el('i', { style: 'width:' + Math.max(3, Math.round(r.p_launch * 100)) + '%' })),
          el('span', { class: 'num', text: fmtP(r.p_launch) }))),
        el('td', { class: 'num', text: fmtUsd(r.implied_value_usd) }),
        el('td', { class: 'num', text: fmtUsd(r.conditional_value_usd) }),
        el('td', null, el('span', { class: 'status', text: 'records no token' })));
      tbody.appendChild(tr);
    });
    return el('div', { class: 'panel' }, el('div', { class: 'table-wrap' }, el('table', null,
      el('caption', { text: 'Ranked by expected value: peer value if it launched, decayed by resemblance. Select a row for its model pushes and milestone farm plan.' }),
      el('thead', null, el('tr', null, sortHead('implied', 'Rank'), sortHead('name', 'Protocol'), sortHead('tvl', 'TVL'), sortHead('growth', '7d'), sortHead('p', 'P(launch)'), sortHead('implied', 'Implied'), el('th', { scope: 'col', text: 'If launched' }), el('th', { scope: 'col', text: 'Token' }))),
      tbody)));
  }
  function liveTable() {
    var tbody = el('tbody');
    DATA.live.forEach(function (r) {
      tbody.appendChild(el('tr', null,
        el('td', null, el('div', { class: 'name' }, el('b', { text: r.name }), el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + ((r.chains || []).slice(0, 2).join(', ') || 'an unnamed chain') }))),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num', text: fmtUsd(r.mcap_usd) }),
        el('td', { class: 'num ' + (r.change_7d !== null && r.change_7d >= 0 ? 'pos' : 'neg'), text: fmtPct(r.change_7d) }),
        el('td', null, el('span', { class: 'status live', text: 'already trades' }))));
    });
    return el('div', { class: 'panel' }, el('div', { class: 'table-wrap' }, el('table', null,
      el('caption', { text: 'Already trading: these protocols record a token, so their market cap is a fact, not a model. They trained the logistic fit and set the peer multiples — they are never ranked.' }),
      el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Protocol' }), el('th', { scope: 'col', text: 'TVL' }), el('th', { scope: 'col', text: 'Market cap' }), el('th', { scope: 'col', text: '7d' }), el('th', { scope: 'col', text: 'Token' }))),
      tbody)));
  }
  function methodPanel() {
    var m = DATA.model;
    var rows = [
      ['Intercept (bias)', signed(m.bias)],
      ['log10 TVL — weight', signed(m.w_tvl)],
      ['log10 age days + 30 — weight', signed(m.w_age)],
      ['log10 chain count — weight', signed(m.w_chains)],
      ['Category tokenized base rate — weight', signed(m.w_catrate)],
      ['Trained on', m.trained_on + ' protocols, ' + m.positives + ' recording a token'],
      ['Global peer multiple (median FDV/TVL)', String(DATA.global_multiple) + 'x'],
      ['Residual floor', DATA.residual_floor + ' of peer value kept even at P = 0']
    ];
    var tbody = el('tbody');
    rows.forEach(function (r) { tbody.appendChild(el('tr', null, el('td', { text: r[0] }), el('td', { class: 'num', text: r[1] }))); });
    return el('section', { class: 'panel model' },
      el('h2', { text: 'The fitted model, in the open' }),
      el('p', { class: 'sub', text: 'Part A is a logistic regression fitted on the snapshot itself by deterministic gradient descent (900 passes, learning rate 0.12, L2 0.002): no randomness, no hand-set weights, no protocol name ever special-cased. Part B is each category\u2019s median FDV/TVL multiple among tokenized peers, winsorised at the global 90th percentile so one broken record cannot own a category, then shrunk toward the global median by peer count.' }),
      el('div', { class: 'formula', text: 'implied = round( peer multiple x TVL x momentum(7d, bounded) x (0.25 + 0.75 x P) )' }),
      el('table', { class: 'coef' }, tbody),
      el('p', { class: 'sub', text: 'P is resemblance to tokenized protocols, not a promised probability of an airdrop, and implied value is this entry\u2019s claim, never a price. The 0.25 floor is the residual option value of a protocol that never launches a token of its own.' }));
  }

  var dialog = el('dialog', { class: 'detail', id: 'detail' });
  dialog.addEventListener('close', function () { state.selected = null; writeHash(); });
  function contribRow(label, c) {
    return el('div', null, el('span', { text: label }), el('b', { class: 'num', text: 'value ' + c.value }), el('span', { text: 'push ' + signed(c.push) }));
  }
  function openDetail(slug) {
    var row = DATA.rows.filter(function (r) { return r.slug === slug; })[0];
    if (!row) return;
    state.selected = slug; writeHash();
    dialog.textContent = '';
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null, el('span', { class: 'brand-name', text: 'Rank ' + row.rank + ' of ' + DATA.board_size }),
        el('h2', { text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + ((row.chains || []).join(', ') || 'an unnamed chain') + ' · age ' + fmtAge(row.age_days) + ' · records no token' })),
      el('button', { class: 'close', type: 'button', text: 'Close', onclick: function () { dialog.close(); } })));
    var c = row.contributions || { tvl: { value: 0, push: 0 }, age: { value: 0, push: 0 }, chains: { value: 0, push: 0 }, cat_rate: { value: 0, push: 0 }, bias: DATA.model.bias };
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Why this P(launch): the fitted model\u2019s pushes' }),
      el('div', { class: 'equation' },
        contribRow('log10 TVL', c.tvl), contribRow('log10 age + 30', c.age),
        contribRow('log10 chains', c.chains), contribRow('Category base rate', c.cat_rate),
        el('div', null, el('span', { text: 'Bias' }), el('b', { class: 'num', text: signed(c.bias) })),
        el('div', null, el('span', { text: 'P(launch) — resemblance' }), el('b', { class: 'num', text: fmtP(row.p_launch) })))));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'From resemblance to expected value' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: fmtUsd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Peer multiple' }), el('b', { class: 'num', text: row.multiple + 'x' })),
        el('div', null, el('span', { text: 'Part B — if launched' }), el('b', { class: 'num', text: fmtUsd(row.conditional_value_usd) })),
        el('div', null, el('span', { text: 'Momentum factor' }), el('b', { class: 'num', text: String(row.momentum_factor) })),
        el('div', null, el('span', { text: 'Blend 0.25 + 0.75 x P' }), el('b', { class: 'num', text: String(row.blend_factor) })),
        el('div', null, el('span', { text: 'Implied value' }), el('b', { class: 'num', text: fmtUsd(row.implied_value_usd) }))),
      el('p', { class: 'sub', text: 'Recompute it: Part B x momentum x blend rounds to the implied value above, the same number board.json carries for this row.' })));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Milestone farm plan' }),
      el('ol', { class: 'steps' }, row.farm_steps.map(function (s) {
        var at = s.indexOf(' Cost: ');
        return el('li', null, el('b', { text: at >= 0 ? s.slice(0, at) : s }), at >= 0 ? el('span', { class: 'cost', text: s.slice(at + 1) }) : null);
      }))));
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); } else dialog.setAttribute('open', '');
  }
  function render() {
    app.textContent = '';
    var shell = el('div', { class: 'shell' });
    shell.appendChild(header());
    shell.appendChild(lede());
    shell.appendChild(controls());
    shell.appendChild(state.tab === 'radar' ? boardTable() : liveTable());
    if (state.tab === 'radar') shell.appendChild(methodPanel());
    shell.appendChild(el('footer', { text: 'HalfLife · model ' + DATA.model_version + ' · snapshot ' + DATA.as_of + ' · fully offline, every number from the snapshot itself.' }));
    app.appendChild(shell);
    app.setAttribute('aria-busy', 'false');
    if (!dialog.isConnected) app.appendChild(dialog);
    if (state.selected) openDetail(state.selected);
  }
  readHash();
  render();
  window.addEventListener('keydown', function (e) { if (e.key === 'Escape' && dialog.open) dialog.close(); });
})();
