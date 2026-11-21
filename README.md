# halflife

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/Gudril-cool/halflife/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/Gudril-cool/halflife/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

HalfLife — an expected-value token launch radar. It fits a logistic regression on all 758 protocols in the snapshot (label: records a token vs records none; features: log TVL, log age, log chain count, and the category's tokenized base rate; deterministic gradient descent, 900 fixed passes) to give every pre-token protocol a P(launch) resemblance score, then prices it at its category's shrunk median FDV/TVL peer multiple times TVL, times a bounded 7-day momentum factor, times a blend of (0.25 + 0.75 × P) — the 0.25 floor is the residual option value of a protocol that never launches a token of its own. P is resemblance to tokenized protocols, not a promised probability of an airdrop. The board ranks 197 protocols — the full 40-protocol watchlist plus 157 earlier finds — sets wrapped, staked and bridged products of live tokens aside instead of ranking them, and gives every row a seven-step milestone farm plan keyed to its expected window, each step with capital, gas and time costs. The dashboard is fully offline, in an amber decay theme: its board shows P(launch) beside implied value, and its detail view shows the fitted model's feature contributions for that row, Part B, and the blend.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input /path/to/snapshot.json
```

Node 22.18+, standard library only. The program is deterministic: two runs on the same snapshot write byte-identical output. `node validate.mjs out /path/to/snapshot.json` re-checks the output against the competition's seven gate rules locally.

---

<sub>Entered on [make.fun](https://make.fun/?ref=x6sa47yh&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
