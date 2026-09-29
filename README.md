<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/wordmark_dark.png">
  <img alt="QuantForge" src="images/wordmark_light.png" width="420">
</picture>

# QuantForge

Flow-aware, inventory-managed market making — a research harness for algorithmic
trading strategy, execution, and the statistics that decide whether any of it
works.

The thesis of the project is narrow and, deliberately, unfashionable: **most
backtests that claim an edge are measuring their own noise.** So the harness is
built around three things a strategy demo usually skips — common random numbers,
exact PnL decompositions, and self-reported statistical power. When a result
does not survive that, the report says so instead of printing a large t.

## What this is not

The market is **synthetic**: latent drift, informed plus benign client flow,
intensity-based fills, no queue position, no exchange latency, no real
adverse-selection dynamics. It is a mechanics lab for testing whether a
mechanism behaves as claimed. It is **not** evidence of live profitability, and
the harness prints that caveat itself — see [Known limitations](#known-limitations).

## Install

Python 3.11, three pinned dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

The entry point is `flow_mm.py`; the project and `--help` prog name are `quantforge`.

```bash
python flow_mm.py backtest                  # ablation ladder, 30 paired sessions
python flow_mm.py backtest --days 10 --fee 0.003 --plot equity.png --audit audit_log.jsonl
python flow_mm.py exec                      # TWAP/VWAP/IS schedule study, 3000 common paths
python flow_mm.py sweep --days 10           # parameter sensitivity -> sweep.csv
python flow_mm.py all                       # all three, in order
```

Useful flags: `--workers N` (0 = auto, `cpu_count - 1`), `--alpha-std`,
`--informed-rate` to override market calibration from the shell, `--exec-n` for
path count, `--out` for the sweep CSV.

`backtest` is the cheapest way in — a 1-day run takes about 8 seconds and a
3-day run about 20 (seconds of wall clock, five rungs per session), and prints
the full ladder with paired t-statistics.

## The five layers

| Layer | Mechanism |
| --- | --- |
| Signals | Order-book imbalance + futures/ETF lead → online RLS with forgetting, predicting the horizon-matched (30 s) forward return. VPIN-style flow toxicity read from an *exogenous* tape. |
| Quoting | Avellaneda–Stoikov reservation price and spread from inventory, variance and arrival intensity; client-tiered quote widths. |
| Inventory | Net delta hedged in a liquid proxy, with cost and cooldown awareness. |
| Execution | TWAP / VWAP / adaptive implementation-shortfall schedules plus an epsilon-greedy venue router. |
| Risk | Pre-trade position and notional limits, fat-finger band, token-bucket message governor, stale-feed quote pull, spread-widening throttle, drawdown kill switch, JSONL audit trail. |

### The ablation ladder

`backtest` never reports a single number. It reports a ladder in which each rung
adds exactly one mechanism to the rung below it, all rungs sharing the same
synthetic market object per seed, so every comparison is paired:

| Rung | What it turns on |
| --- | --- |
| `naive 1-tick` | fixed half-spread. The floor — a market maker that quotes and nothing else. |
| `+ A-S skew/spread` | Avellaneda–Stoikov reservation price and spread. |
| `+ alpha signal` | RLS forecast shifts the reservation price. |
| `+ tiering & toxicity` | informed/benign quote tiers, VPIN widening above a floor. |
| `+ hedging (full)` | net-delta hedge in the futures proxy. |

Each rung's PnL is an exact additive decomposition, asserted in the tests:

```
pnl = spread + inv_drift + hedge_pnl - fees - hedge_cost - flatten
```

The report prints both a **vs base** contrast (this layer against the whole
stack) and a **vs prev** contrast (this layer against the rung directly beneath
it). The second is the one that answers "did this layer add anything" — and it
is where hedging routinely turns out to be a PnL *loss*. The report says so, and
prints the paired drawdown t-statistic alongside, because a significant
regression is a risk/PnL trade rather than a defect to hide.

## Execution study

`exec` compares client-order schedules — TWAP, VWAP, and an adaptive IS schedule
that scales front-loaded urgency by a volume *forecast* — with and without a
venue router over lit / dark / internal.

Three properties make the numbers mean something:

- **Common random numbers.** The exogenous price path comes from its own
  generator, so every algo trades an identical market. With ~157 bps of path
  noise against 1–3 bps of algo effect, unpaired comparisons cannot resolve
  anything; paired ones can.
- **No look-ahead.** The adaptive schedule sizes on `vol_fc`, a noisy lognormal
  view of realised volume drawn from a different stream. `vol_real` is used only
  for cost accounting. A test asserts that a forecast leak would be detectable.
- **Honest decomposition.** Every cost is attributed against a fixed arrival
  price so the buckets always sum: `total = drift + impact + fee`.

The run reports its own power. If the sample is too small to resolve the
contrast it is studying, it prints the required `n` and labels itself
**UNDERPOWERED** rather than presenting a null as a result.

## Sensitivity

`sweep` varies the market's calibration knobs one at a time — `alpha_std`,
`informed_horizon`, `informed_rate`, `A`, `intraday_k`, the per-share fee, and
the toxicity response — and writes a CSV with the mean PnL of every rung plus
paired contrasts at each grid point. It also runs a separate
`alpha_std × informed_horizon` interaction block, which is the cross a
one-at-a-time grid structurally cannot see.

The `alpha_std = 0` row is the honest anchor. With no exploitable drift, anything
the full stack still earns is spread capture, not prediction, and does not
depend on the forecast being right.

## Statistics

`stats.py` implements paired inference in pure Python — no SciPy — because the
ladder's informative quantity is the per-session *difference*, not any single
config's level. The same seeds and market paths mean the shared variation
cancels.

- `paired(x, y)` — mean, sd, sem, 95% CI, win rate and t on `y - x`, with
  Student-t critical values to df = 30 and a normal limit beyond.
- `summarize(x)` — the same band for a standalone series.
- Power is computed, never assumed.

## Outputs

| File | Contents |
| --- | --- |
| `sweep.csv` | one row per grid point: PnL per rung, paired deltas/t/win/CI, fills, informed share, drawdown, PnL/DD. |
| `audit_log.jsonl` | append-only event trail for the top rung on seed 0 — fills, hedges, risk rejects (first, 100th, 1000th), throttle transitions, kill switch. Each record carries a real wall clock and a sim-time field. |
| `equity.png` | mean cumulative PnL path per rung. |

A worked example of the latter two is committed under `baseline/`.

## Tests

```bash
python -m pytest -q      # 71 tests, ~65 s
```

Coverage is behavioural rather than incidental: RLS against a closed-form least
squares solution and across a regime shift; the running-sum VPIN against a naive
one; market fill intensity against the intensity formula; the PnL identity;
common-random-number and no-look-ahead properties of the execution study;
router non-degeneracy; a fast path proven equivalent to its general path; and
parallel-vs-serial backtest agreement in both chunking regimes.

## Project layout

```
flow_mm.py      market, signals, quoting, hedging, risk, backtest, CLI  (entry point)
exec_algos.py   execution schedules + venue router
analysis.py     sensitivity sweep and the interaction block
stats.py        paired statistics, no scipy
tests/          pytest suite
```

## Known limitations

The code refers back to this section; these are not hedges, they are the
boundary of what the harness measures.

1. **The market is synthetic.** Fills follow a Poisson intensity model. There is
   no queue position, no exchange latency, no order-book replenishment, and no
   genuine adverse-selection dynamics. Anything about queue effects or
   fill-probability realism is out of scope.
2. **The sim is generous.** The reported breakeven fee lands orders of magnitude
   above the real ~0.3 bps exchange fee. The harness prints that caveat itself:
   a large breakeven means the environment is easy, not that the strategy is
   good.
3. **The sweep is one-at-a-time, not factorial.** A full factorial over these
   factors costs orders of magnitude more than one backtest and does not fit in a
   terminal. Only the `alpha_std × informed_horizon` interaction is measured
   directly; other cross terms are unmeasured, not absent.
4. **The exec router rows are degenerate by construction.** `Router.SHAPE`
   declares dark cheaper than lit, so the bandit never faces a real decision and
   its enormous t is a statement about that fee table, not a discovered edge. The
   meaningful exec rows are the lit-only schedule comparisons.
5. **The exec demo is deliberately underpowered.** Default `n = 3000` against a
   contrast that needs an order of magnitude more. It reports the shortfall
   instead of hiding it.
6. **Hedging trades PnL for drawdown.** The top rung is usually PnL-negative
   against the rung below. It is a risk control, and a poor one unless drawdown
   matters more than PnL.
7. **Only the calibrated regime measures anything.** `alpha_std` is set so ~20%
   of 30-second return variance is predictable. Outside that regime the signal
   and adverse-selection layers are unmeasurable and the ladder reports noise.
8. **Paired t-statistics are within-path.** Seeds are shared across rungs, so the
   tests describe this synthetic market, not a population of real ones. No
   parameter uncertainty is propagated into the intervals.
9. **`adverse` is a memo line, not a PnL term.** It is the 30-second markout of
   fills, a diagnostic sub-view of `inv_drift`, and is deliberately excluded from
   the additive decomposition to avoid double counting.

## License

MIT — see [LICENSE](LICENSE).
