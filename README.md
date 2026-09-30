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

To go the other way — from the live market instead of a synthetic one:

```bash
pip install websockets                      # the only added dependency, live source only
python flow_mm.py record --symbol btcusdt --seconds 3600 --capture data/btc.jsonl
```

```python
from replay import capture_to_dataset, capture_is_continuous, sessions, ReplayMarket
from flow_mm import LADDER, MarketParams, backtest

ds = capture_to_dataset("data/btc.jsonl")   # merged capture -> one-second frames
assert capture_is_continuous("data/btc.jsonl")   # one writer, no crash/restart seam
backtest(days=1, markets=[ReplayMarket(ds, mp=MarketParams())], ladder=LADDER)
```

`record` writes the *merged capture* tier `readers` documents but had no producer
for: book snapshots and prints interleaved in one time-ordered JSONL stream, at
native frequency, which is the only tier that supports a real reachability check
on fills. Read it back with `capture_to_dataset`, which derives its own
one-second window from the file. It is a separate mode rather than part of `all`
because it blocks on a socket.

Recorder flags: `--seconds` (duration), `--depth` (5, 10, or 20 — the partial-book
snapshot depth), `--rotate-mb` (roll to a new segment past a size, 0 = never),
`--futures-symbol` to override the perp whose mid fills the `futures` column.
`record` also streams a public futures `bookTicker` alongside spot, because the
hedge layer needs a lead and a capture without one cannot replay a hedge
faithfully — see limitation 15 for what it does instead.

`capture_is_continuous` is the check to reach for before trusting a capture, and
it is cheap: it reads only the sidecar, never the data file. A capture written by
two processes returns `False`, and `sessions(path)` returns the `session_start`
lines so you can see where the seam is. Neither one repairs the seam — a
resumed capture has to be split before it is worth replaying (limitation 13).

Other flags: `--workers N` (0 = auto, `cpu_count - 1`), `--alpha-std`,
`--informed-rate` to override market calibration from the shell, `--exec-n` for
path count, `--out` for the sweep CSV. Counts are validated at the parser, so
`--days 0` or `--exec-n 1` is a one-line usage error rather than a run that
quietly substitutes a default.

`backtest` is the cheapest way in — on the machine these numbers were measured
on, a 1-day run takes about 6 seconds of wall clock and a 3-day run about 15
(five rungs per session, 10 seeds), and prints the full ladder with paired
t-statistics. `exec` is the slow one by design.

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
| `audit_log.jsonl` | per-run event trail for the **last** rung of the ladder (`+ hedging (full)`) on seed 0 — fills, hedges, risk rejects (first, 100th, 1000th), throttle transitions, kill switch. Each record carries a real wall clock, a sim-time field, and the rung in `run`. The file is rewritten on each run, not appended to. |
| `equity.png` | mean cumulative PnL path per rung. |
| `<capture>.jsonl` | `record` output: one merged event per line, book columns and trade columns mutually exclusive via `null`. Strict JSON. Rotated into `<capture>.partNNNN.jsonl` once a segment passes `--rotate-mb`. |
| `<capture>.meta.jsonl` | recorder diagnostics beside the capture — a `session_start` per writer, sequence gaps, stale streams, disconnects, and a `closed` line. Kept out of the capture so a gap can never be read as a quiet second. |

The capture and its sidecar are two views of the same run, split by whether a
line is market data or a claim about it. Two consecutive rows of a capture — one
book, one print, the second half of the row `null`:

```json
{"ts_ms":1790763416817,"mid":100.0,"bid":99.9,"ask":100.1,"bid_qty":3.0,"ask_qty":4.0,"futures":null,"px":null,"qty":null,"side":null}
{"ts_ms":1790763416917,"mid":null,"bid":null,"ask":null,"bid_qty":null,"ask_qty":null,"futures":null,"px":100.0,"qty":1.0,"side":1}
```

And the whole sidecar for that session, which is two lines long because nothing
went wrong:

```json
{"kind":"session_start","wall":1790763416.714,"session":"1a0f1d13c89-29b6","pid":10678,"resumed":false,"prior_bytes":null,"venue":"binance:BTCUSDT","path":"…/btc.jsonl"}
{"kind":"closed","venue":"binance:BTCUSDT","rows":22,"book":11,"trades":11,"anomalies":1,"segments":1,"session":"1a0f1d13c89-29b6","path":"…/btc.jsonl"}
```

The `session` id is the join key between them, and the absence of a diagnostic
between the two is itself information: a sidecar that is only `session_start` and
`closed` says the feed was clean for its whole length.

These are real lines from a real capture, written against a stub socket rather
than Binance (limitation 9), with only the `path` field elided. They are quoted
here because they are illustrative, but deliberately not committed as *files*: a
capture sitting in the repo is indistinguishable from live market data at a
glance, and `data/` is gitignored for that reason. Regenerate one with `record`;
the committed `baseline/` artifacts are the two from the ablation ladder, not a
capture.

## Tests

```bash
python -m pytest -q      # 167 tests, ~1.5 min
```

Coverage is behavioural rather than incidental: RLS against a closed-form least
squares solution and across a regime shift; the running-sum VPIN against a naive
one; market fill intensity against the intensity formula; the PnL identity;
common-random-number and no-look-ahead properties of the execution study;
router non-degeneracy; a fast path proven equivalent to its general path;
parallel-vs-serial backtest agreement in both chunking regimes; and one test per
report number a reader could otherwise be misled by — the fee units, the
breakeven sign, which rung the audit trail belongs to, and the CLI refusing
counts too small to mean anything.

Two more are worth calling out because they guard prose rather than numbers. The
ablation report's NOTE paragraph is rung-aware: it used to describe every losing
rung as paying "0 of futures PnL against 0 of cost", which is nonsense for a
`hedge=False` config — reachable through the `ladder=` argument. And the
interaction block reads both of its axes out of `GRID` instead of a private copy
of them, so editing the grid cannot leave the cross term describing a grid that
no longer exists.

The recorder's own tests are named after the failure each one prevents, and a few
of them exist because a green suite was not enough. The clearest case: both
stream tasks used `while not True` to hold their receive loop open, which is
`while False` — the recorder connected, wrote nothing, and reconnected as fast
as the exchange accepted, reporting a clean zero-row session. Every other test
drove the message handlers directly, so none of them could see it.

The fix was to test above the handlers, not below them, and the suite now has
two layers that do. A fake socket is driven in-process to assert the reconnect
gap is non-zero. A stub `websockets` module, installed via `PYTHONPATH` in a
subprocess, runs the real CLI end to end and asserts the things a handler-level
test structurally cannot: that rows come out, that rotation reassembles through
the reader, that the futures mid reaches every frame, that the sidecar stays
silent when nothing is wrong, and that a zero-row or resumed capture is reported
rather than returned as a success. The subprocess matters — `import websockets`
happens inside the stream tasks, so a `sys.modules` patch in-process would leak
into the tests that assert the missing-dependency path.

Neither layer can confirm Binance's field names. That is what limitation 9 is
for, and it is why the stub is described there as a stub rather than a test of
the wire format.

## Project layout

```
flow_mm.py      market, signals, quoting, hedging, risk, backtest, CLI  (entry point)
exec_algos.py   execution schedules + venue router
analysis.py     sensitivity sweep and the interaction block
stats.py        paired statistics, no scipy
replay/         run the harness on recorded data instead of a synthetic market
  schema.py       canonical frame/trade tables; a new venue is a missing reader
  readers.py      capture and archive readers, capture_to_dataset
  store.py        Dataset, and the event->one-second resampler
  market.py       ReplayMarket: a recording presented as a Market lookalike
  record.py       live capture: CaptureWriter, BinanceSource, session helpers
  fixtures.py     hermetic test sessions, deliberately including the bad cases
tests/          pytest suite
baseline/       committed ablation artifacts (audit_log.jsonl, equity.png)
```

## Known limitations

The code refers back to this section; these are not hedges, they are the
boundary of what the harness measures.

1. **The market is synthetic.** Fills follow a Poisson intensity model. There is
   no queue position, no exchange latency, no order-book replenishment, and no
   genuine adverse-selection dynamics. Anything about queue effects or
   fill-probability realism is out of scope.
2. **The sim is generous.** The reported breakeven fee is about 8x the real
   ~0.3 bps exchange fee. The harness computes and prints that ratio rather
   than asserting it, because a breakeven well above the real fee means the
   environment is easy, not that the strategy is good.
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
8. **The quoting layer is calibrated to a ~$100 price level.** The
   Avellaneda-Stoikov half-spread is `_vterm * var + _kterm`, and `var` is a
   per-second variance in dollars squared while `_kterm` is a fixed dollar
   constant. Both are absolute, so the spread does not survive a change of
   instrument: in basis points the variance term grows *linearly with price* and
   the constant term shrinks inversely. At $100 the spread lands near 1bp, which
   is why the ladder looks sane; on a $64,000 instrument the same relative
   volatility quotes ~100bp wide, every quote falls outside a 3bp book, and the
   day returns zero fills and zero PnL. That zero is the correct answer for a
   strategy quoting that wide — what is not acceptable is that it reads like a
   strategy that found no edge, so it is pinned as a test
   (`test_a_price_level_the_quoting_layer_was_not_calibrated_for_fills_nothing`).
   Making the spread dimensionless is a prerequisite for trading anything that
   is not a ~$100 share, and it will move every number in `baseline/`.
9. **`record` trades nothing and cannot.** `BinanceSource` subscribes to public
   market data only; there is no authenticated endpoint anywhere in the project.
   It has also never been run against Binance: `websockets` is not installed
   here, so the message mapping and the reconnect behaviour are covered by fakes
   and by a stub socket, not by the real wire. A stub confirms the loop, the
   rotation, the futures carry-forward and the reader round trip; it cannot
   confirm the field names, and a stream rename would surface as a zero-row
   capture rather than an error. `record` now warns on a zero-row capture and on
   a resumed one, but check the `closed` line's count before trusting any file.
   A capture-only tool is what makes the sequencing survivable — capture, replay,
   paper, then live — since a reconnect bug cannot cost money while the
   recorder is incapable of sending an order.
10. **A capture is a partial-book snapshot stream, so it has no queue.** `@depthN`
   sends absolute top-N, not deltas, which is why a reconnect is self-healing:
   the book is correct the instant the stream opens and no venue backfills. The
   cost is that queue position is still unobserved, exactly as limitation 1 says
   for the synthetic market.
11. **Paired t-statistics are within-path.** Seeds are shared across rungs, so the
   tests describe this synthetic market, not a population of real ones. No
   parameter uncertainty is propagated into the intervals.
12. **`adverse` is a memo line, not a PnL term.** It is the 30-second move in the
    mid after each fill, signed so positive means the market went against us. It
    is the *negation* of the per-fill contribution to `inv_drift` and the
    sign-flip of the markout (per fill `mo == spread_c - adverse_c`), and is
    excluded from the additive decomposition to avoid double counting. Note the
    units differ: `spread` and `adverse` are dollars, `mo_b`/`mo_i` are basis
    points per share. It is also computed and aggregated but never printed or
    written, so it is currently a dead diagnostic.
13. **A resumed recording is detectable but not repaired.** Each writer stamps a
    `session_start` line into the sidecar with a per-process id and a `resumed`
    flag, so `sessions(path)` and `capture_is_continuous(path)` will tell you a
    capture was written by two runs, and `record` prints a warning. What that
    buys is knowledge, not a fix: the rows on both sides are still one flat
    timeline, and `resample` forward-fills the gap between them into seconds that
    are flagged `stale` but otherwise indistinguishable from a quiet market. A
    capture with two sessions has to be split before it is worth replaying. The
    marker lives in the sidecar rather than the data file precisely so that
    splitting it needs no format change.
14. **An out-of-order row is written, not resorted — the sidecar is the only
    record.** A backwards `ts_ms` (NTP correction, or a reconnect after the
    exchange and the local clock disagree) still gets written, because refusing
    it would lose a real print, but `resample` buckets by second and will assign
    it to the earlier one, building a frame from two events the venue never put
    in the same second. The writer notes this as `out_of_order`, once per
    excursion, and that note is the only trace: `capture_to_dataset` does not
    consult it, so a replay built from such a capture looks entirely normal. The
    check is on the second, not the millisecond, because that is what the
    resampler buckets on — the two streams interleave inside a second routinely
    and flagging that would train you to ignore the sidecar. If a capture has an
    `out_of_order` note, treat the affected second as suspect.
15. **A capture with no futures feed replays as if the lead were exactly zero.**
    `resample` substitutes the spot `mid` wherever `futures` is missing or
    non-finite (`store.py`, `fut = np.where(np.isfinite(fut), fut, mid)`), so a
    capture recorded without a futures stream — or one whose futures stream
    died — produces frames whose futures lead is identically zero rather than
    absent. The choice is deliberate: a NaN would propagate into the RLS and
    poison every downstream rung with a non-finite input, and a zero lead is at
    least a real number. But the cost is that a dead futures feed is
    indistinguishable in the replay from a perp that tracked spot exactly, which
    is not a market that exists. Nothing warns about it: the writer's
    `futures_stale` note lands in the sidecar and `capture_to_dataset` does not
    read it, exactly as in limitation 14. If a capture has a `futures_stale`
    note, the hedge layer in that replay is fiction, and the `+ hedging (full)`
    rung should not be read as a result.

## License

MIT — see [LICENSE](LICENSE).
