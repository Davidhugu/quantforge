"""
analysis.py -- sensitivity grid over the market's calibration knobs.

The ablation ladder varies *features*. It cannot tell you which *parameters*
matter, and in the original harness the entire result hinged on two numbers
nobody had examined: `tox_coef`/`tox_floor` decide how wide the informed tier
gets, which in turn decides what fraction of informed arrivals actually cross.
A ladder run at one arbitrary setting is an anecdote.

This sweeps those knobs one at a time and reports, per grid point, the mean PnL
of every rung plus the paired t-statistic of each rung against the baseline.
Three things decide whether a result is worth anything:

  * does the edge survive plausible fees?
  * does it survive alpha_std = 0? That row removes the exploitable drift
    entirely, so any PnL still standing there is spread capture, not prediction.
  * does the RANKING of the rungs survive, or is it an artifact of one setting?
"""
from __future__ import annotations

import csv
from dataclasses import replace

from flow_mm import LADDER, Market, MarketParams, backtest
from stats import paired

# (field, values). `None` as the field means "sweep the fee", which is a Cfg
# knob rather than a MarketParams one and travels through cfg_fee instead.
GRID = [
    # alpha_std = 0 is the honest anchor. With no exploitable drift the signal
    # layer has nothing to harvest and the informed tier has no edge to defend
    # against; if the top rung's edge survives that row, it is not an edge.
    ("alpha_std",     [0.0, 5e-4, 1.2e-3, 2.5e-3]),
    ("informed_horizon", [10, 30, 60]),
    ("informed_rate", [0.02, 0.04, 0.10]),
    ("A",             [0.25, 1.0]),
    ("intraday_k",    [0.0, 1.5]),
    (None,            [-0.0005, 0.0, 0.0030]),      # fee, $/share
    ("tox_coef",      [0.0, 0.03, 0.06]),           # Cfg knob, not MarketParams
]

# the rung whose paired t against the baseline drives the '*' column
_REF = "+ alpha signal"


def _key(name: str) -> str:
    return name.replace("+ ", "").replace(" ", "_").replace("&", "and").replace("/", "_")


def _interaction(days: int, workers: int) -> list:
    """The cross term that one-at-a-time cannot see.

    `alpha_std` sets how much drift is predictable and `informed_horizon` sets
    how long the market's informed traders can wait to hit us. Together they
    decide whether the signal is *reachable* before the informed tier cuts it
    off. Moving either alone leaves the other fixed, so the interaction is
    invisible in a one-at-a-time grid -- and it is the one that matters.
    """
    out = []
    print(f"\nalpha_std x informed_horizon, {days} sessions per cell. The question is "
          f"whether the\n  signal rung still beats the no-signal rung once the "
          f"informed trader can wait longer.\n")
    print(f"  {'alpha_std':>10}{'horizon':>9}{'pred%':>7}{'dPrev':>9}{'tPrev':>8}"
          f"{'win':>6}{'inf%':>7}{'PnL full':>10}")
    print("  " + "-" * 68)
    for a_std in (0.0, 5e-4, 1.2e-3, 2.5e-3):
        for h in (10, 30, 60):
            mp = replace(MarketParams(), alpha_std=a_std, informed_horizon=h)
            r = backtest(days=days, mp=mp, quiet=True, workers=workers, ladder=LADDER)
            k = _key(_REF)
            i = [c.name for c in LADDER].index(_REF)
            pp = paired(r["pnl"][LADDER[i - 1].name], r["pnl"][_REF])
            top = r["agg"][LADDER[-1].name]
            # same column names as the one-at-a-time grid, so both blocks are
            # directly comparable in the CSV rather than needing a join
            row = dict(factor="interaction", value=f"{a_std:g}x{h}",
                       alpha_std=a_std, informed_horizon=h,
                       predictable_share=round(r["cal"]["predictable_share"], 4),
                       **{f"dprev_{k}": round(pp["delta"], 1),
                          f"tprev_{k}": round(pp["t"], 2),
                          f"winprev_{k}": round(pp["win"], 3),
                          f"ci_prev_{k}": f"[{pp['lo']:.0f}, {pp['hi']:.0f}]"},
                       inf_share=top["inf_share"], pnl_full=top["pnl"],
                       pnl_over_dd=top["pnl"] / max(1.0, top["dd"]))
            out.append(row)
            print(f"  {a_std:>10g}{h:>9}{row['predictable_share'] * 100:>6.1f}%"
                  f"{pp['delta']:>9,.0f}{pp['t']:>8.1f}{pp['win'] * 100:>5.0f}%"
                  f"{top['inf_share'] * 100:>6.1f}%{top['pnl']:>10,.0f}"
                  f"{'  *' if pp['sig'] else ''}")
    return out


def sweep(days: int = 10, out: str | None = "sweep.csv", quiet: bool = False,
          workers: int = 1) -> list:
    """One-at-a-time sensitivity around the MarketParams defaults.

    One-at-a-time rather than a full factorial: a factorial over these six
    factors is 144x the cost of one backtest and does not fit in a terminal.
    The one interaction that matters -- alpha_std against the informed tier's
    reach -- is measured separately by `_interaction`; see README, Known
    limitations.
    """
    base = MarketParams()
    # The market cache only helps serially: handing 20k-step Market objects to a
    # worker process costs more than rebuilding them there. In parallel we let
    # each worker rebuild from (params, seed), which is deterministic and so
    # still keeps every grid point paired on identical paths.
    markets = [Market(base, s) for s in range(days)] if workers == 1 else None
    rows = []

    def run(mp: MarketParams, fee: float, tag: str, factors: str,
            ladder=None) -> dict:
        if workers == 1:
            # identical params => identical market path; reuse the cache
            mk = markets if mp == base else [Market(mp, s) for s in range(days)]
            r = backtest(days=days, mp=mp, cfg_fee=fee, quiet=True, markets=mk,
                         ladder=ladder or LADDER)
        else:
            r = backtest(days=days, mp=mp, cfg_fee=fee, quiet=True,
                         workers=workers, ladder=ladder or LADDER)
        sig = r["agg"][_REF]
        top = r["agg"][LADDER[-1].name]
        row = dict(factor=tag, value=factors,
                   predictable_share=round(r["cal"]["predictable_share"], 4),
                   pnl_full=top["pnl"], pnl_best=max(a["pnl"] for a in r["agg"].values()),
                   pnl_naive=r["agg"][LADDER[0].name]["pnl"],
                   inf_share=top["inf_share"], fills=top["fills"],
                   alpha_corr=sig["alpha_corr"], oracle_corr=sig["oracle_corr"],
                   avg_inv=top["avg_inv"], max_dd=top["dd"],
                   pnl_over_dd=top["pnl"] / max(1.0, top["dd"]))
        # Both contrasts, exactly as the ladder prints them: a grid point that
        # only carries the vs-base t flatters the stack and hides a layer that
        # is significant-but-negative against the rung beneath it.
        for i, c in enumerate(LADDER[1:], start=1):
            k = _key(c.name)
            pb = paired(r["pnl"][LADDER[0].name], r["pnl"][c.name])
            pp = paired(r["pnl"][LADDER[i - 1].name], r["pnl"][c.name])
            row[f"dbase_{k}"] = round(pb["delta"], 1)
            row[f"tbase_{k}"] = round(pb["t"], 2)
            row[f"dprev_{k}"] = round(pp["delta"], 1)
            row[f"tprev_{k}"] = round(pp["t"], 2)
            row[f"winprev_{k}"] = round(pp["win"], 3)
            row[f"ci_prev_{k}"] = (f"[{pp['lo']:.0f}, {pp['hi']:.0f}]")
        rows.append(row)
        if not quiet:
            k = _key(_REF)
            t = row["tprev_" + k]          # vs the rung beneath: the honest one
            star = "*" if abs(t) > 2.0 else " "
            print(f"  {tag:<16}{factors:<9}{row['predictable_share'] * 100:>6.1f}%"
                  f"{row['pnl_full']:>9,.0f}{row['pnl_best']:>9,.0f}"
                  f"{row['pnl_naive']:>9,.0f}{row['pnl_over_dd']:>7.1f}"
                  f"{row['inf_share'] * 100:>6.1f}%{row['dprev_' + k]:>8,.0f}"
                  f"{t:>7.1f}{row['winprev_' + k] * 100:>5.0f}%{star:>3}")
        return row

    if not quiet:
        print(f"\nSensitivity, {days} paired sessions per point. dPrev/tPrev/win are "
              f"the\n  '+ alpha signal' rung against the rung beneath it (no signal); "
              f"* is |t| > 2.\n")
        print(f"  {'factor':<16}{'value':<9}{'pred%':>7}{'PnL full':>9}"
              f"{'PnL best':>9}{'PnL naive':>9}{'PnL/DD':>7}{'inf%':>7}"
              f"{'dPrev':>8}{'tPrev':>7}{'win':>6}{'':>3}")
        print("  " + "-" * 100)

    run(base, 0.0, "baseline", "default")
    for field, values in GRID:
        for v in values:
            if field is None:                      # fee is a Cfg knob
                run(base, float(v), "fee_ps", str(v))
            elif field == "tox_coef":
                # tox_coef/tox_floor are Cfg fields, so this sweeps the quoting
                # response to toxicity rather than the market's toxicity
                run(base, 0.0, field, str(v),
                    ladder=[replace(c, tox_coef=v) for c in LADDER])
            else:
                run(replace(base, **{field: v}), 0.0, field, str(v))

    # The interaction block is 12 more backtests, so `quiet` skips it as well
    # as the printing.
    inter = [] if quiet else _interaction(days, workers)

    if out:
        rows_all = rows + inter
        keys = sorted({k for r in rows_all for k in r})
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows_all)
        if not quiet:
            anchor = [r for r in rows if r["factor"] == "alpha_std" and r["value"] == "0.0"]
            print(f"\n  -> {out}")
            if anchor:
                a = anchor[0]
                print(f"  alpha_std=0 anchor: PnL full = {a['pnl_full']:,.0f} $/day, "
                      f"naive = {a['pnl_naive']:,.0f}. Anything the full stack still earns\n"
                      f"  here is spread capture, not signal, and does not depend on the "
                      f"forecast being right.")
    return rows + inter


if __name__ == "__main__":
    import sys
    sweep(days=int(sys.argv[1]) if len(sys.argv) > 1 else 10,
          workers=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
