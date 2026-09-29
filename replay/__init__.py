"""Replay: run the existing harness against recorded market data instead of a
synthetic market.

The point is narrow and testable. In `flow_mm.Market` the imbalance `I`, the
futures proxy `F` and the exogenous tape are all *generated from the same latent
drift the signal layer is trying to predict* -- so the signal can be read off
the state, and the reported "signal value" correlation is partly an artefact of
that coupling. Replay replaces those inputs with observed ones and leaves the
engine, the risk layer and the statistics untouched.

What replay does and does not fix:

  fixed    the signal layer reads a real book, a real tape, real volatility
  fixed    the fee breakeven is compared against the venue's real fee schedule
  NOT fixed  fills. We observe the market but not our counterparty, so a fill
             model is still required; see `market.FillModel`. Queue position --
             the largest single reason the synthetic market is generous -- is
             explicitly out of scope here and is the next step.

Typical use:

    ds = load("btcusdt-2026-09-28.npz")
    backtest(markets=[ReplayMarket(ds)], ladder=LADDER)
"""
from .market import FillModel, ReplayMarket
from .schema import FRAME_COLUMNS, TRADE_COLUMNS, ValidationError
from .store import Dataset, load, resample, save

__all__ = [
    "Dataset",
    "FillModel",
    "FRAME_COLUMNS",
    "ReplayMarket",
    "TRADE_COLUMNS",
    "ValidationError",
    "load",
    "resample",
    "save",
]
