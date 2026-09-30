"""Tests for the capture recorder.

The recorder is the first component in the project that talks to something
unreliable, so these tests are mostly about the ways a capture goes quietly
wrong rather than about the writer doing its obvious job. Each test names the
failure it prevents:

  * a capture that emits a bare `NaN` token, which nothing but Python can read
    back, and which `read_events` would accept -- so the file looks fine until
    the first non-Python tool touches it;
  * a window derived from a capture that started mid-second, whose opening book
    is a snapshot of a book that was already moving;
  * `resample` silently substituting the spot mid for a missing futures value,
    which turns the futures-lead feature into a constant zero and looks exactly
    like a signal that stopped working;
  * rotation, which drops the earlier segments if the reader only globs one file
    -- the failure is a capture that looks complete and starts halfway through;
  * the aggressor side, where a sign error inverts the tape and therefore the
    entire toxicity estimate, and which no aggregate statistic would reveal.

`record()`'s network path is untested here and is meant to be: the tests drive
the writer directly so that the whole module stays verifiable with no socket.
"""
import json
import os
import pathlib
import time

import numpy as np
import pytest

from replay import (CaptureWriter, Dataset, ReplayMarket, ValidationError,
                    capture_is_continuous, capture_to_dataset, read_sidecar,
                    sessions)
from replay.record import BinanceSource


def _session(w, seconds=6, start_ms=1_700_000_000_000, px0=100.0, trades=True):
    """A well-formed session: a book snapshot every second, prints inside it."""
    for t in range(seconds):
        base = start_ms + t * 1000
        m = px0 + 0.25 * t
        w.book(base, m - 0.05, m + 0.05, 10.0, 12.0)
        if trades:
            for k in range(3):
                w.trade(base + 100 + k * 10, m + (0.05 if k % 2 else -0.05), 1.5,
                        1 if k % 2 else -1)
    return start_ms


def _read_sidecar(path):
    side = os.path.splitext(str(path))[0] + ".meta.jsonl"
    with open(side) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- the capture contract ----------------------------------------------------

def test_round_trip_through_the_existing_reader_is_lossless(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        start = _session(w)
    ds = capture_to_dataset(str(p))
    # six written seconds, opening one dropped because its book was never seen
    assert ds.n_seconds == 5
    assert ds.frames["ts_ms"][0] == start + 1000
    assert ds.frames["bid"][0] == pytest.approx(100.20)
    assert ds.frames["ask_qty"][0] == pytest.approx(12.0)
    assert len(ds.trades["px"]) == 15   # the dropped second took its 3 prints
    assert set(np.unique(ds.trades["side"])) == {-1, 1}


def test_capture_is_strict_json_not_python_nan(tmp_path):
    """`json.dumps` emits a bare `NaN` by default; strict parsers reject it.

    `parse_constant` fires only for the `NaN`/`Infinity` tokens, so a file that
    survives this is readable by `jq`, pandas and any other tool -- while still
    round-tripping to NaN inside `_from_jsonl`.
    """
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w)
    with open(p) as fh:
        lines = [line for line in fh if line.strip()]

    def boom(tok):
        raise AssertionError(f"non-standard JSON token {tok!r} in the capture")

    assert len(lines) == 24
    for line in lines:
        row = json.loads(line, parse_constant=boom)
        assert row["px"] is None or isinstance(row["px"], float)


def test_null_is_read_back_as_nan_so_the_padding_path_is_unchanged(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w)
    with open(p) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    trade = next(r for r in rows if r["px"] is not None)
    assert trade["mid"] is None and trade["bid"] is None
    ds = capture_to_dataset(str(p))
    assert np.isfinite(ds.frames["mid"]).all()


def test_the_opening_second_is_dropped_because_the_book_was_never_seen(tmp_path):
    """The first book row is the first thing the recorder ever saw.

    Prints before it were matched against levels nobody was watching, so that
    second's signed volume -- which feeds the toxicity estimator -- is short by
    an unknown amount. The offset within the second is irrelevant and is not
    what the rule keys on: a capture can start at 0.1ms and still have missed
    the prints before the book, because there is no observation of "before".
    """
    p = tmp_path / "cap.jsonl"
    start = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for t in range(6):
            # first event lands 400ms into its second, as a real capture does
            w.book(start + t * 1000 + 400, 99.95, 100.05, 10.0, 12.0)
            w.trade(start + t * 1000 + 500, 100.0, 1.0, 1)
    ds = capture_to_dataset(str(p))
    assert ds.n_seconds == 5
    assert ds.frames["ts_ms"][0] == start + 1000
    assert not ds.frames["stale"].any()
    # and the rule does not depend on where inside the second the first event fell
    q = tmp_path / "aligned.jsonl"
    with CaptureWriter(str(q)) as w2:
        for t in range(6):
            w2.book(start + t * 1000, 99.95, 100.05, 10.0, 12.0)
            w2.trade(start + t * 1000 + 10, 100.0, 1.0, 1)
    assert capture_to_dataset(str(q)).n_seconds == 5


def test_partial_edges_can_be_kept_when_asked(tmp_path):
    p = tmp_path / "cap.jsonl"
    start = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for t in range(4):
            w.book(start + t * 1000 + 400, 99.95, 100.05, 10.0, 12.0)
            w.trade(start + t * 1000 + 500, 100.0, 1.0, 1)
    assert capture_to_dataset(str(p), complete_only=False).n_seconds == 4
    assert capture_to_dataset(str(p), complete_only=True).n_seconds == 3


def test_capture_too_short_to_form_an_interval_is_refused(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.book(1_700_000_000_000, 99.95, 100.05, 10.0, 12.0)
        w.trade(1_700_000_000_500, 100.0, 1.0, 1)
    with pytest.raises(ValidationError, match="two complete seconds"):
        capture_to_dataset(str(p))


# -- the futures lead must not silently go flat --------------------------------

def test_held_futures_mid_is_carried_onto_book_rows(tmp_path):
    """A null futures column is silently replaced by the spot mid downstream.

    `resample` does `fut = where(isfinite(fut), fut, mid)`, so an unpopulated
    futures column becomes a constant zero lead -- the RLS reads "no lead"
    instead of "no data" and nothing anywhere reports an error.
    """
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.hold_futures(100.25)
        _session(w)
    ds = capture_to_dataset(str(p))
    # spot mid is 100.0 -> 100.25 over the session, futures is pinned at 100.25,
    # so the lead is genuinely non-zero rather than an artefact of the fallback
    assert np.allclose(ds.frames["futures"], 100.25)
    assert not np.allclose(ds.frames["futures"], ds.frames["mid"])


def test_explicit_futures_beats_the_held_value(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.hold_futures(100.25)
        for t in range(3):
            w.book(1_700_000_000_000 + t * 1000, 99.95, 100.05, 1.0, 1.0,
                   futures=101.5)
    ds = capture_to_dataset(str(p))
    assert np.allclose(ds.frames["futures"], 101.5)


def test_a_capture_that_never_saw_futures_replays_as_an_exactly_flat_lead(tmp_path):
    """The fallback, pinned from the other direction -- and it is a real trap.

    `test_held_futures_mid_is_carried_onto_book_rows` pins that a *populated*
    futures column survives. This pins what happens when the feed never arrived
    at all, which is the case a reader would assume is obviously broken and which
    is in fact silently fine: `resample` substitutes the spot mid, so `futures`
    equals `mid` in every frame and the lead is exactly, uniformly zero.

    That is a number, so the RLS reads "the perp is tracking spot perfectly"
    rather than "there was no perp". A dead futures feed and a real one are
    therefore indistinguishable in the replay, and the `+ hedging (full)` rung
    will report a hedge against a lead of zero as though it were a result. This
    is README limitation 15; asserted here so that changing the fallback to NaN
    -- which would poison the signal layer instead -- is a deliberate act.
    """
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w)          # book + prints, no futures kwarg anywhere
    ds = capture_to_dataset(str(p))
    lead = ds.frames["futures"] - ds.frames["mid"]
    assert np.all(np.isfinite(ds.frames["futures"])), "NaN would reach the RLS"
    assert np.allclose(lead, 0.0), "the fallback is mid substitution, not NaN"
    # and nothing in the sidecar or the dataset records that it was a fallback
    assert not [n for n in _read_sidecar(p) if n["kind"] == "futures_stale"]


# -- reconnect behaviour ------------------------------------------------------

def test_resending_a_snapshot_is_a_no_op(tmp_path):
    """The property that justifies snapshots over diffs.

    A partial-depth stream re-sends absolute top-N on every connect, so a
    reconnect replays rows the capture already has. If duplicates moved the
    frames, every reconnect would corrupt the book; with last-wins they are
    inert, which is why the live source subscribes to snapshots.
    """
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with CaptureWriter(str(a)) as w:
        _session(w, seconds=6)
    with CaptureWriter(str(b)) as w:
        for t in range(6):                       # every snapshot sent twice
            m = 100.0 + 0.25 * t
            for _ in range(2):
                w.book(1_700_000_000_000 + t * 1000, m - 0.05, m + 0.05, 10.0, 12.0)
    da, db = capture_to_dataset(str(a)), capture_to_dataset(str(b))
    assert da.n_seconds == db.n_seconds == 5
    for k in ("mid", "bid", "ask", "bid_qty", "ask_qty"):
        assert np.allclose(da.frames[k], db.frames[k])


def test_a_sequence_gap_is_recorded_with_its_size_and_never_manufactured(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.note_seq_gap("trade", 1000, 1000)          # contiguous: not a gap
        w.note_seq_gap("trade", 1000, 1001)          # +1: reconnect churn
        w.note_seq_gap("trade", 1000, 1200)          # 200 missed prints
    notes = [n for n in _read_sidecar(p) if n["kind"] == "seq_gap"]
    assert [n["missed"] for n in notes] == [1, 200]
    # the market file is untouched by diagnostics
    with open(p) as fh:
        assert not [line for line in fh if line.strip()]


def test_silence_is_reported_but_never_smoothed(tmp_path):
    """A quiet book is a real market state, not proof of loss."""
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=4)
        w.note_stale("depth", 1_700_000_003_000, 1_700_000_009_000, 1000)
    note = [n for n in _read_sidecar(p) if n["kind"] == "stale"][0]
    assert note["silent_ms"] == 6000
    assert capture_to_dataset(str(p)).n_seconds == 3


def test_interior_hole_becomes_a_stale_second_not_a_flat_price(tmp_path):
    p = tmp_path / "cap.jsonl"
    start = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for t in (0, 1, 4, 5):
            m = 100.0 + 0.25 * t
            w.book(start + t * 1000, m - 0.05, m + 0.05, 10.0, 12.0)
            w.trade(start + t * 1000 + 100, m, 1.0, 1)
    ds = capture_to_dataset(str(p))
    assert ds.n_seconds == 5
    assert ds.frames["stale"].tolist() == [False, True, True, False, False]
    assert np.isfinite(ds.frames["mid"]).all() and (ds.frames["mid"] > 0).all()


# -- rotation ----------------------------------------------------------------

def test_rotation_writes_segments_and_the_reader_reassembles_them(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p), rotate_bytes=400) as w:
        _session(w, seconds=30)
        summary = w.close()
    assert summary["segments"] > 1
    # segment 0 keeps the original name; rollovers are `<stem>.partNNNN.jsonl`
    parts = sorted(q.name for q in tmp_path.glob("*.part*.jsonl")
                   if not q.name.endswith(".meta.jsonl"))
    assert len(parts) == summary["segments"] - 1
    assert p.exists()
    # a directory holds the sidecars too, and reading one as market data would
    # be a spectacular way to poison a replay
    assert len(list(tmp_path.glob("*.meta.jsonl"))) == summary["segments"]
    assert capture_to_dataset(str(p)).n_seconds == 29
    assert capture_to_dataset(str(tmp_path)).n_seconds == 29


def test_no_rotation_means_one_segment(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p), rotate_bytes=0) as w:
        _session(w, seconds=10)
    assert p.exists() and p.stat().st_size > 0
    assert not list(tmp_path.glob("*.part*.jsonl"))
    # a directory glob must find it too, or small captures look empty
    assert capture_to_dataset(str(tmp_path)).n_seconds == 9


# -- refusals ----------------------------------------------------------------

def test_a_capture_that_stops_on_a_rotation_boundary_is_still_readable(tmp_path):
    """A rotation fired by the last row must not cost the whole session.

    The rollover opens a new segment, and the session ends before anything
    reaches it. `read_events` rejects a zero-byte file as a missing-column
    error, so the reader used to die on that empty tail *before* reading any
    real data -- one rotation boundary at the end of a run and the entire
    capture was unreadable. Reproduced with a segment size small enough that
    the final row trips it.
    """
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p), rotate_bytes=300, flush_rows=1) as w:
        _session(w, seconds=21)
    parts = sorted(q for q in tmp_path.glob("*.jsonl")
                   if not q.name.endswith(".meta.jsonl"))
    assert all(q.stat().st_size > 0 for q in parts), \
        "a zero-byte segment makes the capture unreadable"
    assert capture_to_dataset(str(p)).n_seconds == 20


def test_reading_one_capture_does_not_pull_in_a_differently_named_one(tmp_path):
    """A prefix glob is cross-symbol contamination.

    `cap*.jsonl` also matches `cap2.jsonl`, so a second instrument's capture
    got concatenated into the first. Two symbols at different price levels then
    fail validation as "trade px falls outside the book range", which blames the
    feed for a bug in the reader. `eth.jsonl` vs `ethereum.jsonl` is the same
    mistake with a longer name.
    """
    a, b = tmp_path / "cap.jsonl", tmp_path / "cap2.jsonl"
    with CaptureWriter(str(a)) as w:
        _session(w, seconds=6, px0=100.0)
    with CaptureWriter(str(b)) as w:
        _session(w, seconds=6, start_ms=1_700_000_900_000, px0=64_000.0)

    only_a = capture_to_dataset(str(a))
    assert only_a.n_seconds == 5
    assert float(only_a.frames["mid"][0]) < 200.0, "the other symbol leaked in"

    eth, ethn = tmp_path / "eth.jsonl", tmp_path / "ethereum.jsonl"
    with CaptureWriter(str(eth)) as w:
        _session(w, seconds=6, px0=3_000.0)
    with CaptureWriter(str(ethn)) as w:
        _session(w, seconds=6, start_ms=1_700_000_900_000, px0=3_000.0)
    assert float(capture_to_dataset(str(eth)).frames["mid"][0]) < 1e4


def test_a_row_written_after_close_is_refused_rather_than_lost(tmp_path):
    """Silently dropping rows is the failure this module exists to prevent.

    A write after `close()` used to append to a buffer that was never flushed
    again: no exception, and `rows_written` went up, so the summary claimed 5
    rows while the file held 3. The live path is exactly where this happens --
    a reconnect handler still draining a queue while the main task tears the
    writer down.
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p))
    w.book(1_700_000_000_000, 99.95, 100.05, 10.0, 12.0)
    w.close()
    with pytest.raises(ValidationError, match="closed"):
        w.book(1_700_000_001_000, 99.95, 100.05, 10.0, 12.0)
    with pytest.raises(ValidationError, match="closed"):
        w.trade(1_700_000_001_000, 100.0, 1.0, 1)
    with pytest.raises(ValidationError, match="closed"):
        w.note("seq_gap", got=5, expected=6)
    with pytest.raises(ValidationError, match="closed"):
        w.flush()
    assert w.rows_written == 1, "rows_written must not count a refused row"
    assert len(p.read_text().splitlines()) == 1


def test_closing_twice_returns_the_same_summary_instead_of_a_traceback(tmp_path):
    """`__exit__` and an explicit `close()` both fire on the normal path."""
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.book(1_700_000_000_000, 99.95, 100.05, 10.0, 12.0)
        first = dict(w.close())
    second = w.close()
    assert first == second
    kinds = [n["kind"] for n in _read_sidecar(p)]
    assert kinds.count("closed") == 1, "a double close wrote two summaries"


def test_a_backwards_clock_step_is_noted_rather_than_silently_resorted(tmp_path):
    """An NTP correction lands a row in a second the capture already passed.

    The row is still written -- refusing it would lose a real print -- but
    `resample` buckets by second, so it gets assigned to the *earlier* second and
    a frame is built from two real events that the venue never put in the same
    second. Venue event time makes this rare rather than impossible, and rare is
    not never: a reconnect after the exchange and the local clock disagree is
    exactly this. The sidecar is what makes the relocation visible.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for k in range(4):
            w.book(t0 + k * 1_000, 99.95, 100.05, 10.0, 12.0)
        w.book(t0 - 4_000, 99.95, 100.05, 10.0, 12.0)   # 4s backwards
    notes = [n for n in _read_sidecar(p) if n["kind"] == "out_of_order"]
    assert len(notes) == 1, notes
    # 7s, not 4s: the distance is measured from the start of the last second the
    # capture passed, not from the last row. 3s of the file had already been
    # recorded past that boundary, and those 3s are exactly what makes the row
    # late rather than merely early.
    assert notes[0]["behind_ms"] == 7_000
    assert notes[0]["passed_second"] == (t0 + 3_000) // 1000
    # the row is kept, not dropped: it is a real print
    assert len(p.read_text().splitlines()) == 5


def test_one_note_per_excursion_not_one_per_row(tmp_path):
    """A feed that stayed behind for a minute must not write 60 notes."""
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        w.book(t0 + 5_000, 99.95, 100.05, 10.0, 12.0)
        for k in range(6):
            w.book(t0 + 1_000 + k * 100, 99.95, 100.05, 10.0, 12.0)
        w.book(t0 + 9_000, 99.95, 100.05, 10.0, 12.0)    # recovered
        w.book(t0 + 2_000, 99.95, 100.05, 10.0, 12.0)    # a second excursion
    notes = [n for n in _read_sidecar(p) if n["kind"] == "out_of_order"]
    assert len(notes) == 2, f"one per excursion, got {len(notes)}"


def test_two_streams_interleaving_inside_one_second_are_not_reported(tmp_path):
    """Book and trade events interleave constantly; that is not disorder.

    The check is on the *second*, not the millisecond, because that is what
    `resample` buckets on. A trade stamped 200ms behind the last book update is
    ordinary market structure, and flagging it would train you to ignore the
    sidecar -- which is the same failure as never writing the note at all.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        w.book(t0 + 100, 99.95, 100.05, 10.0, 12.0)
        w.trade(t0 + 50, 100.0, 1.0, 1)      # earlier, same second
        w.trade(t0 + 300, 100.0, 1.0, -1)
        w.book(t0 + 700, 99.95, 100.05, 10.0, 12.0)
    kinds = [n["kind"] for n in _read_sidecar(p)]
    assert "out_of_order" not in kinds, kinds
    assert w.rows_written == 4


def test_a_healthy_live_capture_is_not_drowned_in_false_out_of_order_notes(tmp_path):
    """The bug the live smoke test found: 52 notes in 20s on a clean feed.

    Binance will not interleave `@depth10@100ms` and `@trade` in time order.
    Measured on the real feed, the depth stream runs ~1.4s behind the trade
    stream, so the *file* steps backwards constantly while each stream is
    perfectly monotonic on its own. A shared watermark cannot tell those two
    situations apart, and it fired on every single interleaving -- turning a
    diagnostic that is supposed to mean "your clock moved" into one that means
    "a Tuesday".

    The shape below is the real one: trades march forward steadily, books lag
    behind them and arrive late but in order. Nothing here is wrong, so the
    sidecar must be silent.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    lag_ms = 1_400
    with CaptureWriter(str(p)) as w:
        for k in range(40):
            w.trade(t0 + k * 500, 100.0, 1.0, 1 if k % 2 else -1)
            w.book(t0 + k * 500 - lag_ms, 99.95, 100.05, 10.0, 12.0)
    notes = [n for n in _read_sidecar(p) if n["kind"] == "out_of_order"]
    assert notes == [], f"{len(notes)} false positives on a healthy capture"

    # ...and each stream is genuinely ordered, so this is not a test that passes
    # because the fixture is too tame to trigger anything
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    book = [r["ts_ms"] for r in rows if r["mid"] is not None]
    trades = [r["ts_ms"] for r in rows if r["mid"] is None]
    assert all(b >= a for a, b in zip(book, book[1:])), "fixture books regressed"
    assert all(b >= a for a, b in zip(trades, trades[1:])), "fixture trades regressed"
    # and the interleaving really does step the file backwards, which is what
    # used to trip the check: 40 trades, each followed by a book 1.4s behind it
    assert sum(1 for a, b in zip(rows, rows[1:]) if b["ts_ms"] < a["ts_ms"]) == 40


def test_a_stream_that_moves_backwards_relative_to_itself_is_still_caught(tmp_path):
    """Narrowing the watermark must not narrow it to nothing.

    A book stream that regresses against its own history is the real fault --
    an NTP step, or a reconnect to a lagging server -- and it is still caught,
    with the note now saying which stream moved.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for k in range(4):
            w.book(t0 + k * 1_000, 99.95, 100.05, 10.0, 12.0)
        w.book(t0 - 4_000, 99.95, 100.05, 10.0, 12.0)    # 4s back, same stream
    notes = [n for n in _read_sidecar(p) if n["kind"] == "out_of_order"]
    assert len(notes) == 1, notes
    assert notes[0]["stream"] == "book"
    assert notes[0]["behind_ms"] == 7_000

    # a trade stream regression is caught independently, and does not clear or
    # consume the book's excursion state
    p2 = tmp_path / "b.jsonl"
    with CaptureWriter(str(p2)) as w:
        for k in range(4):
            w.trade(t0 + k * 1_000, 100.0, 1.0, 1)
            w.book(t0 + k * 1_000, 99.95, 100.05, 10.0, 12.0)
        w.trade(t0 - 3_000, 100.0, 1.0, -1)
    kinds = [n for n in _read_sidecar(p2) if n["kind"] == "out_of_order"]
    assert len(kinds) == 1 and kinds[0]["stream"] == "trade", kinds


def test_a_second_boundary_crossing_is_caught_even_when_it_is_small(tmp_path):
    """1ms backwards can still change the bucket, and that is the point.

    The threshold is the second, not a tolerance, so a step small enough to be
    harmless is harmless and a step that crosses a boundary is caught even at
    1ms. Anything else would need a magic number for a condition that already
    has an exact definition.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000          # exactly on a second boundary
    with CaptureWriter(str(p)) as w:
        w.book(t0 + 1_500, 99.95, 100.05, 10.0, 12.0)   # second t0/1000 + 1
        w.book(t0 + 1_400, 99.95, 100.05, 10.0, 12.0)   # 100ms back, same one
    assert not [n for n in _read_sidecar(p) if n["kind"] == "out_of_order"]

    p2 = tmp_path / "b.jsonl"
    with CaptureWriter(str(p2)) as w:
        w.book(t0 + 1_000, 99.95, 100.05, 10.0, 12.0)  # first ms of a second
        w.book(t0 + 999, 99.95, 100.05, 10.0, 12.0)    # 1ms back: crosses
    notes = [n for n in _read_sidecar(p2) if n["kind"] == "out_of_order"]
    assert len(notes) == 1 and notes[0]["behind_ms"] == 1


def test_a_dead_futures_feed_is_reported_once_in_the_sidecar(tmp_path):
    """A carried-forward futures mid is a promise, and this is where it is kept.

    `hold_futures` said a dead feed was "reported in the sidecar where it is
    visible" and nothing reported it. So a frozen perp mid persisted forever,
    and a dead feed looked exactly like a genuinely flat basis -- which is the
    one thing the carry-forward was written to prevent, since the RLS would read
    "no lead" where the truth is "no data".
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p), futures_stale_ms=1_000,
                       futures_symbol="btcusdt") as w:
        w.hold_futures(100.10)
        w._held_at_ms = t0 - 5_000        # pretend the perp feed died 5s ago
        w.book(t0, 99.95, 100.05, 10.0, 12.0)                  # held: stale
        # a live mid clears staleness: the feed recovered, so the held value's
        # age restarts from here and one note is enough for the first outage
        w.book(t0 + 500, 99.95, 100.05, 10.0, 12.0, 100.10)     # fresh
        w.book(t0 + 600, 99.95, 100.05, 10.0, 12.0)             # held, 100ms
        w._held_at_ms = t0                 # died again, 1.2s before the next row
        w.book(t0 + 1_200, 99.95, 100.05, 10.0, 12.0)          # past threshold
        w.book(t0 + 1_300, 99.95, 100.05, 10.0, 12.0)          # still, no repeat
    notes = [n for n in _read_sidecar(p) if n["kind"] == "futures_stale"]
    assert len(notes) == 2, "one note per outage, not one per row"
    assert notes[0]["age_ms"] == 5_000
    assert notes[0]["after_ms"] == 1_000
    assert notes[0]["symbol"] == "btcusdt"
    assert notes[1]["age_ms"] == 1_200
    # the rows are still written, with the held value: fallback, not a hole
    assert capture_to_dataset(str(p), complete_only=False).n_seconds >= 1


def test_a_live_futures_mid_never_raises_a_stale_note(tmp_path):
    """The note must be about staleness, not about any held value."""
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p), futures_stale_ms=60_000) as w:
        w.hold_futures(100.10)
        w._held_at_ms = t0 - 1_000
        for k in range(3):
            w.book(t0 + k * 100, 99.95, 100.05, 10.0, 100.11 + k)
    assert not [n for n in _read_sidecar(p) if n["kind"] == "futures_stale"]


def test_a_capture_that_never_wrote_anything_still_leaves_evidence(tmp_path):
    """Segment 0 is not removed, unlike an empty rollover.

    Deleting a zero-byte tail has to stop at the base segment: a capture that
    was opened and then failed has to leave a trace, or a crash looks identical
    to a run that never happened.
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p))
    summary = w.close()
    assert summary["rows"] == 0
    # a capture that went wrong has to be distinguishable from a clean one, so
    # `anomalies` is the number a reader checks. Counting the session marker
    # made every clean capture report one fault it did not have -- found by the
    # live smoke test, which printed "1 anomalies" over a sidecar with no
    # diagnostic in it.
    assert summary["anomalies"] == 0, summary
    assert p.exists() and p.stat().st_size == 0
    assert [n["kind"] for n in _read_sidecar(p)] == ["session_start", "closed"]


def test_a_clean_live_shaped_capture_reports_zero_anomalies(tmp_path):
    """The whole point of the counter: zero means zero, on a busy capture."""
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for k in range(60):
            w.book(t0 + k * 500, 99.95, 100.05, 10.0, 12.0, futures=100.02)
            w.trade(t0 + k * 500 + 100, 100.0, 1.0, 1 if k % 2 else -1)
    summary = w.close()
    assert summary["rows"] == 120
    assert summary["anomalies"] == 0, summary
    assert summary["book"] == 60 and summary["trades"] == 60
    assert [n["kind"] for n in _read_sidecar(p)] == ["session_start", "closed"]


def test_a_restarted_capture_is_distinguishable_from_a_continuous_one(tmp_path):
    """Append mode with no marker reads back as one unbroken session.

    Two recorders pointed at one path -- a crash and restart, or an operator who
    forgot the first was still running -- produce a file that looks exactly like
    a single session, and `resample` then builds frames straight across the gap
    as though nothing was missing. Rotation segments and a `closed` line per
    segment were the only boundaries available, and a restart that lands in the
    same segment erases the seam entirely.
    """
    p = tmp_path / "cap.jsonl"
    t0 = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=6, start_ms=t0)
        first = w.session_id
    # a second process, same path, much later: a gap of an hour
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=6, start_ms=t0 + 3_600_000)
        second = w.session_id

    assert first != second, "two writers must not share a session id"
    s = sessions(str(p))
    assert len(s) == 2, f"expected a visible seam, got {s}"
    assert s[1]["resumed"] is True, "the second writer appended to real data"
    # the first writer's rows are still on disk, which is what `resumed` means
    assert s[1]["prior_bytes"] > 0
    assert s[0]["resumed"] is False and s[0]["prior_bytes"] is None
    # the market data itself is untouched -- append still works -- but the hour
    # between the two runs becomes 3600 seconds of forward-filled book, flagged
    # `stale` and otherwise indistinguishable from a quiet market. This is the
    # damage the marker is there to warn about: nothing downstream is wrong, and
    # nothing downstream will tell you.
    ds = capture_to_dataset(str(p), complete_only=False)
    assert ds.n_seconds == 3606
    # 3594 of the gap seconds are stale: the first second after a gap is not
    # flagged, because the book was carried forward into it, so the flat run
    # starts one second late. That off-by-one is the resampler's rule, not this
    # test's, and it is why the exact count is asserted loosely and the shape
    # sharply.
    assert ds.frames["stale"].sum() > 3_500
    assert not ds.frames["stale"][:6].any(), "the recorded run is real"
    assert not ds.frames["stale"][-6:].any()
    assert ds.frames["stale"][10:3590].all(), "the whole gap is flagged"
    assert not capture_is_continuous(str(p))


def test_a_single_writer_is_one_continuous_session(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=6)
    assert capture_is_continuous(str(p))
    assert len(sessions(str(p))) == 1


def test_a_capture_with_no_sidecar_is_not_evidence_of_continuity(tmp_path):
    """No diagnostics means absence of evidence, not evidence of continuity."""
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=6)
    os.remove(os.path.splitext(str(p))[0] + ".meta.jsonl")
    assert sessions(str(p)) == []
    assert capture_is_continuous(str(p)) is False


def test_a_session_marker_costs_nothing_when_the_capture_rotates(tmp_path):
    """One marker per writer, not one per segment.

    A rollover moves the sidecar to the new segment, so emitting a marker per
    segment would make a single session look like a dozen restarts -- the same
    false alarm as flagging every interleaved row, and it would train you to
    ignore the check entirely.
    """
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p), rotate_bytes=300, flush_rows=1) as w:
        _session(w, seconds=21)
    parts = sorted(q.name for q in tmp_path.glob("*.jsonl")
                   if not q.name.endswith(".meta.jsonl"))
    assert len(parts) > 1, "the capture did not rotate; the test proves nothing"
    assert len(sessions(str(p))) == 1
    assert capture_is_continuous(str(p))


def test_a_dead_futures_stream_reconnects_instead_of_ending_the_recording(tmp_path):
    """`asyncio.TimeoutError` was re-raised out of the futures task.

    90s of perp silence is not a normal gap, but re-raising killed the task
    outright: no note, no reconnect, and a capture whose futures column quietly
    froze while the spot stream carried on. The held mid is flagged
    `futures_stale` by then, so all that was left was to reconnect.
    """
    import asyncio

    import sys

    class FakeWS:
        def __init__(self, conn):
            self.conn = conn

        async def recv(self):
            if self.conn.calls == 1:
                raise asyncio.TimeoutError
            # park on the retry, so the test terminates even if the loop is wrong
            await asyncio.sleep(3600)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeConnect:
        def __init__(self):
            self.calls = 0

        def __call__(self, *a, **k):
            self.calls += 1
            return FakeWS(self)

    conn = FakeConnect()
    sys.modules["websockets"] = type("M", (), {"connect": staticmethod(conn)})
    p = tmp_path / "cap.jsonl"
    src = BinanceSource("btcusdt", "ethusdt")
    try:
        with CaptureWriter(str(p)) as w:
            w.hold_futures(100.10)

            async def drive():
                stop = asyncio.Event()
                t = asyncio.create_task(src._run_futures(w, stop))
                # poll rather than sleep a guessed interval: the backoff is
                # 0.5s plus jitter, so a fixed wait is either flaky or slow
                for _ in range(60):
                    if conn.calls >= 2:
                        break
                    await asyncio.sleep(0.05)
                stop.set()
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            asyncio.run(drive())
    finally:
        sys.modules.pop("websockets", None)

    notes = [n for n in _read_sidecar(p) if n["kind"] == "futures_timeout"]
    assert len(notes) == 1, f"one note per timed-out attempt, got {len(notes)}"
    assert conn.calls == 2, f"it should reconnect exactly once, got {conn.calls}"


def test_trade_side_cannot_be_defaulted(tmp_path):
    """A wrong sign silently inverts the tape and the toxicity estimate with it."""
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        with pytest.raises(ValidationError, match="side must be"):
            w.trade(1_700_000_000_000, 100.0, 1.0, 0)
        with pytest.raises(ValidationError, match="side must be"):
            w.trade(1_700_000_000_000, 100.0, 1.0, 2)


def test_a_removed_level_is_dropped_rather_than_written_as_a_zero_price(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        w.book(1_700_000_000_000, 99.95, 100.05, 10.0, 12.0)
        w.book(1_700_000_001_000, 0.0, 100.05, 0.0, 12.0)   # bid level removed
        w.trade(1_700_000_001_500, 0.0, 1.0, 1)            # bad print
        w.book(1_700_000_002_000, 99.95, 100.05, 10.0, 12.0)
    assert w.n_book == 2 and w.n_trade == 0
    kinds = [n["kind"] for n in _read_sidecar(p)]
    assert kinds == ["session_start", "bad_book", "bad_trade", "closed"]
    # second 1 has no book event at all, so it is ffilled and flagged stale
    ds = capture_to_dataset(str(p), complete_only=False)
    assert ds.frames["stale"].tolist() == [False, True, False]
    assert ds.frames["bid"][1] == pytest.approx(99.95)


def test_unsupported_depth_is_refused_before_a_socket_is_opened():
    # 1, 7, 15: the source checked `1 <= depth <= 20`, so these reached the stream
    # URL as invalid depth subscriptions, which Binance answers by closing the
    # socket -- a stream that opens and then dies, far harder to diagnose than a
    # refusal at construction.
    for kwargs in ({"depth": 25}, {"depth": 0}, {"depth": 1}, {"depth": 7},
                   {"depth": 15}, {"update_ms": 500}):
        with pytest.raises(ValidationError):
            BinanceSource("btcusdt", **kwargs)
    for ok in (5, 10, 20):
        assert BinanceSource("btcusdt", depth=ok).depth == ok


# -- Binance message mapping -------------------------------------------------

def test_aggressor_side_is_derived_from_the_maker_flag_not_guessed():
    """`m` true means the buyer was passive, so the aggressor sold."""
    class W:
        def __init__(self):
            self.rows = []
        def trade(self, ts, px, qty, side):
            self.rows.append(side)
    w = W()
    BinanceSource._on_trade(w, {"p": "100.0", "q": "1.0", "T": 1, "m": True})
    BinanceSource._on_trade(w, {"p": "100.0", "q": "1.0", "T": 2, "m": False})
    assert w.rows == [-1, 1]


def test_depth_uses_exchange_event_time_and_the_top_of_book():
    """Wall-clock-on-arrival would bake this machine's latency into the replay."""
    class W:
        def __init__(self):
            self.row = None
        def book(self, ts, bid, ask, bq, aq, futures=None):
            self.row = (ts, bid, ask, bq, aq)
    w = W()
    BinanceSource._on_depth(w, {"E": 1_700_000_000_123, "bids": [["99.9", "3"], ["99.8", "9"]],
                  "asks": [["100.1", "4"], ["100.2", "9"]]}, 999)
    assert w.row == (1_700_000_000_123, 99.9, 100.1, 3.0, 4.0)
    # the partial-book spelling, and a snapshot that has only bid or ask
    w2 = W()
    BinanceSource._on_depth(w2, {"b": [["1", "1"]], "a": [["2", "1"]]}, 5)
    assert w2.row == (5, 1.0, 2.0, 1.0, 1.0)
    w3 = W()
    BinanceSource._on_depth(w3, {"E": 1, "bids": [["1", "1"]], "asks": []}, 5)
    assert w3.row is None


def test_one_spot_connection_carries_both_streams():
    s = BinanceSource("btcusdt")
    assert s._streams() == "btcusdt@depth10@100ms/btcusdt@trade"
    assert s.base.startswith("wss://") and s.fbase.startswith("wss://")


def test_the_reconnect_loop_waits_instead_of_spinning_on_a_closed_socket(tmp_path):
    """A closed socket is a real condition, and the backoff is the point.

    With the inner loop fixed to `while not stop.is_set()`, a connection that
    fails repeatedly used to come straight back with no pause at all -- a
    connect-per-millisecond loop against the exchange, which is a good way to get
    an IP banned and a bad way to notice anything is wrong. The backoff is what
    makes the reconnect cheap, so it is asserted, not assumed.
    """
    import asyncio
    import sys

    class FakeWS:
        async def recv(self):
            raise ConnectionResetError("closed by peer")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeConnect:
        def __init__(self):
            self.calls = 0
            self.stamps = []

        def __call__(self, *a, **k):
            self.calls += 1
            self.stamps.append(time.monotonic())
            return FakeWS()

    conn = FakeConnect()
    sys.modules["websockets"] = type("M", (), {"connect": staticmethod(conn)})
    p = tmp_path / "cap.jsonl"
    src = BinanceSource("btcusdt")
    try:
        with CaptureWriter(str(p)) as w:

            async def drive():
                stop = asyncio.Event()
                t = asyncio.create_task(src._run_spot(w, stop))
                for _ in range(200):
                    if conn.calls >= 3:
                        break
                    await asyncio.sleep(0.02)
                stop.set()
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            asyncio.run(drive())
    finally:
        sys.modules.pop("websockets", None)

    assert conn.calls >= 3, f"only {conn.calls} attempts"
    gaps = [b - a for a, b in zip(conn.stamps, conn.stamps[1:])]
    assert all(g > 0.2 for g in gaps), \
        f"reconnected with no backoff: gaps={gaps}"
    kinds = [n["kind"] for n in _read_sidecar(p)]
    assert kinds.count("spot_disconnect") >= 2, kinds


def test_a_connected_socket_actually_produces_rows(tmp_path):
    """The receive loop was `while not True`, i.e. `while False`.

    Both stream tasks connected, fell straight out of the inner loop, and
    reconnected as fast as the exchange would accept -- a tight spin that wrote
    nothing. `record --seconds 3600` would have produced a valid, clean-looking
    zero-row capture, so the only symptom was a capture with no market in it.
    Every other test in this file drove the message handlers directly, which is
    exactly why this survived them: the loop around them never ran.
    """
    import asyncio
    import sys

    class FakeWS:
        def __init__(self, msgs):
            self.msgs = list(msgs)

        async def recv(self):
            if self.msgs:
                return self.msgs.pop(0)
            await asyncio.sleep(3600)     # nothing more to say

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    depth = {"e": "depthUpdate", "E": 1_700_000_000_123,
             "b": [["99.90", "3"]], "a": [["100.10", "4"]]}
    trade = {"e": "trade", "T": 1_700_000_000_456, "t": 7,
             "p": "100.00", "q": "1.5", "m": True}

    class FakeConnect:
        def __init__(self, msgs):
            self.msgs = msgs
            self.calls = 0

        def __call__(self, *a, **k):
            self.calls += 1
            return FakeWS(self.msgs)

    conn = FakeConnect([json.dumps({"stream": "x", "data": depth}),
                        json.dumps({"stream": "x", "data": trade})])
    sys.modules["websockets"] = type("M", (), {"connect": staticmethod(conn)})
    p = tmp_path / "cap.jsonl"
    src = BinanceSource("btcusdt")
    try:
        with CaptureWriter(str(p)) as w:

            async def drive():
                stop = asyncio.Event()
                t = asyncio.create_task(src._run_spot(w, stop))
                for _ in range(60):
                    if w.rows_written >= 2:
                        break
                    await asyncio.sleep(0.02)
                stop.set()
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            asyncio.run(drive())
    finally:
        sys.modules.pop("websockets", None)

    assert w.rows_written == 2, f"rows={w.rows_written} connects={conn.calls}"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["px"] for r in rows] == [None, 100.0]
    assert rows[0]["bid"] == 99.9 and rows[1]["side"] == -1


def test_writer_needs_no_websocket_client():
    """`replay` must stay importable on the three pinned research dependencies."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import replay, sys; "
         "assert 'websockets' not in sys.modules, 'importing replay pulled in a "
         "websocket client'; print('ok')"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_record_without_a_websocket_client_says_so_clearly(tmp_path):
    import importlib.util
    if importlib.util.find_spec("websockets") is not None:
        pytest.skip("websockets is installed; this covers the missing-dep path")
    from replay import record
    out = tmp_path / "c.jsonl"
    with pytest.raises(ValidationError, match="pip install websockets"):
        record(symbol="btcusdt", out=str(out), seconds=1)


def test_a_failed_record_leaves_no_capture_behind(tmp_path):
    """A zero-byte capture is indistinguishable from an empty market.

    Opening the writer before the source meant every run without `websockets`
    installed left `c.jsonl` and its sidecar on disk. Anything listing the
    directory then saw a capture, and `capture_to_dataset` on it failed on a file
    that was never going to have data. This is what produced the stray `data/`
    in the working tree.
    """
    import importlib.util
    if importlib.util.find_spec("websockets") is not None:
        pytest.skip("websockets is installed; this covers the missing-dep path")
    from replay import record
    out = tmp_path / "c.jsonl"
    with pytest.raises(ValidationError):
        record(symbol="btcusdt", out=str(out), seconds=1)
    assert not out.exists(), "a refused run must not create a capture"
    assert not list(tmp_path.iterdir()), f"left {sorted(tmp_path.iterdir())}"


def test_the_recorder_cli_rejects_arguments_that_cannot_mean_anything(tmp_path):
    """`--seconds -5` used to be accepted and just recorded nothing.

    The library entry points raise, but the live source fails first on the
    missing dependency, so the user got told to install a package rather than
    that their own flag was wrong.
    """
    import subprocess
    import sys
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    for flag, val, msg in (("--seconds", "-5", "must be > 0"),
                           ("--seconds", "0", "must be > 0"),
                           ("--rotate-mb", "-1", "must be >= 0"),
                           ("--depth", "7", "must be 5, 10 or 20"),
                           ("--depth", "1", "must be 5, 10 or 20")):
        r = subprocess.run(
            [sys.executable, "flow_mm.py", "record", flag, val,
             "--capture", str(tmp_path / "c.jsonl")],
            capture_output=True, text=True, cwd=root)
        assert r.returncode == 2, f"{flag} {val}: expected exit 2"
        assert msg in r.stderr, f"{flag} {val}: {r.stderr.strip()[:200]}"
    assert not list(tmp_path.iterdir()), "a rejected flag must not create files"


# -- end to end ---------------------------------------------------------------

_STUB_WEBSOCKETS = '''
"""A stand-in for the `websockets` client, speaking Binance's message shapes."""
import asyncio, itertools, json, time


class _WS:
    def __init__(self, url, t0):
        self.url, self.t0, self.n = url, t0, 0
        self.futures = "fstream" in url
        self.ids = itertools.count(1)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def recv(self):
        self.n += 1
        base = self.t0 + self.n * 100
        if self.futures:
            d = {"e": "bookTicker", "E": base, "b": "99.90", "a": "100.10",
                 "B": "3.0", "A": "4.0"}
        elif self.n % 2:
            d = {"e": "depthUpdate", "E": base,
                 "b": [["99.90", "3"]], "a": [["100.10", "4"]]}
        else:
            d = {"e": "trade", "T": base, "t": next(self.ids),
                 "p": "100.00", "q": "1.0", "m": self.n % 4 == 0}
        await asyncio.sleep(0.02)
        return json.dumps({"stream": "s", "data": d})


def connect(url, **kw):
    return _WS(url, int(time.time() * 1000))
'''


def _record_against_stub(cap, seconds=1.0, rotate_mb=0.0):
    """Run `record` with a stub socket installed, in a subprocess.

    A subprocess because `import websockets` happens inside the stream tasks, and
    installing a stub into `sys.modules` in-process would leak into every later
    test -- including the ones that assert the missing-dependency path.
    """
    import subprocess
    import sys
    import tempfile
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    stub = pathlib.Path(tempfile.mkdtemp(prefix="qf-stub-"))
    (stub / "websockets.py").write_text(_STUB_WEBSOCKETS)
    env = dict(os.environ, PYTHONPATH=str(stub))
    return subprocess.run(
        [sys.executable, "flow_mm.py", "record", "--seconds", str(seconds),
         "--rotate-mb", str(rotate_mb), "--capture", str(cap)],
        capture_output=True, text=True, cwd=root, env=env)


def test_a_full_recording_round_trips_through_the_reader(tmp_path):
    """The stub-socket path, which the handler-level tests cannot reach.

    A stub cannot confirm Binance's field names -- that is limitation 9 -- but it
    does confirm everything the tests around it mock out: that a connected socket
    yields rows, that the futures mid reaches every frame rather than being
    dropped, that rotation produces segments the reader reassembles in order, and
    that the sidecar stays quiet when nothing is wrong. A capture that reported
    spurious `futures_disconnect` or `seq_gap` notes here would mean the writers
    are misreading a well-formed message.
    """
    cap = tmp_path / "btc.jsonl"
    r = _record_against_stub(cap, seconds=1.0, rotate_mb=0.001)
    assert r.returncode == 0, r.stderr
    assert "captured" in r.stdout
    assert "WARNING" not in r.stderr, r.stderr
    assert "no rows were captured" not in r.stderr

    parts = sorted(q.name for q in tmp_path.glob("*.jsonl")
                   if not q.name.endswith(".meta.jsonl"))
    assert len(parts) > 1, "rotation did not fire; the reassembly is untested"

    kinds = [n["kind"] for n in read_sidecar(str(cap))]
    assert kinds.count("session_start") == 1
    assert not [k for k in kinds if k in ("futures_disconnect", "seq_gap",
                                          "out_of_order", "bad_book",
                                          "bad_trade", "futures_stale")], kinds
    assert capture_is_continuous(str(cap))

    ds = capture_to_dataset(str(cap))
    assert ds.n_seconds >= 1
    assert int(ds.frames["stale"].sum()) == 0, "a live feed has no gaps"
    assert (ds.frames["futures"] > 0).all(), "futures never reached the frames"
    assert float(ds.frames["mid"].mean()) == pytest.approx(100.0, abs=0.05)


def test_a_record_against_stub_keeps_market_and_sidecar_separate(tmp_path):
    """The sidecar must never be readable as market data.

    `capture_to_dataset` filters `.meta.jsonl` by suffix, and a diagnostic line
    parsed as a book row is a session that fails validation for no visible
    reason. Asserted on the directory read as well as the file read, since the
    two use different globs.
    """
    cap = tmp_path / "btc.jsonl"
    r = _record_against_stub(cap, seconds=0.6)
    assert r.returncode == 0, r.stderr
    for target in (str(cap), str(tmp_path)):
        ds = capture_to_dataset(target)
        assert ds.n_seconds >= 1
        assert not np.isnan(ds.frames["mid"]).any()
    rows = [json.loads(l) for f in tmp_path.glob("*.jsonl")
            for l in f.read_text().splitlines() if l.strip()
            and not f.name.endswith(".meta.jsonl")]
    assert all(r["ts_ms"] > 0 for r in rows)


def test_a_recorded_capture_runs_through_the_unchanged_engine(tmp_path):
    """The payoff: capture -> Dataset -> ReplayMarket -> run_day, no sim."""
    from flow_mm import LADDER, MarketParams, run_day
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        rng = np.random.default_rng(0)
        start = 1_700_000_000_000
        for t in range(120):
            m = 100.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, 120))[t])
            w.book(start + t * 1000, m - 0.05, m + 0.05,
                   float(rng.uniform(5, 20)), float(rng.uniform(5, 20)))
            for _ in range(6):
                s = 1 if rng.random() < 0.5 else -1
                w.trade(start + t * 1000 + int(rng.integers(1, 999)),
                        m + s * 0.02, 1.0, s)
    mkt = ReplayMarket(capture_to_dataset(str(p)), mp=MarketParams())
    mkt.finalize_params(len(mkt))
    out = run_day(mkt, LADDER[-1])
    assert set(out) >= {"pnl", "fills", "spread", "inv_drift", "fees"}
    assert out["fills"] >= 0
    # the exact additive identity from the README still holds on recorded data
    assert out["pnl"] == pytest.approx(
        out["spread"] + out["inv_drift"] + out["hedge_pnl"] - out["fees"]
        - out["hedge_cost"] - out["flatten"], abs=1e-6)


def test_a_price_level_the_quoting_layer_was_not_calibrated_for_fills_nothing(tmp_path):
    """A limitation pinned as a test, because the failure looks like a result.

    The Avellaneda-Stoikov half-spread is `_vterm * var + _kterm`, where `var` is
    a per-second variance in *dollars squared* and `_kterm` is a fixed dollar
    constant. Both terms are therefore in absolute dollars, so the quoted spread
    does not survive a change of price level: in basis points the variance term
    grows linearly with the instrument's price and the constant term shrinks
    inversely. At the ~$100 equities the harness was calibrated for it lands near
    1bp; on a $64,000 instrument it lands in the hundreds, every quote falls
    outside the book, and the day comes back with zero fills and zero PnL.

    Zero fills is the *correct* behaviour for a strategy quoting 500bp wide on a
    market that never moves more than 3bp. What is not acceptable is that it is
    indistinguishable from a strategy that found no edge, which is why this is
    asserted rather than left to be discovered.
    """
    from dataclasses import replace as _replace
    from flow_mm import Engine, LADDER, MarketParams, run_day

    def half_bps(s0, sigma_ann):
        e = Engine(LADDER[-1], _replace(MarketParams(), s0=s0, sigma_ann=sigma_ann))
        return 1e4 * (e._vterm * e.var + e._kterm) / s0

    assert 0.5 < half_bps(100.0, 0.25) < 3.0      # the calibrated regime
    assert half_bps(64000.0, 0.25) > 50.0         # ~100bp: unquotable
    # and it is a units problem, not a volatility problem: matching *relative*
    # vol across a 640x change in price moves the spread by the same 640x
    assert half_bps(64000.0, 0.25) / half_bps(100.0, 0.25) > 50.0

    p = tmp_path / "btc.jsonl"
    with CaptureWriter(str(p)) as w:
        rng = np.random.default_rng(3)
        start = 1_757_000_000_000
        px = 64000.0
        for t in range(300):
            px *= float(np.exp(rng.normal(0, 2e-4)))
            half = px * 3e-4
            w.book(start + t * 1000, px - half, px + half, 20.0, 20.0)
            for _ in range(20):
                s = 1 if rng.random() < 0.5 else -1
                w.trade(start + t * 1000 + int(rng.integers(1, 999)),
                        px + s * half * 0.5, 1.0, s)
    ds = capture_to_dataset(str(p))
    mkt = ReplayMarket(ds, mp=_replace(MarketParams(), s0=64000.0))
    mkt.finalize_params(len(mkt))
    out = run_day(mkt, LADDER[-1])
    # the capture is perfectly good -- it is the quoting layer that cannot use it
    assert ds.n_seconds > 250 and len(ds.trades["px"]) > 5000
    assert out["fills"] == 0


def test_dataset_from_a_capture_is_a_dataset(tmp_path):
    p = tmp_path / "cap.jsonl"
    with CaptureWriter(str(p)) as w:
        _session(w, seconds=8)
    ds = capture_to_dataset(str(p), venue="binance:BTCUSDT")
    assert isinstance(ds, Dataset)
    assert ds.meta["venue"] == "binance:BTCUSDT"
    assert "imbalance" in ds.frames
    assert ds.signed_volume().shape == (7,)


def test_imbalance_comes_from_the_recorded_depth_not_from_the_simulator(tmp_path):
    p = tmp_path / "cap.jsonl"
    start = 1_700_000_000_000
    with CaptureWriter(str(p)) as w:
        for t in range(6):
            lean = t % 2 == 0
            w.book(start + t * 1000, 99.95, 100.05,
                   100.0 if lean else 1.0, 1.0 if lean else 100.0)
    imb = capture_to_dataset(str(p), complete_only=False).frames["imbalance"]
    assert imb[0] == pytest.approx(100 / 101 - 0.5)
    assert imb[1] < 0 < imb[0]
    assert imb.std() > 0
