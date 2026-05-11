"""Snapshot the live Postgres tables (and SPY benchmark) into Parquet files.

Run this ONCE after the validation window closes. The dashboard then reads
those Parquet files in deployed (snapshot) mode and never needs DB access.

Usage:
    python3 scripts/snapshot_db.py
    python3 scripts/snapshot_db.py --start 2026-03-15 --out dashboard/data

Reads connection from one of (in priority order):
    1. --database-url CLI arg
    2. DATABASE_URL env var
    3. dashboard/.streamlit/secrets.toml [connections.postgresql] block
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
from typing import Optional

try:
    import tomllib  # Python ≥ 3.11
except ModuleNotFoundError:  # pragma: no cover — Python 3.10 and below
    import tomli as tomllib  # type: ignore[no-redef]

import pandas as pd
from sqlalchemy import create_engine, text


REPO_ROOT      = pathlib.Path(__file__).resolve().parent.parent
DASHBOARD_DIR  = REPO_ROOT / "dashboard"
SECRETS_FILE   = DASHBOARD_DIR / ".streamlit" / "secrets.toml"
DEFAULT_OUTPUT = DASHBOARD_DIR / "data"
DEFAULT_START  = "2026-03-15"
BENCHMARK      = "SPY"
START_EQUITY   = 100_000.0


def _rel_to_repo(path: pathlib.Path) -> pathlib.Path:
    """Print-friendly path; works when ``--out`` is outside the repo."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Connection resolution
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_database_url(cli_url: Optional[str]) -> str:
    if cli_url:
        return cli_url
    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    if SECRETS_FILE.exists():
        with SECRETS_FILE.open("rb") as fh:
            data = tomllib.load(fh)
        pg = (data.get("connections") or {}).get("postgresql")
        if pg:
            user = pg["username"]
            pw   = pg["password"]
            host = pg["host"]
            port = pg.get("port", 5432)
            db   = pg["database"]
            return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    raise RuntimeError(
        "No database URL found. Pass --database-url, set DATABASE_URL, "
        f"or fill in {SECRETS_FILE}."
    )


def _resolve_alpaca_keys() -> tuple[str, str]:
    key    = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or ""
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or ""
    if (not key or not secret) and SECRETS_FILE.exists():
        with SECRETS_FILE.open("rb") as fh:
            data = tomllib.load(fh)
        key    = key    or data.get("ALPACA_API_KEY", "")
        secret = secret or data.get("ALPACA_SECRET_KEY", "")
    return key, secret


# ─────────────────────────────────────────────────────────────────────────────
# Per-table queries
# ─────────────────────────────────────────────────────────────────────────────
TABLE_QUERIES = {
    "portfolio_snapshots": """
        SELECT arm_id, cash, equity, total_value, captured_at
        FROM portfolio_snapshots
        WHERE captured_at >= :start
        ORDER BY arm_id, captured_at
    """,
    "trades": """
        SELECT id, arm_id, symbol, side, qty, price, status, created_at
        FROM trades
        WHERE created_at >= :start
        ORDER BY arm_id, created_at
    """,
    "agent_decision_logs": """
        SELECT id, arm_id, cycle_ts, agent, reasoning,
               orders_proposed, orders_executed, orders_blocked,
               memory_packet_in, sentiment_per_symbol,
               llm_tokens_used, llm_cost_usd,
               post_mortem, created_at
        FROM agent_decision_logs
        WHERE cycle_ts >= :start
        ORDER BY arm_id, cycle_ts
    """,
    "market_snapshots": """
        SELECT symbol, price, captured_at
        FROM market_snapshots
        WHERE captured_at >= :start
        ORDER BY symbol, captured_at
    """,
    "sentiment_snapshots": """
        SELECT cycle_ts, symbol, label, score, confidence, created_at
        FROM sentiment_snapshots
        WHERE cycle_ts >= :start
        ORDER BY cycle_ts, symbol
    """,
}


def _dump_table(engine, table: str, query: str, start: str, out_dir: pathlib.Path) -> int:
    print(f"  • {table}…", end=" ", flush=True)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"start": start})
    out = out_dir / f"{table}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df):,} rows → {_rel_to_repo(out)}")
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark dump (calls Alpaca)
# ─────────────────────────────────────────────────────────────────────────────
def _dump_spy_benchmark(start: str, end: str, out_dir: pathlib.Path) -> int:
    print(f"  • spy_benchmark…", end=" ", flush=True)
    key, secret = _resolve_alpaca_keys()
    if not key or not secret:
        print("SKIPPED (no Alpaca credentials available)")
        return 0
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame
    except ImportError:
        print("SKIPPED (alpaca-py not installed)")
        return 0

    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=[BENCHMARK],
        timeframe=TimeFrame.Day,
        start=dt.datetime.fromisoformat(start),
        end=dt.datetime.fromisoformat(end),
    )
    bars = client.get_stock_bars(req)
    rows = [{"date": b.timestamp.date(), "close": float(b.close)}
            for b in bars[BENCHMARK]]
    if not rows:
        print("SKIPPED (Alpaca returned no bars)")
        return 0
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    p0 = df["close"].iloc[0]
    df["equity"]         = START_EQUITY * df["close"] / p0
    df["cum_return_pct"] = (df["close"] / p0 - 1) * 100
    out = out_dir / "spy_benchmark.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows → {_rel_to_repo(out)}")
    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-url", default=None,
                    help="Override DATABASE_URL / secrets.toml.")
    ap.add_argument("--start", default=DEFAULT_START,
                    help=f"Start date (inclusive). Default: {DEFAULT_START}.")
    ap.add_argument("--end", default=None,
                    help="End date for SPY benchmark. Default: today UTC.")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT),
                    help=f"Output directory. Default: {DEFAULT_OUTPUT}.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_url = _resolve_database_url(args.database_url)
    engine = create_engine(db_url, pool_pre_ping=True)

    end = args.end or dt.datetime.utcnow().date().isoformat()
    print(f"Snapshotting → {_rel_to_repo(out_dir)}  "
          f"(window {args.start} → {end})")

    total = 0
    for table, query in TABLE_QUERIES.items():
        total += _dump_table(engine, table, query, args.start, out_dir)
    total += _dump_spy_benchmark(args.start, end, out_dir)

    # Manifest with snapshot metadata
    manifest = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "window_start": args.start,
        "window_end":   end,
        "tables": {
            t: int((out_dir / f"{t}.parquet").stat().st_size)
            for t in (*TABLE_QUERIES.keys(), "spy_benchmark")
            if (out_dir / f"{t}.parquet").exists()
        },
    }
    import json
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest → {_rel_to_repo(out_dir / 'manifest.json')}")
    print(f"Total rows: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
