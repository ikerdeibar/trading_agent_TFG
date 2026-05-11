# Frozen dashboard snapshot

These Parquet files are the **source of truth** the deployed dashboard reads
in `DASHBOARD_SOURCE=snapshot` mode (the default). They are produced from
the live Postgres tables by:

```bash
python scripts/snapshot_db.py
```

| File | Source table | Cols of interest |
|---|---|---|
| `portfolio_snapshots.parquet` | `portfolio_snapshots` | `arm_id, equity, captured_at` |
| `trades.parquet`              | `trades`              | `arm_id, symbol, side, qty, price, created_at` |
| `agent_decision_logs.parquet` | `agent_decision_logs` | `arm_id, cycle_ts, llm_cost_usd, llm_tokens_used, orders_*` |
| `market_snapshots.parquet`    | `market_snapshots`    | `symbol, price (→ mid), captured_at` |
| `sentiment_snapshots.parquet` | `sentiment_snapshots` | `cycle_ts, symbol, label, score` |
| `spy_benchmark.parquet`       | Alpaca daily SPY bars | `date, close, equity, cum_return_pct` |
| `manifest.json`               | (generated)           | snapshot date + window + per-table file sizes |

To re-snapshot after new experiments, see `dashboard/DEPLOY.md`.
