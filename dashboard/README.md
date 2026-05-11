# Thesis dashboard (Streamlit)

Static **snapshot** mode reads frozen Parquet files under [`data/`](data/) so the app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) without a database.

## Regenerate the frozen data

From the repo root (local Postgres must be running and populated):

```bash
export DATABASE_URL="postgresql+psycopg2://..."
python scripts/snapshot_db.py --start 2026-03-15
```

Optional SPY benchmark rows require Alpaca API keys (see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)).

## Run locally

```bash
cd dashboard
pip install -r requirements.txt
export DASHBOARD_SOURCE=snapshot   # default
streamlit run home.py
```

Live Postgres mode: set `DASHBOARD_SOURCE=live` and configure `[connections.postgresql]` in `.streamlit/secrets.toml`.

## Deploy

See [`DEPLOY.md`](DEPLOY.md).
