# Deploying the dashboard

The deployed dashboard runs in **snapshot mode** — it reads from
`dashboard/data/*.parquet` and never opens a database connection. This is
what the jury sees from a QR code during the defense.

## TL;DR

```bash
# 1. Refresh the snapshot once (locally, with DB access)
python scripts/snapshot_db.py

# 2. Commit the snapshot files
git add dashboard/data/*.parquet dashboard/data/manifest.json
git commit -m "data: refresh dashboard snapshot"
git push

# 3. Deploy via Streamlit Community Cloud (one-time setup, see below)
```

## Streamlit Community Cloud — one-time setup

1. Go to <https://share.streamlit.io> and sign in with the GitHub account
   that owns this repo.
2. Click **New app** → pick this repo → branch `main` → main file path
   `dashboard/home.py` → Python `3.11`.
3. **Advanced settings** → set environment variable:
   ```
   DASHBOARD_SOURCE=snapshot
   ```
   (No secrets are needed for snapshot mode; leave the secrets editor blank.)
4. Click **Deploy**. First build takes ~3 min (installs `dashboard/requirements.txt`).
5. Note the public URL — typically
   `https://<app-name>-<hash>.streamlit.app`.

## QR code for the defense

Install **`segno`** + **`reportlab`** locally once (`pip install segno reportlab`) —
they are also listed under **`optional-dependencies.dev`** in the repo root
`pyproject.toml` (`pip install -e ".[dev]"`).

Once you have the deployed URL:

```bash
python scripts/make_qr.py "https://<your-app>.streamlit.app"
```

Outputs:
- `presentation/qr/dashboard_qr.png` — high-res QR for slides.
- `presentation/qr/dashboard_qr_card.pdf` — A6 printable card with QR + URL +
  a one-line caption. Print and place on the table during the defense.

## Refreshing the snapshot later

If you re-run experiments and want the dashboard to reflect new data:

```bash
python scripts/snapshot_db.py
git add dashboard/data/*.parquet dashboard/data/manifest.json
git commit -m "data: refresh dashboard snapshot"
git push
```

Streamlit Community Cloud auto-redeploys on every push to `main` (~30 s).

## Local development

Snapshot mode (default — works offline, no secrets needed):
```bash
streamlit run dashboard/home.py
```

Live mode against the local/Cloud Postgres (requires
`dashboard/.streamlit/secrets.toml` filled in from
`secrets.toml.example`):
```bash
DASHBOARD_SOURCE=live streamlit run dashboard/home.py
```

## What gets deployed

The dashboard is fully self-contained under `dashboard/`. Streamlit Cloud
ignores everything else in the repo, but the Parquet files in
`dashboard/data/` are committed so they ship with the deploy.

```
dashboard/
├── README.md               ← snapshot regenerate + local run
├── home.py                 ← landing page (entry point)
├── pages/
│   ├── 1_Performance.py
│   ├── 2_Factorial_Results.py
│   ├── 3_Trades_and_Risk.py
│   ├── 4_Cost_and_ROI.py
│   ├── 5_Sentiment.py
│   ├── 6_Session_Explorer.py
│   └── 9_Methodology.py
├── data/                   ← frozen snapshot (Parquet)
├── db.py                   ← shared loaders + computations + UI helpers
├── requirements.txt        ← deploy-time deps
├── .streamlit/
│   ├── config.toml         ← dark theme
│   └── secrets.toml.example
└── DEPLOY.md               ← this file
```
