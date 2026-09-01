# VaR — Interim Migration

Interim replacement for `ICEBREAKER/VaR`. This project is a pure downstream
consumer — it holds no ingest logic of its own, only a daily sync step that
copies parquets from two other LSEG projects. Ported unchanged
except for repointing that sync's source paths.

## What's here

- **`Dashboard/var_monitor.py`** — copied verbatim from the ICEBREAKER
  source. Two tabs: **Parametric VaR** (1-day, 99% confidence, 20D/60D/120D
  rolling windows, per-commodity and combined-book views, vol percentile bar,
  Year×Month heatmap) and **Portfolio VaR — Monte Carlo** (Cholesky-correlated
  simulation, normal or Student-t fat tails, component CVaR breakdown,
  editable position book saved to `Dashboard/saved_positions.json`).
- **`Code/ingest.py`** — copies 7 `rollex_{comm}.parquet` files from
  `LSEG/Rollex/Database` and 7 `{comm}_futures.parquet` files
  from `LSEG/Futures/Database` into this project's own
  `Database/`, mtime-gated (skips a file if the destination is already newer).
  Both source projects run their own daily automators, so this step is a
  cheap copy, not a rebuild.
- **`Database/`** — 7 commodities: KC, RC (Robusta), CC, LCC, SB, CT, LSU.
- **`Automator/`** — `run.bat` (ingest → git commit/push → email via
  Outlook), `send_mail.py`.

## Running it

```bash
python Code/ingest.py              # sync parquets from Rollex + Futures
streamlit run Dashboard/var_monitor.py
```

No LSEG/API session required — this project only reads parquets already
produced by the Rollex and Futures migrations.
