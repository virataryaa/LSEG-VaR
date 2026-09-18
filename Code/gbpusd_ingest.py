"""
GBP/USD FX Ingest — LSEG Data
==============================
London Cocoa (LCC) settles in GBP while every other VaR commodity (KC, RC,
CC, SB, CT, LSU) settles in USD. This pulls a daily GBP/USD spot rate so
var_monitor.py can convert LCC prices to USD before any VaR math runs.

Usage:
    python gbpusd_ingest.py            # incremental update
    python gbpusd_ingest.py --full     # full pull from 2009-01-01

Saves to: ../Database/gbpusd.parquet  (columns: Date, GBPUSD)
"""

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import lseg.data as ld

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
OUT_FILE   = DB_DIR / "gbpusd.parquet"
FULL_START = "2009-01-01"
RIC        = "GBP="


def fetch_fx(start: str, end: str, retries: int = 3, delay: int = 30) -> pd.DataFrame:
    log.info("Fetching GBP/USD (%s) from %s", RIC, start)
    for attempt in range(1, retries + 1):
        try:
            df = ld.get_history(
                universe=[RIC], fields=["MID_PRICE"],
                start=start, end=end, interval="daily",
            )
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            df.columns = ["GBPUSD"]
            return df
        except Exception as e:
            if attempt < retries:
                log.warning("fetch_fx attempt %d/%d failed: %s — retrying in %ds", attempt, retries, e, delay)
                time.sleep(delay)
            else:
                log.error("fetch_fx failed after %d attempts: %s", retries, e)
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full pull from 2009-01-01")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("GBP/USD Ingest (LSEG) | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    DB_DIR.mkdir(parents=True, exist_ok=True)

    if args.full or not OUT_FILE.exists():
        start = FULL_START
        log.info("Mode: FULL from %s", start)
    else:
        existing = pd.read_parquet(OUT_FILE, columns=["Date"])
        latest = pd.to_datetime(existing["Date"]).max()
        start = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        log.info("Mode: INCREMENTAL from %s", start)

    end = datetime.date.today().isoformat()

    ld.open_session()
    try:
        new_df = fetch_fx(start, end).reset_index()
        new_df["GBPUSD"] = pd.to_numeric(new_df["GBPUSD"], errors="coerce")
        new_df = new_df.dropna(subset=["GBPUSD"])
        log.info("Rows fetched: %d", len(new_df))

        if OUT_FILE.exists() and not args.full:
            old_df = pd.read_parquet(OUT_FILE)
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            combined = (
                pd.concat([old_df, new_df])
                .drop_duplicates(subset=["Date"], keep="last")
                .sort_values("Date")
                .reset_index(drop=True)
            )
        else:
            combined = new_df.sort_values("Date").reset_index(drop=True)

        today = pd.Timestamp.today().normalize()
        combined = combined[combined["Date"] < today].reset_index(drop=True)

        combined.to_parquet(OUT_FILE, engine="pyarrow", index=False)
        log.info("Saved: %s  (%d rows)", OUT_FILE.name, len(combined))
        log.info("=" * 60)
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
