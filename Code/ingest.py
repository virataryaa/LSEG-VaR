"""
VaR Ingest — Sync
Copies rollex + futures parquets from their source databases into VaR/Database/.
Run daily after Rollex and Futures ingests have completed.
"""
import shutil
import logging
import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

HERE    = Path(__file__).parent.parent
OUT_DIR = HERE / "Database"

ROLLEX_SRC  = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Database")
FUTURES_SRC = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Futures\Database")

COMMS = ["KC", "RC", "CC", "LCC", "SB", "CT", "LSU"]

SOURCES = (
    [(ROLLEX_SRC  / f"rollex_{c}.parquet",  OUT_DIR / f"rollex_{c}.parquet")  for c in COMMS] +
    [(FUTURES_SRC / f"{c.lower()}_futures.parquet", OUT_DIR / f"{c.lower()}_futures.parquet") for c in COMMS]
)


def sync():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []

    for src, dst in SOURCES:
        if not src.exists():
            log.warning(f"  MISSING: {src.name}")
            failed.append(src.name)
            continue
        src_mtime = src.stat().st_mtime
        dst_mtime = dst.stat().st_mtime if dst.exists() else 0
        if src_mtime > dst_mtime:
            shutil.copy2(src, dst)
            log.info(f"  Copied : {src.name}")
        else:
            log.info(f"  Up to date: {src.name}")
        ok += 1

    log.info(f"Sync complete — {ok} files OK, {len(failed)} missing")
    if failed:
        log.warning(f"Missing files: {failed}")
        raise RuntimeError(f"Missing source files: {failed}")


if __name__ == "__main__":
    log.info("=" * 50 + f"\nVaR Ingest | {datetime.date.today()}\n" + "=" * 50)
    sync()
    log.info("=" * 50)
