"""One-time maintenance: collapse duplicate (event_id, object_id) rows in the
LanceDB entity_crops table, keeping a single row per entity (with its existing
vector). Safe and idempotent — running it again on a clean table is a no-op.

Usage:  python scripts/dedup_lancedb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb
from app.config import settings


def main() -> None:
    db = lancedb.connect(str(settings.paths.lancedb_path))
    try:
        table = db.open_table(settings.rag.table_name)
    except Exception:
        print(f"No table '{settings.rag.table_name}' at {settings.paths.lancedb_path} — nothing to do.")
        return
    df = table.to_pandas()
    total_before = len(df)
    distinct = df.drop_duplicates(subset=["event_id", "object_id"])
    total_after = len(distinct)

    print(f"rows before        : {total_before}")
    print(f"distinct entities  : {total_after}")
    print(f"duplicate rows      : {total_before - total_after}")

    if total_before == total_after:
        print("Table is already clean. No changes made.")
        return

    # Only touch events that actually have duplicates.
    counts = df.groupby(["event_id", "object_id"]).size()
    polluted_events = sorted({ev for (ev, _obj), n in counts.items() if n > 1})
    print(f"events to rewrite  : {polluted_events}")

    for ev in polluted_events:
        keep = distinct[distinct["event_id"] == ev].to_dict("records")
        table.delete(f"event_id = '{ev}'")
        table.add(keep)
        print(f"  {ev}: rewrote {len(keep)} unique rows")

    print(f"\nrows after         : {table.count_rows()}  (expected {total_after})")
    print("Done.")


if __name__ == "__main__":
    main()
