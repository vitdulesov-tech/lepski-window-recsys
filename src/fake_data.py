"""Синтетические данные в схеме Yambda — для прогона пайплайна без скачивания.
Пишет файлы в data/raw по тем же путям, что и настоящие.

Запуск:  python -m src.fake_data && python -m src.data --local && python -m src.eda ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.config import load_config


def main(n_users: int = 300, n_items: int = 5000, n_days: int = 120, seed: int = 0) -> None:
    cfg = load_config()
    raw = Path(cfg["general"]["raw_dir"])
    rng = np.random.default_rng(seed)
    unit = cfg["data"]["ts_unit_seconds"]

    artists = rng.integers(0, 400, size=n_items)
    albums = rng.integers(0, 1500, size=n_items)

    recs = []
    for uid in range(n_users):
        n = int(rng.integers(200, 2000))
        # Пользователь слушает в основном «свой» набор треков — чтобы была повторность.
        fav = rng.choice(n_items, size=int(rng.integers(20, 120)), replace=False)
        items = np.where(rng.random(n) < 0.7, rng.choice(fav, n), rng.integers(0, n_items, n))
        ts = np.sort(rng.integers(0, n_days * 86400 // unit, size=n))
        ev_type = rng.choice(["listen", "like", "dislike"], size=n, p=[0.92, 0.06, 0.02])
        organic = rng.integers(0, 2, size=n).astype(np.uint8)
        is_listen = ev_type == "listen"
        ratio_raw = rng.integers(1, 120, size=n)
        length_raw = rng.integers(60, 400, size=n)
        ratio = [int(r) if l else None for r, l in zip(ratio_raw, is_listen)]
        length = [int(r) if l else None for r, l in zip(length_raw, is_listen)]
        recs.append(pl.DataFrame({
            "uid": np.full(n, uid, dtype=np.uint32), "item_id": items.astype(np.uint32),
            "timestamp": ts.astype(np.uint32), "is_organic": organic, "event_type": ev_type.tolist(),
            "played_ratio_pct": pl.Series(ratio, dtype=pl.UInt16),
            "track_length_seconds": pl.Series(length, dtype=pl.UInt32),
        }))
    ev = pl.concat(recs)

    for key in ("events_file", "artist_file", "album_file"):
        (raw / cfg["data"][key]).parent.mkdir(parents=True, exist_ok=True)
    ev.write_parquet(raw / cfg["data"]["events_file"])
    pl.DataFrame({"artist_id": artists.astype(np.uint32), "item_id": np.arange(n_items, dtype=np.uint32)}) \
        .write_parquet(raw / cfg["data"]["artist_file"])
    pl.DataFrame({"album_id": albums.astype(np.uint32), "item_id": np.arange(n_items, dtype=np.uint32)}) \
        .write_parquet(raw / cfg["data"]["album_file"])
    print(f"[fake] {ev.height:,} событий, {n_users} пользователей -> {raw}")


if __name__ == "__main__":
    main()
