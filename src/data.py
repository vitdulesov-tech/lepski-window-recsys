"""Шаги 2–3 фазы 0: скачивание Yambda-50M, подвыборка пользователей с ПОЛНОЙ историей,
присоединение артиста и альбома, сохранение единой таблицы событий.

Запуск:  python -m src.data
Результат: data/processed/events.parquet со столбцами
    uid, item_id, artist_id, album_id, ts (секунды), day (целый день от начала данных),
    event_type (str), is_organic (0/1), played_ratio_pct, track_length_seconds
Отсортировано по (uid, ts).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

from src.config import load_config, processed_path


def download(cfg: dict) -> dict[str, Path]:
    """Скачивает три файла в data/raw с сохранением структуры путей репозитория."""
    raw_dir = cfg["general"]["raw_dir"]
    repo = cfg["data"]["hf_repo"]
    paths = {}
    for key in ("events_file", "artist_file", "album_file"):
        fname = cfg["data"][key]
        t0 = time.time()
        local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=raw_dir)
        print(f"[download] {fname} -> {local} ({time.time() - t0:.0f} s)")
        paths[key] = Path(local)
    return paths


def _load_mapping(path: Path, entity: str) -> pl.DataFrame:
    """Маппинг item_id -> artist_id / album_id. Имена столбцов проверяем, а не предполагаем."""
    df = pl.read_parquet(path)
    cols = df.columns
    assert "item_id" in cols, f"{path}: нет столбца item_id, есть {cols}"
    other = [c for c in cols if c != "item_id"]
    assert len(other) == 1, f"{path}: ожидался ровно один столбец кроме item_id, есть {cols}"
    df = df.rename({other[0]: f"{entity}_id"}).select("item_id", f"{entity}_id")
    # У трека может быть несколько артистов/альбомов. Берём один (минимальный id) —
    # детерминированно и достаточно для признаков уровня «артист».
    n_before = df.height
    df = df.group_by("item_id").agg(pl.col(f"{entity}_id").min())
    print(f"[mapping] {entity}: {n_before} строк -> {df.height} уникальных item_id")
    return df


def build_events(cfg: dict, paths: dict[str, Path]) -> pl.DataFrame:
    seed = cfg["general"]["seed"]
    n_users = cfg["data"]["n_users"]
    unit = cfg["data"]["ts_unit_seconds"]

    lf = pl.scan_parquet(paths["events_file"])
    print("[events] схема исходного файла:", lf.collect_schema())

    all_uids = np.sort(lf.select("uid").unique().collect()["uid"].to_numpy())
    print(f"[events] всего пользователей в срезе: {len(all_uids)}")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(all_uids, size=min(n_users, len(all_uids)), replace=False))
    print(f"[events] выбрано пользователей: {len(chosen)} (seed={seed})")

    t0 = time.time()
    ev = (
        lf.filter(pl.col("uid").is_in(chosen.tolist()))
        .with_columns(
            pl.col("event_type").cast(pl.Utf8),
            (pl.col("timestamp").cast(pl.Int64) * unit).alias("ts"),
        )
        .collect()
    )
    print(f"[events] событий после фильтра: {ev.height:,} ({time.time() - t0:.0f} s)")

    artist = _load_mapping(paths["artist_file"], "artist")
    album = _load_mapping(paths["album_file"], "album")
    ev = ev.join(artist, on="item_id", how="left").join(album, on="item_id", how="left")
    miss_a = ev["artist_id"].null_count()
    miss_b = ev["album_id"].null_count()
    print(f"[events] событий без артиста: {miss_a:,}, без альбома: {miss_b:,}")

    ts_min = ev["ts"].min()
    ev = (
        ev.with_columns(((pl.col("ts") - ts_min) // 86400).cast(pl.Int32).alias("day"))
        .select(
            "uid", "item_id", "artist_id", "album_id", "ts", "day",
            "event_type", "is_organic", "played_ratio_pct", "track_length_seconds",
        )
        .sort(["uid", "ts"])
    )
    return ev


def local_paths(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["general"]["raw_dir"])
    return {k: raw_dir / cfg["data"][k] for k in ("events_file", "artist_file", "album_file")}


def main() -> None:
    cfg = load_config()
    # python -m src.data --local  : не скачивать, использовать файлы, уже лежащие в data/raw
    paths = local_paths(cfg) if "--local" in sys.argv else download(cfg)
    ev = build_events(cfg, paths)
    out = processed_path(cfg, "events.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.write_parquet(out)
    print(f"[done] {out}: {ev.height:,} событий, {ev['uid'].n_unique()} пользователей, "
          f"{ev['day'].max() + 1} дней")


if __name__ == "__main__":
    main()
