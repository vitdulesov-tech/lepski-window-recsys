"""Шаги 5–6 фазы 0: временное разбиение и фиксированная таблица строк (u, i, t, y).

Строка = прослушивание из рекомендательной ленты (is_organic = 0).
Метка  = played_ratio_pct >= порога (Listen+).

Разбиение глобальное по времени, от последнего дня данных:
    train: [0, train_end)   gap   val: [val_start, val_end)   gap   test: [test_start, last_day]

Запуск:  python -m src.rows
Результат: data/processed/rows.parquet со столбцами
    row_id, uid, item_id, artist_id, album_id, ts, day, split, y, played_ratio_pct, track_length_seconds
"""
from __future__ import annotations

import json

import polars as pl

from src.config import load_config, processed_path


def split_bounds(cfg: dict, last_day: int) -> dict[str, int]:
    s = cfg["split"]
    test_start = last_day - s["test_days"] + 1
    val_end = test_start - s["gap_days"]          # исключительно
    val_start = val_end - s["val_days"]
    train_end = val_start - s["gap_days"]         # исключительно
    train_start = 0 if s["train_max_days"] == 0 else max(0, train_end - s["train_max_days"])
    b = dict(train_start=train_start, train_end=train_end, val_start=val_start,
             val_end=val_end, test_start=test_start, last_day=last_day)
    assert train_start < train_end < val_start < val_end <= test_start <= last_day, b
    return b


def assign_split(day: pl.Expr, b: dict[str, int]) -> pl.Expr:
    return (
        pl.when((day >= b["train_start"]) & (day < b["train_end"])).then(pl.lit("train"))
        .when((day >= b["val_start"]) & (day < b["val_end"])).then(pl.lit("val"))
        .when(day >= b["test_start"]).then(pl.lit("test"))
        .otherwise(pl.lit("gap"))
    )


def main() -> None:
    cfg = load_config()
    t = cfg["task"]
    ev = pl.read_parquet(processed_path(cfg, "events.parquet"))

    b = split_bounds(cfg, int(ev["day"].max()))
    print("[split] границы (дни от начала данных):", b)

    rows = (
        ev.filter(
            (pl.col("event_type") == t["row_event_type"])
            & (pl.col("is_organic") == t["row_is_organic"])
            & pl.col("played_ratio_pct").is_not_null()
        )
        .with_columns(
            assign_split(pl.col("day"), b).alias("split"),
            (pl.col("played_ratio_pct") >= t["label_played_ratio_threshold"]).cast(pl.Int8).alias("y"),
        )
        .filter(pl.col("split") != "gap")
        .sort(["uid", "ts"])
        .with_row_index("row_id")
        .select("row_id", "uid", "item_id", "artist_id", "album_id", "ts", "day", "split", "y",
                "played_ratio_pct", "track_length_seconds")
    )

    summary = (
        rows.group_by("split")
        .agg(pl.len().alias("n_rows"), pl.col("uid").n_unique().alias("n_users"),
             pl.col("y").mean().alias("pos_rate"))
        .sort("split")
    )
    print(summary)

    # Пользователи теста без истории в train — важно знать, насколько их много.
    train_users = set(rows.filter(pl.col("split") == "train")["uid"].to_list())
    for sp in ("val", "test"):
        users = set(rows.filter(pl.col("split") == sp)["uid"].to_list())
        cold = len(users - train_users)
        print(f"[split] {sp}: пользователей без строк в train: {cold} из {len(users)}")

    out = processed_path(cfg, "rows.parquet")
    rows.write_parquet(out)
    with open(processed_path(cfg, "split_bounds.json"), "w") as f:
        json.dump(b, f, indent=2)
    print(f"[done] {out}: {rows.height:,} строк")


if __name__ == "__main__":
    main()
