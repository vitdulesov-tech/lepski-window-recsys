"""Шаг 4 фазы 0: короткий EDA. Печатает числа, от которых зависит постановка задачи.

Запуск:  python -m src.eda
"""
from __future__ import annotations

import polars as pl

from src.config import load_config, processed_path


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    cfg = load_config()
    thr = cfg["task"]["label_played_ratio_threshold"]
    ev = pl.read_parquet(processed_path(cfg, "events.parquet"))

    section("Размер")
    print(f"событий: {ev.height:,}; пользователей: {ev['uid'].n_unique():,}; "
          f"треков: {ev['item_id'].n_unique():,}; артистов: {ev['artist_id'].n_unique():,}; "
          f"альбомов: {ev['album_id'].n_unique():,}")
    print(f"дней покрыто: {ev['day'].max() + 1}")

    section("Типы событий и источник")
    print(ev.group_by("event_type", "is_organic").len().sort("event_type", "is_organic"))

    section("Прослушивания: played_ratio_pct")
    listens = ev.filter(pl.col("event_type") == "listen")
    print(listens.select(
        pl.col("played_ratio_pct").quantile(q).alias(f"q{int(q*100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)
    ))
    print(f"доля played_ratio_pct > 100: {(listens['played_ratio_pct'] > 100).mean():.4f}")
    print(f"null played_ratio среди listen: {listens['played_ratio_pct'].null_count()}")

    section(f"Баланс будущей метки (played_ratio >= {thr}) по источнику")
    print(
        listens.with_columns((pl.col("played_ratio_pct") >= thr).alias("listen_plus"))
        .group_by("is_organic")
        .agg(pl.len().alias("n"), pl.col("listen_plus").mean().alias("share_listen_plus"))
        .sort("is_organic")
    )

    section("Длина истории на пользователя")
    per_user = ev.group_by("uid").agg(
        pl.len().alias("n_events"),
        pl.col("item_id").n_unique().alias("n_items"),
        pl.col("artist_id").n_unique().alias("n_artists"),
        (pl.col("day").max() - pl.col("day").min() + 1).alias("active_span_days"),
    )
    print(per_user.select(
        [pl.col(c).quantile(q).alias(f"{c}_q{int(q*100)}") for c in ("n_events", "n_items") for q in (0.1, 0.5, 0.9)]
    ))
    print(per_user.select(pl.col("active_span_days").quantile(0.5).alias("span_median"),
                          pl.col("active_span_days").min().alias("span_min")))

    section("Повторность (среди прослушиваний, в порядке времени пользователя)")
    rep = listens.sort(["uid", "ts"]).with_columns(
        (~pl.col("item_id").is_first_distinct().over("uid")).alias("repeat_item"),
        (~pl.col("artist_id").is_first_distinct().over("uid")).alias("repeat_artist"),
        (~pl.col("album_id").is_first_distinct().over("uid")).alias("repeat_album"),
    )
    print(rep.group_by("is_organic").agg(
        pl.col("repeat_item").mean(), pl.col("repeat_album").mean(), pl.col("repeat_artist").mean()
    ).sort("is_organic"))
    print("Если repeat_item для ленты мал, а repeat_artist велик — косвенные связи важнее прямых.")

    section("Суточный профиль (часы от начала суток в неизвестной зоне) и недельный (day mod 7)")
    print(ev.with_columns(((pl.col("ts") % 86400) // 3600).alias("hour"))
          .group_by("hour").len().sort("hour").to_series(1).to_list())
    print(ev.with_columns((pl.col("day") % 7).alias("dow")).group_by("dow").len().sort("dow").to_series(1).to_list())

    section("Активность по дням (события в день, первые/последние 10 дней)")
    daily = ev.group_by("day").len().sort("day")
    print("начало:", daily.head(10).to_series(1).to_list())
    print("конец: ", daily.tail(10).to_series(1).to_list())
    print("Если в конце данных резкий спад — последние дни могут быть неполными; тогда test сдвинуть.")


if __name__ == "__main__":
    main()
