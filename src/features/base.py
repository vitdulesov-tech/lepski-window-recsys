"""Примитивы для point-in-time признаков.

Контракт любого признака:  feat(events, rows) -> DataFrame[row_id, <feature columns>]
где для строки с моментом t используются только события с ts < t (строго).

Строгое неравенство реализовано через asof-join с запросом ts - 1: события в той же
5-секундной корзине, что и целевое (включая само целевое), в признак не попадают.
Это консервативно и соответствует постановке из PDF (t_e < t).
"""
from __future__ import annotations

from collections.abc import Callable

import polars as pl

FeatureFn = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def pit_cumcount(
    events: pl.DataFrame,
    rows: pl.DataFrame,
    key_cols: list[str],
    name: str,
    where: pl.Expr | None = None,
) -> pl.DataFrame:
    """Число событий (удовлетворяющих where) с теми же key_cols, произошедших строго до ts строки."""
    ev = events if where is None else events.filter(where)
    cum = (
        ev.group_by([*key_cols, "ts"]).len()
        .sort([*key_cols, "ts"])
        .with_columns(pl.col("len").cum_sum().over(key_cols).alias(name))
        .drop("len")
        .sort("ts")
    )
    q = rows.select("row_id", *key_cols, (pl.col("ts") - 1).alias("ts_q")).sort("ts_q")
    res = q.join_asof(cum, left_on="ts_q", right_on="ts", by=key_cols, strategy="backward")
    return res.select("row_id", pl.col(name).fill_null(0)).sort("row_id")


def pit_last_ts(
    events: pl.DataFrame,
    rows: pl.DataFrame,
    key_cols: list[str],
    name: str,
    where: pl.Expr | None = None,
) -> pl.DataFrame:
    """Время (в секундах) с последнего события по key_cols до ts строки; null, если событий не было."""
    ev = events if where is None else events.filter(where)
    last = (
        ev.select(*key_cols, "ts").unique().sort("ts")
        .with_columns(pl.col("ts").alias("ts_prev"))   # явная копия: ключ asof-join в вывод не попадает
        .rename({"ts": "ts_key"})
    )
    q = rows.select("row_id", *key_cols, "ts", (pl.col("ts") - 1).alias("ts_q")).sort("ts_q")
    res = q.join_asof(last, left_on="ts_q", right_on="ts_key", by=key_cols, strategy="backward")
    return res.select("row_id", (pl.col("ts") - pl.col("ts_prev")).alias(name)).sort("row_id")


# --- Тривиальный признак для проверки инфраструктуры и теста на утечки ----------------

def feat_user_listen_count(events: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Сколько прослушиваний (любых) у пользователя было до момента строки."""
    return pit_cumcount(events, rows, ["uid"], "u_listen_cnt_all",
                        where=pl.col("event_type") == "listen")
