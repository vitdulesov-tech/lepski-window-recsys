"""Шаг 7 фазы 0: тест на утечки.

Для случайных строк (u, i, t) признак считается дважды:
  1) на полном логе событий (как в пайплайне),
  2) на логе, из которого удалены ВСЕ события с ts >= t (всех пользователей).
Значения обязаны совпасть. Если признак «видит» будущее или само целевое событие —
тест упадёт. Запускать на каждом новом признаке.

Запуск:  python -m src.leakage            (проверяет все зарегистрированные признаки)
"""
from __future__ import annotations

import math
import sys

import polars as pl

from src.config import load_config, processed_path
from src.features.base import FeatureFn, feat_user_listen_count

# Сюда добавлять каждый новый признак/блок признаков.
REGISTRY: dict[str, FeatureFn] = {
    "user_listen_count": feat_user_listen_count,
}


def _values_equal(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def check_leakage(name: str, feat: FeatureFn, events: pl.DataFrame, rows: pl.DataFrame,
                  n_rows: int, seed: int) -> bool:
    full = feat(events, rows)
    feat_cols = [c for c in full.columns if c != "row_id"]
    assert full.height == rows.height, f"{name}: признак вернул {full.height} строк, ожидалось {rows.height}"
    assert full["row_id"].to_list() == rows.sort("row_id")["row_id"].to_list(), f"{name}: порядок row_id нарушен"

    sample = rows.sample(n=min(n_rows, rows.height), seed=seed)
    n_bad = 0
    for r in sample.iter_rows(named=True):
        trunc = events.filter(pl.col("ts") < r["ts"])
        one = feat(trunc, rows.filter(pl.col("row_id") == r["row_id"]))
        ref = full.filter(pl.col("row_id") == r["row_id"])
        for c in feat_cols:
            a, b = ref[c][0], one[c][0]
            if not _values_equal(a, b):
                n_bad += 1
                print(f"  [LEAK] {name}.{c} row_id={r['row_id']} uid={r['uid']} ts={r['ts']} "
                      f"split={r['split']}: full={a} truncated={b}")
    status = "OK" if n_bad == 0 else f"FAILED ({n_bad} расхождений)"
    print(f"[leakage] {name}: {status} на {sample.height} строках, признаков: {len(feat_cols)}")
    return n_bad == 0


def main() -> None:
    cfg = load_config()
    ev = pl.read_parquet(processed_path(cfg, "events.parquet"))
    rows = pl.read_parquet(processed_path(cfg, "rows.parquet"))
    ok = True
    for name, feat in REGISTRY.items():
        ok &= check_leakage(name, feat, ev, rows, cfg["leakage"]["n_check_rows"], cfg["general"]["seed"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
