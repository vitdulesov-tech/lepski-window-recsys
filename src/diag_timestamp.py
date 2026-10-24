"""Диагностика семантики timestamp: начало или конец прослушивания?

Для test-строк считаем разрыв до предыдущего и до следующего события пользователя (любого типа)
и смотрим, насколько они предсказывают метку. Если ts — конец прослушивания, то
gap_prev / track_len ≈ played_ratio и AUC(gap_prev) будет около 0.95+. Если ts — начало,
то то же верно для gap_next (которым мы не пользуемся, так что утечки нет).

Запуск:  python -m src.diag_timestamp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config, processed_path


def main() -> None:
    cfg = load_config()
    ev = pd.read_parquet(processed_path(cfg, "events.parquet"), columns=["uid", "ts", "event_type"])
    rows = pd.read_parquet(processed_path(cfg, "rows.parquet"))
    rows = rows[rows["split"] == "test"].sample(200_000, random_state=0)

    # предыдущее и следующее событие пользователя (строго раньше / строго позже t)
    ev = ev.sort_values(["uid", "ts"]).reset_index(drop=True)
    key = ev["uid"].to_numpy(np.int64) * (1 << 33) + ev["ts"].to_numpy(np.int64)
    qk = rows["uid"].to_numpy(np.int64) * (1 << 33) + rows["ts"].to_numpy(np.int64)
    lo = np.searchsorted(key, qk, side="left")      # первый индекс с ts >= t
    hi = np.searchsorted(key, qk, side="right")     # первый индекс с ts >  t
    ts_arr = ev["ts"].to_numpy(np.int64)
    uid_arr = ev["uid"].to_numpy(np.int64)
    u = rows["uid"].to_numpy(np.int64)
    t = rows["ts"].to_numpy(np.int64)

    prev_ok = (lo > 0) & (uid_arr[np.maximum(lo - 1, 0)] == u)
    next_ok = (hi < len(ev)) & (uid_arr[np.minimum(hi, len(ev) - 1)] == u)
    gap_prev = np.where(prev_ok, t - ts_arr[np.maximum(lo - 1, 0)], np.nan).astype(float)
    gap_next = np.where(next_ok, ts_arr[np.minimum(hi, len(ev) - 1)] - t, np.nan).astype(float)
    tl = rows["track_length_seconds"].to_numpy(float)
    y = rows["y"].to_numpy()

    def auc(x):
        m = ~np.isnan(x)
        return roc_auc_score(y[m], x[m]), m.mean()

    print(f"строк: {len(rows):,}; доля y=1: {y.mean():.3f}")
    for name, x in (("gap_prev (с)", gap_prev), ("gap_prev / track_len", gap_prev / tl),
                    ("gap_next (с)", gap_next), ("gap_next / track_len", gap_next / tl),
                    ("track_len", tl)):
        a, cov = auc(x)
        print(f"AUC({name:<22}) = {a:.4f}   (доступно для {cov:.1%} строк)")

    for lab in (0, 1):
        m = y == lab
        q = lambda x: np.nanpercentile(x[m], [25, 50, 75])
        print(f"y={lab}: медианы gap_prev/track_len {q(gap_prev / tl)}  |  gap_next/track_len {q(gap_next / tl)}")

    # Сколько строк имеют событие того же пользователя в той же 5-секундной корзине (ts == t)?
    same = hi - lo - 1
    print(f"других событий пользователя с тем же ts, что у строки: среднее {same.mean():.3f}, доля строк с >0: {(same > 0).mean():.3f}")


if __name__ == "__main__":
    main()
