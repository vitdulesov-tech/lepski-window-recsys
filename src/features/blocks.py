"""Блоки признаков фазы 1. Все функции принимают pandas-таблицы events и rows (схема фазы 0)
и возвращают DataFrame признаков в порядке rows. Всё — строго по событиям с ts < t.

Блоки:
  common  — Φ0: пользователь, объект, артист, контекст (общий для всех моделей)
  manual  — M0: оконные счётчики 1/7/30/∞ дней по (уровень, тип события)
  grid    — M0_all: все окна сетки 1,2,4,...,128,∞ дней
  lepski  — M3+: (r_hat, log W_hat, B) по той же сетке, для набора порогов z
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.counters import PITCounter, pair_key, window_counts
from src.features.lepski import lepski_select

DAY = 86400
LEVELS = {"item": "item_id", "album": "album_id", "artist": "artist_id"}
EVENT_TYPES = ("listen", "like")


# ----------------------------------------------------------------------------- helpers

def _g(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(-1).to_numpy(dtype=np.int64)


def _event_mask(ev: pd.DataFrame, m: str) -> np.ndarray:
    return (ev["event_type"].to_numpy() == m)


class PairCounters:
    """Счётчики пар (uid, атрибут объекта) для каждого (уровень, тип события)."""

    def __init__(self, ev: pd.DataFrame, levels: dict[str, str], event_types: tuple[str, ...]):
        self.levels, self.event_types = levels, event_types
        self.g_max = {lvl: int(max(ev[col].max(), 0)) for lvl, col in levels.items()}
        self.counters: dict[tuple[str, str], PITCounter] = {}
        ts = ev["ts"].to_numpy(dtype=np.int64)
        uid = ev["uid"].to_numpy(dtype=np.int64)
        for m in event_types:
            mask = _event_mask(ev, m)
            for lvl, col in levels.items():
                key = pair_key(uid[mask], _g(ev, col)[mask], self.g_max[lvl])
                self.counters[(lvl, m)] = PITCounter(key, ts[mask])

    def row_keys(self, rows: pd.DataFrame, lvl: str) -> tuple[np.ndarray, np.ndarray]:
        g = _g(rows, self.levels[lvl])
        key = pair_key(rows["uid"].to_numpy(dtype=np.int64), g, self.g_max[lvl])
        return key, g >= 0


def first_event_ts(ev: pd.DataFrame) -> pd.Series:
    return ev.groupby("uid")["ts"].min()


def with_decision_time(ev: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Сдвигает момент предсказания строки к НАЧАЛУ прослушивания.

    В Yambda timestamp прослушивания — момент его окончания (проверено: у дослушанных треков
    разрыв до предыдущего события равен длине трека). Считать признаки на момент t значит
    видеть длительность текущего прослушивания, то есть метку. Поэтому
        t* = ts последнего прослушивания пользователя строго раньше t  (= начало текущего),
        t* = t - track_len, если предыдущего прослушивания нет.
    Возвращает копию rows с ts := t*, day := t* // DAY и исходным временем в ts_end.
    """
    is_listen = _event_mask(ev, "listen")
    c = PITCounter(ev["uid"].to_numpy(np.int64)[is_listen], ev["ts"].to_numpy(np.int64)[is_listen])
    t = rows["ts"].to_numpy(np.int64)
    prev = c.last_before(rows["uid"].to_numpy(np.int64), t)
    fallback = np.maximum(t - rows["track_length_seconds"].fillna(0).to_numpy(np.int64), 0)
    t_star = np.where(prev >= 0, prev, fallback)
    assert (t_star <= t).all()
    out = rows.copy()
    out["ts_end"] = t
    out["ts"] = t_star
    out["day"] = t_star // DAY
    return out


# ----------------------------------------------------------------------------- Φ0

def block_common(ev: pd.DataFrame, rows: pd.DataFrame, p0: float | None = None,
                 alpha: float = 10.0) -> pd.DataFrame:
    t = rows["ts"].to_numpy(dtype=np.int64)
    uid = rows["uid"].to_numpy(dtype=np.int64)
    item = rows["item_id"].to_numpy(dtype=np.int64)
    artist = _g(rows, "artist_id")
    out = pd.DataFrame(index=rows.index)

    ts = ev["ts"].to_numpy(dtype=np.int64)
    ev_uid = ev["uid"].to_numpy(dtype=np.int64)
    ev_item = ev["item_id"].to_numpy(dtype=np.int64)
    ev_artist = _g(ev, "artist_id")
    is_listen = _event_mask(ev, "listen")
    is_like = _event_mask(ev, "like")
    is_lp = is_listen & (ev["played_ratio_pct"].fillna(0).to_numpy() >= 50)
    is_rec = ev["is_organic"].to_numpy() == 0

    # Априорная доля дослушиваний — по train-строкам (point-in-time по построению сплита).
    if p0 is None:
        tr = rows["split"].to_numpy() == "train"
        p0 = float(rows.loc[tr, "y"].mean()) if tr.any() else 0.5

    # Пользователь
    c_ul = PITCounter(ev_uid[is_listen], ts[is_listen])
    c_uk = PITCounter(ev_uid[is_like], ts[is_like])
    c_url = PITCounter(ev_uid[is_listen & is_rec], ts[is_listen & is_rec])
    c_urlp = PITCounter(ev_uid[is_lp & is_rec], ts[is_lp & is_rec])
    n_ul = c_ul.count_before(uid, t)
    n_url = c_url.count_before(uid, t)
    out["u_listen_cnt"] = np.log1p(n_ul)
    out["u_like_cnt"] = np.log1p(c_uk.count_before(uid, t))
    out["u_rec_listen_cnt"] = np.log1p(n_url)
    out["u_rec_lp_rate"] = (c_urlp.count_before(uid, t) + alpha * p0) / (n_url + alpha)
    tau = first_event_ts(ev)
    tau_u = tau.reindex(rows["uid"]).to_numpy(dtype=np.int64)
    out["u_days_since_first"] = (t - tau_u) / DAY
    last = c_ul.last_before(uid, t)
    out["u_hours_since_last_listen"] = np.where(last >= 0, (t - last) / 3600.0, np.nan)

    # Объект и артист (популярность по всем пользователям до t)
    c_il = PITCounter(ev_item[is_listen], ts[is_listen])
    c_ilp = PITCounter(ev_item[is_lp], ts[is_lp])
    n_il = c_il.count_before(item, t)
    out["i_pop"] = np.log1p(n_il)
    out["i_lp_rate"] = (c_ilp.count_before(item, t) + alpha * p0) / (n_il + alpha)
    c_al = PITCounter(ev_artist[is_listen & (ev_artist >= 0)], ts[is_listen & (ev_artist >= 0)])
    out["a_pop"] = np.log1p(c_al.count_before(artist, t))

    # Контекст
    out["track_len"] = rows["track_length_seconds"].to_numpy(dtype=float)
    out["hour"] = (t % DAY) // 3600
    out["dow"] = rows["day"].to_numpy(dtype=np.int64) % 7
    return out


# ----------------------------------------------------------------------------- окна

def _pair_window_counts(pc: PairCounters, rows: pd.DataFrame, windows_days: list[float]
                        ) -> dict[tuple[str, str], np.ndarray]:
    t = rows["ts"].to_numpy(dtype=np.int64)
    windows_sec = [w * DAY if np.isfinite(w) else np.inf for w in windows_days]
    out = {}
    for (lvl, m), counter in pc.counters.items():
        key, ok = pc.row_keys(rows, lvl)
        N = window_counts(counter, key, t, windows_sec)
        N[~ok] = 0
        out[(lvl, m)] = N
    return out


def block_windows(pc: PairCounters, rows: pd.DataFrame, windows_days: list[float], prefix: str
                  ) -> pd.DataFrame:
    """Оконные счётчики log1p(N) для набора окон (manual или grid)."""
    counts = _pair_window_counts(pc, rows, windows_days)
    out = pd.DataFrame(index=rows.index)
    for (lvl, m), N in counts.items():
        for j, w in enumerate(windows_days):
            name = f"{prefix}_{lvl}_{m}_{'inf' if np.isinf(w) else f'{int(w)}d'}"
            out[name] = np.log1p(N[:, j])
    return out


def block_lepski(pc: PairCounters, rows: pd.DataFrame, ev: pd.DataFrame,
                 grid_days: list[float], z_values: dict[str, float]) -> pd.DataFrame:
    """Признаки Лепского для каждого z из z_values. grid_days — конечные окна (без inf);
    окно ∞ := max(t - tau_u, 1 день) добавляется последним; конечные окна W_j >= t - tau_u исключаются."""
    assert all(np.isfinite(grid_days)) and list(grid_days) == sorted(grid_days)
    counts = _pair_window_counts(pc, rows, list(grid_days) + [np.inf])
    t = rows["ts"].to_numpy(dtype=np.int64)
    tau_u = first_event_ts(ev).reindex(rows["uid"]).to_numpy(dtype=np.int64)
    span_days = (t - tau_u) / DAY                                  # длина доступной истории
    n, J = len(rows), len(grid_days) + 1
    W = np.empty((n, J)); valid = np.empty((n, J), dtype=bool)
    W[:, :-1] = np.asarray(grid_days, dtype=float)[None, :]
    W[:, -1] = np.maximum(span_days, 1.0)
    valid[:, :-1] = W[:, :-1] < span_days[:, None]
    valid[:, -1] = True
    # Самое узкое окно обязано быть валидным: если истории меньше 1 дня, оставляем только ∞-окно
    # длиной 1 день — для этого переносим его на первую позицию через маску.
    short = ~valid[:, 0]
    if short.any():
        # для таких строк все конечные окна невалидны; сделаем первое окно = ∞-окну
        for (lvl, m) in counts:
            counts[(lvl, m)][short, 0] = counts[(lvl, m)][short, -1]
        W[short, 0] = W[short, -1]
        valid[short, 0] = True
        valid[short, -1] = False

    out = pd.DataFrame(index=rows.index)
    chunk = 500_000                                   # ограничивает память матрицы T (n, J, J)
    for zname, z in z_values.items():
        for (lvl, m), N in counts.items():
            res = [np.empty(n) for _ in range(3)] + [np.empty(n, dtype=np.int64)]
            for s0 in range(0, n, chunk):
                sl = slice(s0, min(s0 + chunk, n))
                j_hat, r_hat, W_hat, B = lepski_select(N[sl].astype(float), W[sl], valid[sl], z)
                res[0][sl], res[1][sl], res[2][sl], res[3][sl] = r_hat, np.log(W_hat), B, j_hat
            base = f"lep_{zname}_{lvl}_{m}"
            out[f"{base}_rate"], out[f"{base}_logW"], out[f"{base}_B"] = res[0], res[1], res[2]
            out[f"{base}_j"] = res[3]          # для гистограмм; в модель не подаётся
    return out


def z_grid(J: int, delta: float, fractions: list[float]) -> dict[str, float]:
    z_th = 2.0 * np.log(4.0 * J / delta)
    return {f"z{int(round(f * 100)):03d}": z_th * f for f in fractions}
