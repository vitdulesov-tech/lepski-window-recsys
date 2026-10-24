"""Point-in-time счётчики на numpy.

Единственный примитив: PITCounter.count_before(key, s) = число событий с данным ключом
и ts < s (СТРОГО). Из него получаются все оконные счётчики:
    N_W(t) = count_before(key, t) - count_before(key, t - W).

Реализация: события кодируются одним int64 = dense_key * T_MAX + ts и сортируются;
запрос — два searchsorted. Память O(|events|), запрос O(log |events|) на строку.
"""
from __future__ import annotations

import numpy as np

T_MAX = 1 << 33  # > любого ts в секундах (2^33 с ≈ 272 года); dense_key < 2^30 ⇒ нет переполнения


class PITCounter:
    def __init__(self, key: np.ndarray, ts: np.ndarray):
        key = np.asarray(key, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.int64)
        assert key.shape == ts.shape
        assert ts.min() >= 0 and ts.max() < T_MAX, "ts должны быть неотрицательными секундами"
        if len(key) == 0:
            self.uniq = np.empty(0, dtype=np.int64)
            self.sorted = np.empty(0, dtype=np.int64)
            return
        self.uniq, inv = np.unique(key, return_inverse=True)
        assert len(self.uniq) < (1 << 30)
        self.sorted = np.sort(inv.astype(np.int64) * T_MAX + ts)

    def _dense(self, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(self.uniq) == 0:
            return np.zeros(len(key), dtype=np.int64), np.zeros(len(key), dtype=bool)
        pos = np.searchsorted(self.uniq, key)
        pos_c = np.minimum(pos, len(self.uniq) - 1)
        found = self.uniq[pos_c] == key
        return pos_c, found

    def count_before(self, key: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Число событий ключа key с ts < s. Для неизвестных ключей — 0."""
        key = np.asarray(key, dtype=np.int64)
        s = np.clip(np.asarray(s, dtype=np.int64), 0, T_MAX - 1)
        d, found = self._dense(key)
        if len(self.sorted) == 0:
            return np.zeros(len(key), dtype=np.int64)
        lo = np.searchsorted(self.sorted, d * T_MAX, side="left")
        hi = np.searchsorted(self.sorted, d * T_MAX + s, side="left")
        out = hi - lo
        out[~found] = 0
        return out

    def last_before(self, key: np.ndarray, s: np.ndarray) -> np.ndarray:
        """ts последнего события ключа до s; -1, если его нет."""
        key = np.asarray(key, dtype=np.int64)
        s = np.clip(np.asarray(s, dtype=np.int64), 0, T_MAX - 1)
        d, found = self._dense(key)
        if len(self.sorted) == 0:
            return np.full(len(key), -1, dtype=np.int64)
        hi = np.searchsorted(self.sorted, d * T_MAX + s, side="left")
        lo = np.searchsorted(self.sorted, d * T_MAX, side="left")
        has = found & (hi > lo)
        out = np.full(len(key), -1, dtype=np.int64)
        idx = np.maximum(hi - 1, 0)
        out[has] = self.sorted[idx[has]] - d[has] * T_MAX
        return out


def window_counts(counter: PITCounter, key: np.ndarray, t: np.ndarray,
                  windows_sec: list[float]) -> np.ndarray:
    """Матрица (n_rows, J): число событий ключа в [t - W_j, t). W_j = inf ⇒ вся история."""
    base = counter.count_before(key, t)
    out = np.empty((len(t), len(windows_sec)), dtype=np.int64)
    for j, W in enumerate(windows_sec):
        if np.isinf(W):
            out[:, j] = base
        else:
            out[:, j] = base - counter.count_before(key, t - int(W))
    return out


def pair_key(uid: np.ndarray, g: np.ndarray, g_max: int) -> np.ndarray:
    """Ключ пары (пользователь, значение атрибута) одним int64. g = -1 (нет атрибута) допустимо."""
    return np.asarray(uid, dtype=np.int64) * (int(g_max) + 2) + (np.asarray(g, dtype=np.int64) + 1)
