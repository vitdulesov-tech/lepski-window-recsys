"""Адаптивный выбор окна по Лепскому для пуассоновских счётчиков (PDF, разделы 3 и 5).

Вход: матрицы N (события в [t - W_j, t)), W (длины окон в днях), valid (окно входит в сетку
для данной строки). Окна вложены: W_1 < ... < W_J среди valid. Порог z.

Критерий однородности отрезков [t - W_j, t - W_k) и [t - W_k, t), k < j:
    T_jk = W_k * KL(r_k || r_j) + (W_j - W_k) * KL(r_old || r_j),  KL(a||b) = a log(a/b) - a + b,
где r_k = N_k / W_k, r_j = N_j / W_j (объединённая), r_old = (N_j - N_k) / (W_j - W_k).

Правило: j_hat = max{ j valid : T_jk <= z для всех valid k < j }.
Признаки: r_hat = r_{j_hat}; log W_hat; B = max_k T_{j', k}, где j' — первое valid окно после j_hat
(B = 0, если j_hat — последнее окно).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def kl_rate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL(Poi(a) || Poi(b)) на единицу времени, поэлементно. a >= 0, b >= 0.
    Соглашения: a = 0 -> b; b = 0 (тогда и a = 0) -> 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = np.broadcast_arrays(a, b)
    out = np.array(b, dtype=float, copy=True)           # a = 0  ->  KL = b
    pos = a > 0
    ap, bp = a[pos], np.maximum(b[pos], EPS)
    out[pos] = ap * np.log(ap / bp) - ap + b[pos]
    return out


def lepski_select(N: np.ndarray, W: np.ndarray, valid: np.ndarray, z: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (j_hat, r_hat, W_hat, B) для каждой строки.

    N, W: float (n, J); valid: bool (n, J). Среди valid окон длины строго возрастают по j.
    """
    N = np.asarray(N, dtype=float)
    W = np.asarray(W, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n, J = N.shape
    assert valid[:, 0].all(), "первое (самое узкое) окно должно быть валидным для всех строк"

    r = np.where(valid, N / np.maximum(W, EPS), 0.0)
    T = np.zeros((n, J, J), dtype=np.float32)     # T[:, j, k], k < j (float32 — экономия памяти)
    accepted = valid.copy()
    for j in range(1, J):
        for k in range(j):
            both = valid[:, j] & valid[:, k]
            Wk, Wj = W[:, k], W[:, j]
            Nk, Nj = N[:, k], N[:, j]
            r_old = (Nj - Nk) / np.maximum(Wj - Wk, EPS)
            Tjk = Wk * kl_rate(r[:, k], r[:, j]) + (Wj - Wk) * kl_rate(r_old, r[:, j])
            Tjk = np.where(both, Tjk, 0.0)
            T[:, j, k] = Tjk
            accepted[:, j] &= (Tjk <= z) | ~valid[:, k]

    # j_hat = максимальный принятый индекс
    idx = np.arange(J)[None, :]
    j_hat = np.max(np.where(accepted, idx, -1), axis=1)
    assert (j_hat >= 0).all()
    rows = np.arange(n)
    r_hat = r[rows, j_hat]
    W_hat = W[rows, j_hat]

    # j' = первое валидное окно после j_hat (если есть)
    after = valid & (idx > j_hat[:, None])
    has_next = after.any(axis=1)
    j_next = np.where(has_next, np.argmax(after, axis=1), -1)
    B = np.zeros(n, dtype=float)
    if has_next.any():
        rr = rows[has_next]
        B[has_next] = T[rr, j_next[has_next], :].max(axis=1)
    return j_hat, r_hat, W_hat, B


def lepski_select_bruteforce(N_row: np.ndarray, W_row: np.ndarray, valid_row: np.ndarray, z: float
                             ) -> tuple[int, float, float, float]:
    """Та же процедура для одной строки, скалярно и буквально по определению (для тестов)."""
    J = len(N_row)

    def kl(a, b):
        if a == 0:
            return b
        return a * np.log(a / b) - a + b

    def T(j, k):
        rk, rj = N_row[k] / W_row[k], N_row[j] / W_row[j]
        r_old = (N_row[j] - N_row[k]) / (W_row[j] - W_row[k])
        return W_row[k] * kl(rk, rj) + (W_row[j] - W_row[k]) * kl(r_old, rj)

    accepted = []
    for j in range(J):
        if not valid_row[j]:
            continue
        ok = all(T(j, k) <= z for k in range(j) if valid_row[k])
        if ok:
            accepted.append(j)
    j_hat = max(accepted)
    nxt = [j for j in range(j_hat + 1, J) if valid_row[j]]
    B = max(T(nxt[0], k) for k in range(nxt[0]) if valid_row[k]) if nxt else 0.0
    return j_hat, N_row[j_hat] / W_row[j_hat], W_row[j_hat], B
