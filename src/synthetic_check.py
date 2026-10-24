"""Синтетическая проверка критерия Лепского (PDF, раздел 6.4, п. 2).

1. Однородные пуассоновские потоки с интенсивностями mu (событий/день) на отрезке длиной
   span дней. При z = z_th = 2 log(4J/delta) доля строк с j_hat < J должна быть <= delta.
2. Поток со скачком интенсивности mu_old -> mu_new в момент t - tau: j_hat должен
   концентрироваться на наибольшем W_j <= tau.

Запуск:  python -m src.synthetic_check
"""
from __future__ import annotations

import numpy as np

from src.features.lepski import lepski_select, lepski_select_bruteforce

GRID = [1, 2, 4, 8, 16, 32, 64, 128]


def counts_from_times(times_before: np.ndarray, W: np.ndarray) -> np.ndarray:
    """times_before: время до t (дни) для событий потока; W: окна (дни). N_j = #{tau <= W_j}."""
    return (times_before[:, None] <= W[None, :]).sum(axis=0)


def simulate(n_streams: int, mu_fn, span: float, rng: np.random.Generator, J: int):
    W = np.array(GRID + [span], dtype=float)
    valid = np.ones(J, dtype=bool)
    valid[:-1] = W[:-1] < span
    N = np.zeros((n_streams, J))
    for i in range(n_streams):
        # неоднородный Пуассон: интенсивность mu_fn(tau), tau = время до t
        lam_max = mu_fn(np.linspace(0, span, 1000)).max()
        n = rng.poisson(lam_max * span)
        tau = rng.uniform(0, span, size=n)
        keep = rng.uniform(0, lam_max, size=n) < mu_fn(tau)
        N[i] = counts_from_times(tau[keep], W)
    return N, np.tile(W, (n_streams, 1)), np.tile(valid, (n_streams, 1))


def main() -> None:
    rng = np.random.default_rng(0)
    J = len(GRID) + 1
    delta = 0.05
    z_th = 2 * np.log(4 * J / delta)
    print(f"J={J}, delta={delta}, z_th={z_th:.2f}")

    # --- совпадение векторной и скалярной реализаций
    N, W, V = simulate(200, lambda tau: np.full_like(tau, 2.0), span=300, rng=rng, J=J)
    jh, rh, Wh, B = lepski_select(N, W, V, z_th / 2)
    for i in range(200):
        j2, r2, W2, B2 = lepski_select_bruteforce(N[i], W[i], V[i], z_th / 2)
        assert jh[i] == j2 and abs(rh[i] - r2) < 1e-9 and abs(Wh[i] - W2) < 1e-9 and abs(B[i] - B2) < 1e-6, i
    print("[ok] векторная реализация совпадает со скалярной на 200 потоках")

    # --- 1. распространение: однородные потоки
    print("\nОднородные потоки (доля j_hat < J должна быть <= delta при z_th):")
    for mu in (0.05, 0.3, 2.0, 10.0):
        N, W, V = simulate(2000, lambda tau, mu=mu: np.full_like(tau, mu), span=300, rng=rng, J=J)
        for z in (z_th, z_th / 2, z_th / 4):
            jh, *_ = lepski_select(N, W, V, z)
            print(f"  mu={mu:5.2f}/день  z={z:5.2f}  доля обрезанных = {(jh < J - 1).mean():.3f}")

    # --- 2. локализация скачка
    print("\nСкачок интенсивности в момент t - tau (ожидаем W_hat = наибольшее W_j <= tau):")
    for tau_cp, mu_new, mu_old in ((5, 8.0, 0.5), (20, 6.0, 0.3), (50, 0.2, 5.0)):
        f = lambda tau, a=mu_new, b=mu_old, c=tau_cp: np.where(tau <= c, a, b)
        N, W, V = simulate(1000, f, span=300, rng=rng, J=J)
        jh, rh, Wh, B = lepski_select(N, W, V, z_th / 2)
        oracle = max(w for w in GRID if w <= tau_cp)
        vals, cnts = np.unique(Wh, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:3]
        print(f"  tau={tau_cp:3d} {mu_old}->{mu_new}: оракул W*={oracle:3d}; "
              f"мода W_hat={int(top[0][1])} ({top[0][0] / 1000:.0%}); "
              f"|r_hat - mu_new| медиана={np.median(np.abs(rh - mu_new)):.3f}; "
              f"B медиана={np.median(B):.1f}")


if __name__ == "__main__":
    main()
