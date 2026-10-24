"""Тест фазы 1: (а) счётчики совпадают с грубой силой; (б) признаки строки не меняются при
удалении из лога всех событий с ts >= t (нет утечек). Проверяются все четыре блока.

Запуск:  python -m src.leakage_phase1              (на data/processed/{events,rows}.parquet)
         python -m src.leakage_phase1 --synthetic  (на встроенной синтетике, без файлов)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.features import blocks
from src.features.blocks import EVENT_TYPES, LEVELS, PairCounters

GRID = [1, 2, 4, 8, 16, 32, 64, 128]


def synthetic_events_rows(n_users=40, n_items=60, n_days=200, seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n_users):
        n = rng.integers(50, 400)
        ts = np.sort(rng.integers(0, n_days * 86400, size=n))
        items = rng.integers(0, n_items, size=n)
        et = rng.choice(["listen", "like"], size=n, p=[0.9, 0.1])
        recs.append(pd.DataFrame({
            "uid": u, "item_id": items, "artist_id": items % 7, "album_id": items % 13,
            "ts": ts, "day": ts // 86400, "event_type": et,
            "is_organic": rng.integers(0, 2, size=n),
            "played_ratio_pct": np.where(et == "listen", rng.integers(0, 120, size=n), np.nan),
            "track_length_seconds": rng.integers(60, 400, size=n),
        }))
    ev = pd.concat(recs, ignore_index=True).sort_values(["uid", "ts"]).reset_index(drop=True)
    # несколько объектов без артиста (null) — как в реальных данных
    ev.loc[ev["item_id"] == 3, "artist_id"] = np.nan
    rows = ev[(ev["event_type"] == "listen") & (ev["is_organic"] == 0)].copy()
    rows["y"] = (rows["played_ratio_pct"] >= 50).astype(int)
    rows["split"] = np.where(rows["day"] < n_days * 0.7, "train", np.where(rows["day"] < n_days * 0.85, "val", "test"))
    rows = rows.reset_index(drop=True)
    rows.insert(0, "row_id", np.arange(len(rows)))
    return ev, rows


def all_blocks(ev: pd.DataFrame, rows: pd.DataFrame, p0: float | None = None) -> pd.DataFrame:
    pc = PairCounters(ev, LEVELS, EVENT_TYPES)
    z = blocks.z_grid(len(GRID) + 1, 0.05, [1.0, 0.5])
    return pd.concat([
        blocks.block_common(ev, rows, p0=p0),
        blocks.block_windows(pc, rows, [1, 7, 30, np.inf], "win"),
        blocks.block_windows(pc, rows, GRID + [np.inf], "grid"),
        blocks.block_lepski(pc, rows, ev, GRID, z),
    ], axis=1)


def brute_force_window(ev: pd.DataFrame, r: pd.Series, lvl: str, m: str, W_days: float) -> int:
    col = LEVELS[lvl]
    g = r[col]
    if pd.isna(g):
        return 0
    e = ev[(ev["uid"] == r["uid"]) & (ev["event_type"] == m) & (ev[col] == g) & (ev["ts"] < r["ts"])]
    if np.isfinite(W_days):
        e = e[e["ts"] >= r["ts"] - W_days * 86400]
    return len(e)


def main() -> None:
    if "--synthetic" in sys.argv:
        ev, rows = synthetic_events_rows()
    else:
        from src.config import load_config, processed_path
        cfg = load_config()
        ev = pd.read_parquet(processed_path(cfg, "events.parquet"))
        rows = pd.read_parquet(processed_path(cfg, "rows.parquet"))
    rng = np.random.default_rng(0)
    sample = rows.iloc[np.sort(rng.choice(len(rows), size=min(60, len(rows)), replace=False))]
    sample = blocks.with_decision_time(ev, sample)   # тот же сдвиг момента предсказания, что в build
    p0 = float(rows.loc[rows["split"] == "train", "y"].mean())

    full = all_blocks(ev, sample, p0=p0)       # признаки независимы между строками — считаем только выборку

    # (а) счётчики против грубой силы
    n_bad = 0
    for _, r in sample.iterrows():
        for lvl in LEVELS:
            for m in EVENT_TYPES:
                for w, name in ((1, "1d"), (7, "7d"), (30, "30d"), (np.inf, "inf")):
                    bf = brute_force_window(ev, r, lvl, m, w)
                    got = np.expm1(full.loc[r.name, f"win_{lvl}_{m}_{name}"])
                    if abs(got - bf) > 1e-6:
                        n_bad += 1
                        print(f"  [MISMATCH] row {r['row_id']} win_{lvl}_{m}_{name}: fast={got:.1f} brute={bf}")
    print(f"[counters] расхождений с грубой силой: {n_bad}")

    # (б) утечки: удаляем события с ts >= t и пересчитываем все блоки для одной строки
    feat_cols = [c for c in full.columns if not c.endswith("_j")]
    z = blocks.z_grid(len(GRID) + 1, 0.05, [1.0, 0.5])
    n_leak = 0
    for _, r in sample.iterrows():
        trunc = ev[ev["ts"] < r["ts"]]
        if (trunc["uid"] == r["uid"]).sum() == 0:
            continue  # у пользователя нет прошлого: tau_u не определён, признаки пары нулевые по построению
        one_rows = sample.loc[[r.name]]       # строка с уже сдвинутым t*
        pc = PairCounters(trunc, LEVELS, EVENT_TYPES)
        one = pd.concat([
            blocks.block_common(trunc, one_rows, p0=p0),
            blocks.block_windows(pc, one_rows, [1, 7, 30, np.inf], "win"),
            blocks.block_windows(pc, one_rows, GRID + [np.inf], "grid"),
            blocks.block_lepski(pc, one_rows, trunc, GRID, z),
        ], axis=1)
        for c in feat_cols:
            a, b = full.loc[r.name, c], one.loc[r.name, c]
            same = (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
            if not same:
                n_leak += 1
                print(f"  [LEAK] row {r['row_id']} {c}: full={a} truncated={b}")
    print(f"[leakage] расхождений после удаления будущего: {n_leak}")
    sys.exit(0 if n_bad == 0 and n_leak == 0 else 1)


if __name__ == "__main__":
    main()
