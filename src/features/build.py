"""Фаза 1, шаг 1: построение всех блоков признаков и сохранение в data/processed/features.parquet.

Запуск:  python -m src.features.build
Время на 5000 пользователей (~25 млн событий, ~1 млн строк): минуты.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import load_config, processed_path
from src.features import blocks
from src.features.blocks import EVENT_TYPES, LEVELS, PairCounters


def subsample_train(rows: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Случайная доля train-строк; val и test целиком. Порядок строк сохраняется."""
    if frac >= 1.0:
        return rows.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    is_tr = (rows["split"] == "train").to_numpy()
    keep = ~is_tr | (rng.random(len(rows)) < frac)
    return rows.loc[keep].reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    fc = cfg["features"]
    t0 = time.time()
    ev = pd.read_parquet(processed_path(cfg, "events.parquet"))
    rows_all = pd.read_parquet(processed_path(cfg, "rows.parquet"))
    print(f"[load] events {len(ev):,}, rows {len(rows_all):,} ({time.time() - t0:.0f} s)")

    # Априор для сглаженных долей — по ВСЕМ train-строкам, до подвыборки.
    p0 = float(rows_all.loc[rows_all["split"] == "train", "y"].mean())
    rows = subsample_train(rows_all, fc["train_row_fraction"], cfg["general"]["seed"])
    rows = blocks.with_decision_time(ev, rows)       # признаки считаются на момент НАЧАЛА прослушивания
    shift = (rows["ts_end"] - rows["ts"]).to_numpy()
    print(f"[t*] сдвиг момента предсказания к началу трека: медиана {np.median(shift):.0f} с, "
          f"q90 {np.percentile(shift, 90):.0f} с")
    print(f"[rows] train-строк взято {int((rows['split'] == 'train').sum()):,} "
          f"({fc['train_row_fraction']:.0%}), val {int((rows['split'] == 'val').sum()):,}, "
          f"test {int((rows['split'] == 'test').sum()):,}; p0={p0:.4f}")

    grid = [float(w) for w in fc["grid_windows_days"]]
    manual = [float(w) for w in fc["manual_windows_days"]] + [np.inf]
    z = blocks.z_grid(len(grid) + 1, fc["delta"], fc["z_fractions"])
    print(f"[z] пороги: { {k: round(v, 2) for k, v in z.items()} }")

    t0 = time.time()
    pc = PairCounters(ev, LEVELS, EVENT_TYPES)
    print(f"[counters] построены ({time.time() - t0:.0f} s)")

    parts = []
    for name, fn in (
        ("common", lambda: blocks.block_common(ev, rows, p0=p0)),
        ("manual", lambda: blocks.block_windows(pc, rows, manual, "win")),
        ("grid", lambda: blocks.block_windows(pc, rows, grid + [np.inf], "grid")),
        ("lepski", lambda: blocks.block_lepski(pc, rows, ev, grid, z)),
    ):
        t0 = time.time()
        df = fn()
        parts.append(df)
        print(f"[{name}] {df.shape[1]} признаков ({time.time() - t0:.0f} s)")

    feats = pd.concat(parts, axis=1)
    for c in feats.columns:                       # float32: вдвое меньше памяти и диска
        if feats[c].dtype == np.float64:
            feats[c] = feats[c].astype(np.float32)
    feats.insert(0, "row_id", rows["row_id"].to_numpy())
    out = processed_path(cfg, "features.parquet")
    feats.to_parquet(out, index=False)
    with open(processed_path(cfg, "z_values.json"), "w") as f:
        json.dump(z, f, indent=2)
    print(f"[done] {out}: {feats.shape[0]:,} строк × {feats.shape[1] - 1} признаков")


if __name__ == "__main__":
    main()
