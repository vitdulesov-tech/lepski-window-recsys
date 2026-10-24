"""Фаза 1, шаг 2: обучение M0 / M0_all / M3+ и оценка (PDF, раздел 6).

Все модели — один LightGBM с одними гиперпараметрами; различаются только счётчиками:
  M0     : Φ0 + ручные окна (win_*)
  M0_all : Φ0 + вся сетка окон (grid_*)
  M3+    : Φ0 + Лепский (lep_{z}_*_{rate,logW,B}); z выбирается по LogLoss на валидации.

Гипотезы (ΔAUC на тесте, парный кластерный бутстреп по пользователям):
  H1 (не хуже):  нижняя граница ДИ для AUC(M3+) - AUC(M0)  >  -margin
  H2 (лучше):    нижняя граница ДИ для AUC(M3+) - AUC(M0)  >  0
Разрывы до M0_all приводятся как описательная статистика.

Запуск:  python -m src.train_eval
Результаты: results/metrics.json, results/summary.md, results/w_hist.csv, results/importance_windows.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.config import PROJECT_ROOT, load_config, processed_path
from src.features.blocks import EVENT_TYPES, LEVELS

RESULTS = PROJECT_ROOT / "results"


# ----------------------------------------------------------------------------- данные

def load(cfg):
    rows = pd.read_parquet(processed_path(cfg, "rows.parquet"))
    feats = pd.read_parquet(processed_path(cfg, "features.parquet"))
    df = rows.merge(feats, on="row_id", how="inner", validate="one_to_one")
    assert len(df) == len(feats)          # признаки построены для подвыборки train-строк
    with open(processed_path(cfg, "z_values.json")) as f:
        z_values = json.load(f)
    return df, z_values


def feature_sets(df: pd.DataFrame, z_values: dict) -> dict[str, list[str]]:
    cols = list(df.columns)
    common = [c for c in cols if c.startswith(("u_", "i_", "a_", "track_len", "hour", "dow"))]
    win = [c for c in cols if c.startswith("win_")]
    grid = [c for c in cols if c.startswith("grid_")]
    win_1d = [c for c in win if c.endswith("_1d")]
    sets = {"M0": common + win, "M0_all": common + grid}
    for zname in z_values:
        lep = [c for c in cols if c.startswith(f"lep_{zname}_") and not c.endswith("_j")]
        sets[f"M3plus@{zname}"] = common + lep
        sets[f"M3plus1d@{zname}"] = common + lep + win_1d     # два масштаба: адаптивный + «сегодня»
    return sets


# ----------------------------------------------------------------------------- обучение

def lgb_params(mc: dict, **override) -> dict:
    p = dict(objective="binary", metric="binary_logloss", verbosity=-1, seed=0,
             num_leaves=mc["num_leaves"], learning_rate=mc["learning_rate"],
             min_data_in_leaf=mc["min_data_in_leaf"], feature_fraction=mc["feature_fraction"],
             bagging_fraction=mc["bagging_fraction"], bagging_freq=mc["bagging_freq"],
             lambda_l2=mc["lambda_l2"], num_threads=mc["num_threads"])
    p.update(override)
    return p


def fit(df, cols, params, mc, name):
    tr, va = df["split"] == "train", df["split"] == "val"
    dtr = lgb.Dataset(df.loc[tr, cols], df.loc[tr, "y"], free_raw_data=False)
    dva = lgb.Dataset(df.loc[va, cols], df.loc[va, "y"], reference=dtr, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(params, dtr, num_boost_round=mc["num_boost_round"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(mc["early_stopping_rounds"], verbose=False)])
    p_va = booster.predict(df.loc[va, cols], num_iteration=booster.best_iteration)
    ll = log_loss(df.loc[va, "y"], p_va)
    print(f"[fit] {name:<14} признаков={len(cols):3d} итераций={booster.best_iteration:5d} "
          f"val LogLoss={ll:.5f} ({time.time() - t0:.0f} s)")
    return booster, ll


# ----------------------------------------------------------------------------- оценка

def weighted_metrics(y, p, w):
    return roc_auc_score(y, p, sample_weight=w), log_loss(y, p, sample_weight=w)


def paired_cluster_bootstrap(y, preds: dict[str, np.ndarray], uid: np.ndarray, n_boot: int, seed=0):
    """Возвращает массивы AUC (n_boot × модели): одни и те же выборки пользователей для всех моделей."""
    rng = np.random.default_rng(seed)
    users, inv = np.unique(uid, return_inverse=True)
    names = list(preds)
    auc = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        draw = rng.integers(0, len(users), size=len(users))
        mult = np.bincount(draw, minlength=len(users))
        w = mult[inv].astype(float)
        keep = w > 0
        for i, nm in enumerate(names):
            auc[b, i] = roc_auc_score(y[keep], preds[nm][keep], sample_weight=w[keep])
    return pd.DataFrame(auc, columns=names)


def window_importance(booster, cols) -> pd.DataFrame:
    imp = pd.Series(booster.feature_importance("gain"), index=cols)
    rec = []
    for c in cols:
        if c.startswith("grid_"):
            lvl, m, w = c.split("_")[1], c.split("_")[2], c.split("_")[3]
            rec.append(dict(level=lvl, event=m, window=w, gain=imp[c]))
    out = pd.DataFrame(rec)
    out["gain_share"] = out["gain"] / out["gain"].sum()
    return out.sort_values(["level", "event", "gain"], ascending=[True, True, False])


def w_hat_histogram(df, zname, grid_days) -> pd.DataFrame:
    te = df["split"] == "test"
    labels = [f"{int(w)}d" for w in grid_days] + ["inf"]
    rec = []
    for lvl in LEVELS:
        for m in EVENT_TYPES:
            j = df.loc[te, f"lep_{zname}_{lvl}_{m}_j"].to_numpy(dtype=int)
            cnt = np.bincount(j, minlength=len(labels))
            for lab, c in zip(labels, cnt):
                rec.append(dict(level=lvl, event=m, window=lab, share=c / len(j)))
    return pd.DataFrame(rec)


# ----------------------------------------------------------------------------- main

def main() -> None:
    cfg = load_config()
    mc, ec = cfg["model"], cfg["eval"]
    RESULTS.mkdir(exist_ok=True)
    df, z_values = load(cfg)
    sets = feature_sets(df, z_values)
    print({k: len(v) for k, v in sets.items()})

    # Гиперпараметры: по умолчанию фиксированы; при tune_on_m0 — небольшой перебор на M0.
    params = lgb_params(mc)
    if mc["tune_on_m0"]:
        best = None
        for nl in (31, 63, 127):
            for lr in (0.03, 0.1):
                _, ll = fit(df, sets["M0"], lgb_params(mc, num_leaves=nl, learning_rate=lr), mc, f"tune nl={nl} lr={lr}")
                if best is None or ll < best[0]:
                    best = (ll, nl, lr)
        params = lgb_params(mc, num_leaves=best[1], learning_rate=best[2])
        print(f"[tune] выбрано num_leaves={best[1]}, learning_rate={best[2]}")

    # Выбор z для M3+ по валидации
    boosters, val_ll = {}, {}
    for name in sets:
        if name.startswith("M3plus@"):
            boosters[name], val_ll[name] = fit(df, sets[name], params, mc, name)
    z_best = min((n for n in val_ll), key=lambda n: val_ll[n])
    zname = z_best.split("@")[1]
    print(f"[z] выбран {zname} (z={z_values[zname]:.2f}) по LogLoss на валидации")

    for name in ("M0", "M0_all", f"M3plus1d@{zname}"):
        boosters[name], val_ll[name] = fit(df, sets[name], params, mc, name)
    final = {"M0": "M0", "M0_all": "M0_all", "M3plus": z_best, "M3plus1d": f"M3plus1d@{zname}"}

    # Тест
    te = df["split"] == "test"
    y = df.loc[te, "y"].to_numpy()
    uid = df.loc[te, "uid"].to_numpy()
    preds, metrics = {}, {}
    for short, name in final.items():
        p = boosters[name].predict(df.loc[te, sets[name]], num_iteration=boosters[name].best_iteration)
        preds[short] = p
        auc, ll = weighted_metrics(y, p, np.ones(len(y)))
        metrics[short] = dict(auc=auc, logloss=ll, n_features=len(sets[name]),
                              best_iteration=int(boosters[name].best_iteration), val_logloss=val_ll[name])
        print(f"[test] {short:<7} AUC={auc:.5f} LogLoss={ll:.5f} признаков={len(sets[name])}")

    # Парный кластерный бутстреп
    t0 = time.time()
    boot = paired_cluster_bootstrap(y, preds, uid, ec["n_boot"])
    print(f"[boot] {ec['n_boot']} повторов ({time.time() - t0:.0f} s)")
    deltas = {
        "M3plus-M0": boot["M3plus"] - boot["M0"],
        "M0_all-M0": boot["M0_all"] - boot["M0"],
        "M0_all-M3plus": boot["M0_all"] - boot["M3plus"],
        "M3plus1d-M0": boot["M3plus1d"] - boot["M0"],
        "M3plus1d-M0_all": boot["M3plus1d"] - boot["M0_all"],
    }
    ci = {k: dict(mean=float(v.mean()), lo=float(v.quantile(0.025)), hi=float(v.quantile(0.975))) for k, v in deltas.items()}
    margin = ec["noninferiority_margin"]
    h1 = ci["M3plus-M0"]["lo"] > -margin
    h2 = ci["M3plus-M0"]["lo"] > 0

    # Описательные артефакты
    grid_days = cfg["features"]["grid_windows_days"]
    wh = w_hat_histogram(df, zname, grid_days)
    wh.to_csv(RESULTS / "w_hist.csv", index=False)
    wi = window_importance(boosters["M0_all"], sets["M0_all"])
    wi.to_csv(RESULTS / "importance_windows.csv", index=False)

    out = dict(metrics=metrics, delta_auc_ci=ci, z_selected=dict(name=zname, value=z_values[zname]),
               val_logloss_by_z={n: val_ll[n] for n in val_ll if n.startswith("M3plus@")},
               H1_noninferior=bool(h1), H2_superior=bool(h2), margin=margin, n_test_rows=int(te.sum()),
               n_test_users=int(len(np.unique(uid))), params=params)
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    labels = [f"{int(w)}d" for w in grid_days] + ["inf"]
    write_summary(out, wh, wi, labels)
    print(f"[done] результаты в {RESULTS}")


def write_summary(out, wh: pd.DataFrame, wi: pd.DataFrame, labels: list[str]) -> None:
    m = out["metrics"]; ci = out["delta_auc_ci"]
    lines = ["# Результаты фазы 1", "",
             f"Тест: {out['n_test_rows']:,} строк, {out['n_test_users']:,} пользователей. "
             f"Выбранный порог: {out['z_selected']['name']} (z = {out['z_selected']['value']:.2f}).", "",
             "| Модель | Признаков | AUC | LogLoss | Итераций |", "|---|---|---|---|---|"]
    for k in ("M0", "M0_all", "M3plus", "M3plus1d"):
        lines.append(f"| {k} | {m[k]['n_features']} | {m[k]['auc']:.5f} | {m[k]['logloss']:.5f} | {m[k]['best_iteration']} |")
    lines += ["", "| Разность AUC | среднее | 95% ДИ |", "|---|---|---|"]
    for k, v in ci.items():
        lines.append(f"| {k} | {v['mean']:+.5f} | [{v['lo']:+.5f}, {v['hi']:+.5f}] |")
    lines += ["",
              f"**H1 (M3+ не хуже M0, порог {out['margin']}):** {'подтверждена' if out['H1_noninferior'] else 'не подтверждена'}.",
              f"**H2 (M3+ лучше M0):** {'подтверждена' if out['H2_superior'] else 'не подтверждена'}.", "",
              "M3plus1d — дополнительная модель: Лепский + счётчик за 1 день (24 счётчика, как у M0); "
              "проверяет гипотезу о двух масштабах сигнала (сессия + долгая память).", "",
              "LogLoss на валидации по порогам z: " + ", ".join(f"{k.split('@')[1]}={v:.5f}" for k, v in out["val_logloss_by_z"].items()), "",
              "## Распределение выбранного окна W_hat на тесте (доля строк)", ""]
    piv = wh.pivot_table(index=["level", "event"], columns="window", values="share")
    lines.append(piv[[c for c in labels if c in piv.columns]].round(3).to_markdown())
    lines += ["", "## Важность (gain) оконных признаков в M0_all, доля по окнам", ""]
    piv2 = wi.pivot_table(index=["level", "event"], columns="window", values="gain_share")
    lines.append(piv2[[c for c in labels if c in piv2.columns]].round(3).to_markdown())
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
