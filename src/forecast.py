"""Step 2: weekly demand forecasts per product, tested honestly, then the real forecast for the next 8 weeks.

Methods compared on the same weeks:
  last_week        units sold last week
  avg_8w           average of the last 8 weeks
  same_week_ly     same week last year
  ly_x_trend       same week last year x (last 8 weeks / same 8 weeks last year)   <- the usual business method
  lightgbm         one machine-learning model for all products (recent sales, last-year pattern, price,
                   weeks to Christmas, trading days), P10 / P50 / P90 via quantile models

Backtest: 7 forecast dates in 2011 (every 4 weeks, 28 Mar to 3 Oct); at each date only earlier weeks are used,
the next 8 weeks are forecast and compared with what really sold.
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
H = 8
W_ML = 0.5  # share of the machine-learning forecast in the combined forecast
warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-empty history for products not launched yet
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def load():
    wp = pd.read_parquet(DATA / "weekly_product.parquet")
    products = pd.read_parquet(DATA / "products.parquet")
    total = pd.read_parquet(DATA / "weekly_total.parquet")
    weeks = total["week"].sort_values().reset_index(drop=True)
    codes = products["stock_code"].tolist()
    Y = wp.pivot(index="stock_code", columns="week", values="units").reindex(index=codes, columns=weeks).to_numpy(float)
    P = wp.pivot(index="stock_code", columns="week", values="avg_price").reindex(index=codes, columns=weeks).to_numpy(float)
    td = total.set_index("week").reindex(weeks)["trading_days"].to_numpy(float)
    return products, weeks, Y, P, td


def future_calendar(weeks: pd.Series, td: np.ndarray, n: int):
    """Next n weeks; trading days are planned in advance: take the same week last year (Christmas closure etc.)."""
    fw = pd.Series(pd.date_range(weeks.iloc[-1] + pd.Timedelta(weeks=1), periods=n, freq="W-MON"))
    ftd = np.array([td[len(weeks) - 52 + i] for i in range(n)], dtype=float)
    return fw, ftd


AGG = {}  # category and store weekly units: smoother seasonal signals than one product's own history


def set_aggregates(Y_all, cat_codes, store_units):
    AGG["cat"] = np.vstack([np.nansum(Y_all[cat_codes == c], axis=0) for c in range(cat_codes.max() + 1)])
    AGG["store"] = store_units


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / den, 0.2, 5.0)


def features(Y, P, td_all, weeks_all, cat_codes, t, hs):
    """Feature rows for every product at forecast date t (index of the last known week) for horizons hs.

    Sales are turned into units per trading day, so a short week (Easter, bank holiday) or the Christmas
    closure does not look like a fall in demand. The model predicts a daily rate; x planned trading days.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        per_day = np.where(td_all > 0, 1.0 / td_all, np.nan)
    Y = Y * per_day
    n = Y.shape[0]
    C, S = AGG["cat"][cat_codes] * per_day, AGG["store"] * per_day
    hist = Y[:, : t + 1]
    last = lambda k: hist[:, max(0, t + 1 - k):]  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.nanmean(last(13), axis=1) + 1.0
        m4, m8, m13, m26 = (np.nanmean(last(k), axis=1) for k in (4, 8, 13, 26))
        s8 = np.nanstd(last(8), axis=1)
        ly8 = np.nanmean(Y[:, max(0, t - 59): max(0, t - 51)], axis=1) if t >= 59 else np.full(n, np.nan)
        growth = np.clip(m8 / ly8, 0.3, 3.0)
        med8 = np.nanmedian(last(8), axis=1)
        lumpy = np.nanmax(last(8), axis=1) / (np.nansum(last(8), axis=1) + 1)
        c_ly8 = C[:, t - 59: t - 51].mean(axis=1) if t >= 59 else np.full(n, np.nan)
        s_ly8 = S[t - 59: t - 51].mean() if t >= 59 else np.nan
        cat_trend = _ratio(C[:, t - 7: t + 1].mean(axis=1), c_ly8)
        p4 = np.nanmean(P[:, max(0, t - 3): t + 1], axis=1)
        pmed = np.nanmedian(P[:, : t + 1], axis=1)
    rows = []
    for h in hs:
        T = t + h
        wk = weeks_all.iloc[T]
        xmas = pd.Timestamp(year=wk.year, month=12, day=25)
        ly = Y[:, T - 52] if T - 52 >= 0 else np.full(n, np.nan)
        ly_pm1 = np.nanmean(Y[:, max(0, T - 53): T - 50], axis=1) if T - 53 >= 0 else np.full(n, np.nan)
        f = {"h": np.full(n, h), "log_scale": np.log(scale), "cat": cat_codes,
             "m4": m4 / scale, "m8": m8 / scale, "m13": m13 / scale, "m26": m26 / scale, "s8": s8 / scale,
             "ly": ly / scale, "ly_pm1": ly_pm1 / scale, "growth": growth, "ly_x_growth": ly * growth / scale,
             "woy": np.full(n, wk.isocalendar().week), "month": np.full(n, wk.month),
             "weeks_to_xmas": np.full(n, ((xmas - wk).days % 365) / 7), "td_target": np.full(n, td_all[T]),
             "td_origin": np.full(n, td_all[t]), "price_rel": p4 / pmed,
             "med8": med8 / scale, "lumpy": lumpy, "cat_trend": cat_trend}
        cat_si = _ratio(C[:, T - 52], c_ly8) if T - 52 >= 0 else np.full(n, np.nan)
        store_si = _ratio(np.full(n, S[T - 52]), s_ly8) if T - 52 >= 0 else np.full(n, np.nan)
        f.update(cat_si=cat_si, store_si=store_si, m8_x_cat_si=m8 / scale * cat_si, m8_x_store_si=m8 / scale * store_si,
                 prod_si=_ratio(ly_pm1, ly8))
        for k in range(8):
            f[f"lag{k}"] = (hist[:, t - k] if t - k >= 0 else np.full(n, np.nan)) / scale
        rows.append(pd.DataFrame(f).assign(p=np.arange(n), t=t, T=T, scale=scale * max(td_all[T], 0)))
    return pd.concat(rows, ignore_index=True)


def baselines(Y, t, T, td=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.nansum(Y[:, t - 7: t + 1], axis=1) / td[t - 7: t + 1].sum() if td is not None else None
        m8 = np.nanmean(Y[:, t - 7: t + 1], axis=1)
        ly = Y[:, T - 52] if T - 52 >= 0 else np.full(Y.shape[0], np.nan)
        ratio = np.nansum(Y[:, t - 7: t + 1], axis=1) / np.nansum(Y[:, t - 59: t - 51], axis=1)
        return {"last_week": Y[:, t], "avg_8w": m8, "same_week_ly": ly,
                "ly_x_trend": ly * np.clip(np.nan_to_num(ratio, nan=1.0), 0.5, 2.0),
                "avg_8w_per_day": daily * td[T] if td is not None else None}


PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "y_scaled") -> dict:
    X_cols = [c for c in train.columns if c not in ("p", "t", "T", "scale", "y", "y_scaled")]
    out = {}
    for name, q in QUANTILES.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
        m.fit(train[X_cols], train[target_col], categorical_feature=["cat"])
        out[name] = np.clip(m.predict(test[X_cols]), 0, None) * test["scale"].to_numpy()
    out["p10"], out["p50"], out["p90"] = np.sort(np.vstack([out["p10"], out["p50"], out["p90"]]), axis=0)
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **PARAMS)  # the expected value
    m.fit(train[X_cols], train[target_col], categorical_feature=["cat"])
    out["mean"] = np.clip(m.predict(test[X_cols]), 0, None) * test["scale"].to_numpy()
    return out


def training_set(Y, P, td, weeks, cat, last_known: int) -> pd.DataFrame:
    """Every (forecast date, horizon) pair whose target week is known by `last_known`."""
    parts = []
    for t in range(12, last_known):
        hs = [h for h in range(1, H + 1) if t + h <= last_known]
        if hs:
            parts.append(features(Y, P, td, weeks, cat, t, hs))
    df = pd.concat(parts, ignore_index=True)
    df["y"] = Y[df["p"], df["T"]]
    df = df[np.isfinite(df["y"]) & np.isfinite(df["lag0"]) & (td[df["T"]] > 0)]
    y_scaled = df["y"] / df["scale"]
    df["y_scaled"] = y_scaled.clip(upper=y_scaled.quantile(0.995))  # a few giant one-off orders would dominate training
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    products, weeks, Y, P, td = load()
    cat = products["category"].astype("category").cat.codes.to_numpy()
    fw, ftd = future_calendar(weeks, td, H)
    weeks_all = pd.concat([weeks, fw], ignore_index=True)
    td_all = np.concatenate([td, ftd])
    Y_all = np.hstack([Y, np.full((Y.shape[0], H), np.nan)])
    P_all = np.hstack([P, np.full((P.shape[0], H), np.nan)])
    store_units = np.concatenate([pd.read_parquet(DATA / "weekly_total.parquet")["units"].to_numpy(float), np.full(H, np.nan)])
    set_aggregates(Y_all, cat, store_units)

    # ---------------- backtest
    start = int(np.where(weeks == pd.Timestamp("2011-03-28"))[0][0])
    origins = [t for t in range(start, len(weeks)) if (t - start) % 4 == 0 and t + H <= len(weeks) - 1]
    bt = []
    for t in origins:
        train = training_set(Y_all, P_all, td_all, weeks_all, cat, t)
        test = features(Y_all, P_all, td_all, weeks_all, cat, t, range(1, H + 1))
        pred = fit_predict(train, test)
        test = test[["p", "t", "T"]].assign(actual=Y[test["p"], test["T"]], **pred)
        for T in range(t + 1, t + H + 1):
            b = baselines(Y, t, T, td_all)
            idx = test["T"] == T
            for k, v in b.items():
                test.loc[idx, k] = v[test.loc[idx, "p"]]
        closed = td_all[test["T"]] == 0  # the store is closed: every method says 0
        test.loc[closed, ["p10", "p50", "p90", "mean", "last_week", "avg_8w", "same_week_ly", "ly_x_trend", "avg_8w_per_day"]] = 0
        bt.append(test)
        print(f"  forecast date {weeks.iloc[t]:%d %b %Y}: trained on {len(train):,} examples", flush=True)
    bt = pd.concat(bt, ignore_index=True).rename(columns={"p50": "lightgbm", "mean": "lightgbm_mean"})
    bt = bt[np.isfinite(bt["actual"])]
    bt["stock_code"] = products["stock_code"].to_numpy()[bt["p"]]
    bt["week"], bt["origin"] = weeks.iloc[bt["T"]].to_numpy(), weeks.iloc[bt["t"]].to_numpy()

    # the forecast we use: half machine learning (expected value), half the 8-week daily average
    bt["combined"] = W_ML * bt["lightgbm_mean"] + (1 - W_ML) * bt["avg_8w_per_day"]
    methods = ["last_week", "avg_8w", "avg_8w_per_day", "same_week_ly", "ly_x_trend", "lightgbm", "lightgbm_mean", "combined"]
    ok = bt.dropna(subset=methods).copy()
    ok["block"] = (ok["T"] - ok["t"] - 1) // 4  # weeks 1-4 and 5-8 after the forecast date
    levels = {"product_week": ok, "product_4_weeks": ok.groupby(["p", "t", "block"])[methods + ["actual"]].sum(),
              "store_week": ok.groupby(["t", "T"])[methods + ["actual"]].sum(),
              "store_4_weeks": ok.groupby(["t", "block"])[methods + ["actual"]].sum()}
    wape = lambda d, m: round(float((d[m] - d["actual"]).abs().sum() / d["actual"].sum()), 4)  # noqa: E731
    simple = ["last_week", "avg_8w", "avg_8w_per_day", "same_week_ly", "ly_x_trend"]

    # ranges: shape from the quantile models, width calibrated on the first 4 forecast dates, checked on the last 3
    ratio = lambda a, b: np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)  # noqa: E731
    ok["lo_raw"] = ok["combined"] * np.nan_to_num(ratio(ok["p10"], ok["lightgbm"]), nan=0.3)
    ok["hi_raw"] = ok["combined"] * np.nan_to_num(ratio(ok["p90"], ok["lightgbm"]), nan=2.5)
    fit_rows, check_rows = ok[ok["t"] < origins[4]], ok[ok["t"] >= origins[4]]
    f_lo = lambda d: float(np.nanquantile(ratio(d["actual"], d["lo_raw"])[d["lo_raw"] > 0], 0.10))  # noqa: E731
    f_hi = lambda d: float(np.nanquantile(ratio(d["actual"], d["hi_raw"])[d["hi_raw"] > 0], 0.90))  # noqa: E731
    lo_k, hi_k = f_lo(fit_rows), f_hi(fit_rows)
    inside = (check_rows["actual"] >= check_rows["lo_raw"] * lo_k) & (check_rows["actual"] <= check_rows["hi_raw"] * hi_k)
    below_hi = check_rows["actual"] <= check_rows["hi_raw"] * hi_k

    metrics = {
        "products": int(Y.shape[0]), "forecast_dates": [f"{weeks.iloc[t]:%Y-%m-%d}" for t in origins], "horizon_weeks": H,
        "product_weeks_tested": int(len(ok)), "combined_weight_ml": W_ML,
        "wape": {lvl: {m: wape(d, m) for m in methods} for lvl, d in levels.items()},
        "bias": {m: round(float((ok[m] - ok["actual"]).sum() / ok["actual"].sum()), 4) for m in methods},
        "best_simple_method": {lvl: min(simple, key=lambda m: wape(d, m)) for lvl, d in levels.items()},
        "range_check_on_last_3_dates": {"inside_low_high": round(float(inside.mean()), 4), "below_high": round(float(below_hi.mean()), 4)},
    }
    metrics["error_reduction_vs_best_simple"] = {
        lvl: round(1 - metrics["wape"][lvl]["combined"] / metrics["wape"][lvl][metrics["best_simple_method"][lvl]], 4) for lvl in levels}
    lo_k, hi_k = f_lo(ok), f_hi(ok)  # final factors use all forecast dates
    metrics["range_factors"] = {"low": round(lo_k, 4), "high": round(hi_k, 4)}
    bt = bt.merge(ok[["p", "t", "T", "lo_raw", "hi_raw"]], on=["p", "t", "T"], how="left")
    bt["low"], bt["high"] = bt["lo_raw"] * lo_k, bt["hi_raw"] * hi_k
    bt.drop(columns=["lo_raw", "hi_raw"]).to_parquet(OUT / "backtest.parquet", index=False)

    # ---------------- the real forecast: next 8 weeks after the last full week
    t = len(weeks) - 1
    train = training_set(Y_all, P_all, td_all, weeks_all, cat, t)
    test = features(Y_all, P_all, td_all, weeks_all, cat, t, range(1, H + 1))
    pred = fit_predict(train, test)
    fc = test[["p", "T"]].assign(**pred)
    for T in range(t + 1, t + H + 1):
        idx = fc["T"] == T
        fc.loc[idx, "avg_8w_per_day"] = baselines(Y_all, t, T, td_all)["avg_8w_per_day"][fc.loc[idx, "p"]]
    fc["forecast"] = W_ML * fc["mean"] + (1 - W_ML) * fc["avg_8w_per_day"]
    fc["low"] = fc["forecast"] * np.nan_to_num(ratio(fc["p10"], fc["p50"]), nan=0.3) * lo_k
    fc["high"] = fc["forecast"] * np.nan_to_num(ratio(fc["p90"], fc["p50"]), nan=2.5) * hi_k
    fc.loc[td_all[fc["T"]] == 0, ["p10", "p50", "p90", "mean", "forecast", "low", "high"]] = 0
    fc["stock_code"] = products["stock_code"].to_numpy()[fc["p"]]
    fc["week"] = weeks_all.iloc[fc["T"]].to_numpy()
    fc["trading_days"] = td_all[fc["T"]]
    fc = fc.drop(columns=["p", "T"])
    fc.to_parquet(OUT / "forecast.parquet", index=False)
    metrics["forecast_weeks"] = [f"{w:%Y-%m-%d}" for w in fw]
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("wape", "bias", "best_simple_method", "error_reduction_vs_best_simple",
                                             "range_check_on_last_3_dates", "range_factors")}, indent=1))
