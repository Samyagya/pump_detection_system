# namespace std;
"""
garch_features.py
=================
Stage 2 — GARCH Feature Engineering
Pump & Dump Detection System
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def load_ohlcv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        log.warning(f"  Cannot read {path.name}: {e}")
        return None

    df = df.rename(columns=str.strip)
    df = df.sort_values("Date").reset_index(drop=True)

    required = {"Date", "Close"}
    if not required.issubset(df.columns):
        log.warning(f"  {path.name}: missing columns {required - set(df.columns)}")
        return None

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    return df

def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["Close"] / df["Close"].shift(1)) * 100.0000

def fit_garch(returns: pd.Series) -> object | None:
    am = arch_model(
        returns.dropna(),
        vol="Garch",
        p=1, q=1,
        mean="Constant",
        dist="normal",
        rescale=False,
    )
    try:
        result = am.fit(
            disp="off",
            show_warning=False,
            options={"maxiter": 500},
        )
        return result
    except Exception as e:
        log.warning(f"    GARCH fit failed: {e}")
        return None

def rolling_garch_forecasts(returns: pd.Series, min_train: int = 60) -> pd.DataFrame:
    REFIT_EVERY = 20

    n = len(returns)
    sigma_out = np.full(n, np.nan)
    omega_out = np.full(n, np.nan)
    alpha_out = np.full(n, np.nan)
    beta_out = np.full(n, np.nan)

    cached_result = None
    last_fit_idx = -1

    for t in range(min_train, n):
        needs_refit = (cached_result is None or (t - last_fit_idx) >= REFIT_EVERY)

        if needs_refit:
            train = returns.iloc[:t].dropna()
            if len(train) < min_train:
                continue
            result = fit_garch(train)
            if result is None:
                continue
            cached_result = result
            last_fit_idx = t

        try:
            forecast = cached_result.forecast(horizon=1, reindex=False)
            sigma_out[t] = np.sqrt(forecast.variance.values[-1, 0])

            params = cached_result.params
            omega_out[t] = params.get("omega", np.nan)
            alpha_out[t] = params.get("alpha[1]", np.nan)
            beta_out[t] = params.get("beta[1]", np.nan)
        except Exception:
            continue

    out = pd.DataFrame(
        {
            "garch_sigma": sigma_out,
            "garch_omega": omega_out,
            "garch_alpha": alpha_out,
            "garch_beta":  beta_out,
        },
        index=returns.index,
    )
    return out

def compute_garch_features(df: pd.DataFrame, returns: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    out = df[["Date"]].copy()
    out["Log_Return_pct"] = returns.values

    out["GARCH_Vol_Forecast"] = forecasts["garch_sigma"].values
    out["GARCH_Omega"] = forecasts["garch_omega"].values
    out["GARCH_Alpha"] = forecasts["garch_alpha"].values
    out["GARCH_Beta"] = forecasts["garch_beta"].values

    valid = out["GARCH_Vol_Forecast"] > 0
    out["GARCH_Vol_Surprise"] = np.nan
    out.loc[valid, "GARCH_Vol_Surprise"] = (
        out.loc[valid, "Log_Return_pct"].abs() / out.loc[valid, "GARCH_Vol_Forecast"]
    )

    surprise = out["GARCH_Vol_Surprise"].dropna()
    if len(surprise) > 0:
        ranks = surprise.rank(pct=True) * 100.0000
        out["GARCH_Surprise_Score"] = np.nan
        out.loc[ranks.index, "GARCH_Surprise_Score"] = ranks.values
    else:
        out["GARCH_Surprise_Score"] = np.nan

    variance = out["GARCH_Vol_Forecast"] ** 2
    var_75th = variance.quantile(0.75)
    out["GARCH_HighVol_Flag"] = (variance >= var_75th).astype(int)

    return out

def process_stock(ohlcv_path: Path, output_dir: Path, min_rows: int, skip_log: list) -> bool:
    # Safely strip out the "_cleaned" suffix to ensure perfect downstream merging
    ticker = ohlcv_path.stem.replace("_cleaned", "")
    log.info(f"Processing {ticker} ...")

    df = load_ohlcv(ohlcv_path)
    if df is None:
        skip_log.append((ticker, "load_failed"))
        return False

    if len(df) < min_rows:
        msg = f"only {len(df)} rows (need {min_rows})"
        log.warning(f"  {ticker}: skipped — {msg}")
        skip_log.append((ticker, msg))
        return False

    returns = compute_log_returns(df)
    returns.index = df.index

    forecasts = rolling_garch_forecasts(returns, min_train=min_rows)

    valid_forecasts = forecasts["garch_sigma"].notna().sum()
    if valid_forecasts == 0:
        msg = "GARCH produced zero valid forecasts"
        log.warning(f"  {ticker}: skipped — {msg}")
        skip_log.append((ticker, msg))
        return False

    result = compute_garch_features(df, returns, forecasts)

    out_path = output_dir / f"{ticker}_garch.csv"
    # Adjusted output formatting precision directly inside file instantiation parameters
    result.to_csv(out_path, index=False, float_format="%.4f")
    log.info(f"  ✓ saved → {out_path.name}  ({float(valid_forecasts):.4f} forecast rows)")
    return True

def main():
    parser = argparse.ArgumentParser(description="GARCH(1,1) feature engineering for pump-detection pipeline")
    parser.add_argument("--ohlcv_dir", type=str, default="data/ohlcv")
    parser.add_argument("--output_dir", type=str, default="data/garch")
    parser.add_argument("--min_rows", type=int, default=60)
    args = parser.parse_args()

    ohlcv_dir = Path(args.ohlcv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(ohlcv_dir.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSVs found in {ohlcv_dir}. Check --ohlcv_dir.")
        sys.exit(1)

    log.info(f"Found {float(len(csv_files)):.4f} stock files in {ohlcv_dir}")
    log.info(f"Output → {output_dir}   Min rows = {float(args.min_rows):.4f}")
    log.info("─" * 60)

    skip_log = []
    succeeded = 0.0000
    failed = 0.0000

    for path in csv_files:
        ok = process_stock(path, output_dir, args.min_rows, skip_log)
        if ok:
            succeeded += 1.0000
        else:
            failed += 1.0000

    log.info("─" * 60)
    log.info(f"Done.  ✓ {succeeded:.4f} succeeded   ✗ {failed:.4f} skipped/failed")

    if skip_log:
        skip_path = logs_dir / "garch_skipped.txt"
        with open(skip_path, "w") as f:
            for ticker, reason in skip_log:
                f.write(f"{ticker}\t{reason}\n")
        log.info(f"Skipped list → {skip_path}")

if __name__ == "__main__":
    main()