"""
garch_features.py
=================
Stage 2 — GARCH Feature Engineering
Pump & Dump Detection System

For each stock's OHLCV CSV this script produces a new GARCH CSV with:
  - GARCH_Vol_Forecast   : next-day conditional volatility (σ_t, annualised)
  - GARCH_Vol_Surprise   : |actual return| / forecast σ  (raw ratio)
  - GARCH_Surprise_Score : Vol_Surprise normalised 0-100 vs stock's own history
  - GARCH_HighVol_Flag   : 1 when conditional variance is in top-25% of history
  - GARCH_Omega / Alpha / Beta : fitted params (useful for diagnostics)

Usage
-----
  python garch_features.py \
      --ohlcv_dir  data/ohlcv \
      --output_dir data/garch \
      --min_rows   60

Directory layout assumed
------------------------
  data/
    ohlcv/
      3PLAND_NS.csv          ← raw OHLCV per stock
      21STCENMGM_NS.csv
      ...
    garch/                   ← created automatically, one CSV per stock
      3PLAND_NS_garch.csv
      ...
    logs/
      garch_skipped.txt      ← stocks that were skipped and why

Each OHLCV CSV must have at minimum: Date, Close
(Open, High, Low, Volume are not used by GARCH itself but must be present.)
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

warnings.filterwarnings("ignore")          # suppress arch convergence chatter

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(path: Path) -> pd.DataFrame | None:
    """
    Load a single stock OHLCV CSV.
    Returns None (with a logged reason) if the file is unusable.
    """
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        log.warning(f"  Cannot read {path.name}: {e}")
        return None

    df = df.rename(columns=str.strip)          # remove accidental whitespace
    df = df.sort_values("Date").reset_index(drop=True)

    required = {"Date", "Close"}
    if not required.issubset(df.columns):
        log.warning(
            f"  {path.name}: missing columns {required - set(df.columns)}")
        return None

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RETURN SERIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Log returns in % (arch library expects returns scaled ~1-5, not 0.001).
    Multiplying by 100 keeps GARCH parameters well-conditioned.
    """
    return np.log(df["Close"] / df["Close"].shift(1)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIT GARCH(1,1)
# ─────────────────────────────────────────────────────────────────────────────

def fit_garch(returns: pd.Series) -> object | None:
    """
    Fit a GARCH(1,1) with normal innovations on the full return series.
    Returns the fitted result object, or None on failure.

    Why GARCH(1,1)?
    ---------------
    - It captures volatility clustering (σ² today depends on yesterday's shock
      and yesterday's variance) with only 3 free parameters.
    - Higher-order GARCH rarely helps for daily equity data and is much
      harder to estimate robustly on short histories.
    - GJR-GARCH (asymmetric) is a valid upgrade for a later iteration but
      adds complexity without changing the surprise-score logic here.
    """
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
            disp="off",          # suppress iteration output
            show_warning=False,
            options={"maxiter": 500},
        )
        return result
    except Exception as e:
        log.warning(f"    GARCH fit failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROLLING ONE-STEP-AHEAD FORECASTS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_garch_forecasts(
    returns: pd.Series,
    min_train: int = 60,
) -> pd.DataFrame:
    """
    For each date t (starting at index min_train), fit GARCH(1,1) on
    returns[0 : t] and forecast σ_{t+1}.  This avoids look-ahead bias —
    the model never sees future data.

    Returns a DataFrame indexed like `returns` with columns:
        garch_sigma   : one-step-ahead conditional std (in % units)
        garch_omega   : fitted omega parameter
        garch_alpha   : fitted alpha parameter
        garch_beta    : fitted beta parameter

    Notes
    -----
    - Refitting every single day is rigorous but slow on large universes.
      A practical compromise (used here) is to refit every 20 trading days
      and use the cached model for intermediate days — controlled by
      REFIT_EVERY.  Set to 1 to refit daily (slow but exact).
    - Forecasts before min_train are left as NaN.
    """
    REFIT_EVERY = 20       # refit model every N days; change to 1 for daily

    n = len(returns)
    sigma_out = np.full(n, np.nan)
    omega_out = np.full(n, np.nan)
    alpha_out = np.full(n, np.nan)
    beta_out = np.full(n, np.nan)

    cached_result = None
    last_fit_idx = -1

    for t in range(min_train, n):
        needs_refit = (
            cached_result is None
            or (t - last_fit_idx) >= REFIT_EVERY
        )

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
            # one-step ahead forecast from the last fitted model
            forecast = cached_result.forecast(horizon=1, reindex=False)
            # variance forecast → std (still in % units)
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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DERIVE SURPRISE SCORE & HIGH-VOL FLAG
# ─────────────────────────────────────────────────────────────────────────────

def compute_garch_features(
    df: pd.DataFrame,
    returns: pd.Series,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Takes the raw GARCH forecast and derives the features that enter the
    Isolation Forest feature matrix and the ERTS fusion layer.

    GARCH_Vol_Surprise
    ------------------
    Raw ratio: |actual log-return (%)| / forecast σ (%)
    > 1 → return exceeded expected volatility
    > 2 → return was 2× expected (notable)
    > 3 → return was 3× expected (severe surprise)
    This is NOT yet normalised — it's the raw anomaly signal.

    GARCH_Surprise_Score (0–100)
    ----------------------------
    Normalised using the stock's OWN historical distribution of surprise
    values (percentile rank × 100). This makes the score comparable across
    stocks of different volatility regimes — a penny stock and a large-cap
    get scored relative to their own history, not a global distribution.

    GARCH_HighVol_Flag
    ------------------
    Binary: 1 when today's conditional variance (σ²) is in the top 25%
    of the stock's own variance history. Captures "the stock has entered
    a high-volatility regime" independent of whether today's specific
    return was surprising.
    """
    out = df[["Date"]].copy()
    out["Log_Return_pct"] = returns.values      # in % (same units as sigma)

    out["GARCH_Vol_Forecast"] = forecasts["garch_sigma"].values
    out["GARCH_Omega"] = forecasts["garch_omega"].values
    out["GARCH_Alpha"] = forecasts["garch_alpha"].values
    out["GARCH_Beta"] = forecasts["garch_beta"].values

    # Vol surprise (raw ratio)
    valid = out["GARCH_Vol_Forecast"] > 0
    out["GARCH_Vol_Surprise"] = np.nan
    out.loc[valid, "GARCH_Vol_Surprise"] = (
        out.loc[valid, "Log_Return_pct"].abs()
        / out.loc[valid, "GARCH_Vol_Forecast"]
    )

    # Normalised surprise score 0–100 (percentile rank within stock history)
    surprise = out["GARCH_Vol_Surprise"].dropna()
    if len(surprise) > 0:
        ranks = surprise.rank(pct=True) * 100
        out["GARCH_Surprise_Score"] = np.nan
        out.loc[ranks.index, "GARCH_Surprise_Score"] = ranks.values
    else:
        out["GARCH_Surprise_Score"] = np.nan

    # High-vol regime flag (top 25% of own variance history)
    variance = out["GARCH_Vol_Forecast"] ** 2
    var_75th = variance.quantile(0.75)
    out["GARCH_HighVol_Flag"] = (variance >= var_75th).astype(int)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PER-STOCK PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_stock(
    ohlcv_path: Path,
    output_dir: Path,
    min_rows: int,
    skip_log: list,
) -> bool:
    """
    Full pipeline for one stock.  Returns True on success.
    """
    ticker = ohlcv_path.stem           # e.g. "3PLAND_NS"
    log.info(f"Processing {ticker} ...")

    # --- load ---
    df = load_ohlcv(ohlcv_path)
    if df is None:
        skip_log.append((ticker, "load_failed"))
        return False

    if len(df) < min_rows:
        msg = f"only {len(df)} rows (need {min_rows})"
        log.warning(f"  {ticker}: skipped — {msg}")
        skip_log.append((ticker, msg))
        return False

    # --- returns ---
    returns = compute_log_returns(df)
    returns.index = df.index           # align index

    # --- rolling GARCH forecasts ---
    forecasts = rolling_garch_forecasts(returns, min_train=min_rows)

    valid_forecasts = forecasts["garch_sigma"].notna().sum()
    if valid_forecasts == 0:
        msg = "GARCH produced zero valid forecasts"
        log.warning(f"  {ticker}: skipped — {msg}")
        skip_log.append((ticker, msg))
        return False

    # --- derived features ---
    result = compute_garch_features(df, returns, forecasts)

    # --- save ---
    out_path = output_dir / f"{ticker}_garch.csv"
    result.to_csv(out_path, index=False, float_format="%.6f")
    log.info(f"  ✓ saved → {out_path.name}  ({valid_forecasts} forecast rows)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GARCH(1,1) feature engineering for pump-detection pipeline"
    )
    parser.add_argument(
        "--ohlcv_dir",
        type=str,
        default="data/ohlcv",
        help="Folder containing per-stock OHLCV CSVs  (default: data/ohlcv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/garch",
        help="Folder to write per-stock GARCH CSVs   (default: data/garch)",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=60,
        help="Minimum trading days required to fit GARCH (default: 60)",
    )
    args = parser.parse_args()

    ohlcv_dir = Path(args.ohlcv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # also make a logs dir
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(ohlcv_dir.glob("*.csv"))
    if not csv_files:
        log.error(f"No CSVs found in {ohlcv_dir}. Check --ohlcv_dir.")
        sys.exit(1)

    log.info(f"Found {len(csv_files)} stock files in {ohlcv_dir}")
    log.info(f"Output → {output_dir}   Min rows = {args.min_rows}")
    log.info("─" * 60)

    skip_log = []
    succeeded = 0
    failed = 0

    for path in csv_files:
        ok = process_stock(path, output_dir, args.min_rows, skip_log)
        if ok:
            succeeded += 1
        else:
            failed += 1

    # ── summary ──────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(f"Done.  ✓ {succeeded} succeeded   ✗ {failed} skipped/failed")

    if skip_log:
        skip_path = logs_dir / "garch_skipped.txt"
        with open(skip_path, "w") as f:
            for ticker, reason in skip_log:
                f.write(f"{ticker}\t{reason}\n")
        log.info(f"Skipped list → {skip_path}")


if __name__ == "__main__":
    main()
