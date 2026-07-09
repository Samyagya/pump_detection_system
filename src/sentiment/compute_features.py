"""
compute_features.py
--------------------
Computes all ERTS input features per symbol-day from your existing
data/processed/*_cleaned.csv (Date, Open, High, Low, Close, Volume,
Delivery_Percentage) and merges optional sentiment from
data/textual/daily_sentiment.csv.

Circuit-limit / circuit-frequency features are intentionally SKIPPED for
now (not coded yet) - can be added later as another column without
touching anything else here.

Output:
  data/erts/features/all_features.csv   (long format: symbol, date, all features)

Design principle: nothing here should crash or zero-out a stock's score just
because sentiment is missing for that stock/day. Sentiment columns default
to 0 / low-quality flags rather than dropping rows.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TEXTUAL_DIR = Path("data/textual")
OUT_DIR = Path("data/erts/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

ROLL_SHORT = 20   # volume / delivery rolling window
ROLL_LONG = 60    # z-score rolling window for return anomaly


def zscore(series, window):
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std()
    z = (series - mean) / (std + EPS)
    return z.clip(-5, 5)  # cap extreme outliers so a single bad denominator
    # doesn't dominate the whole rolling window's signal


def compute_stock_features(df, symbol):
    df = df.sort_values("Date").reset_index(drop=True)
    df["symbol"] = symbol

    # ---------- Price / return features ----------
    df["R_t"] = np.log(df["Close"] / df["Close"].shift(1))

    df["sigma_t"] = df["R_t"].rolling(ROLL_SHORT, min_periods=5).std()

    up_vol = df["R_t"].where(df["R_t"] > 0).rolling(
        ROLL_SHORT, min_periods=5).std()
    down_vol = df["R_t"].where(df["R_t"] < 0).rolling(
        ROLL_SHORT, min_periods=5).std()
    df["V_t"] = (down_vol / (up_vol + EPS)).fillna(1.0)  # asymmetry ratio

    df["IF_z"] = zscore(df["R_t"], ROLL_LONG)
    df["IF_norm"] = np.tanh(df["IF_z"])  # squashed to [-1, 1]

    # EWMA memory of the anomaly signal - persistent unusual moves score higher
    df["M_t"] = df["IF_norm"].ewm(alpha=0.1, min_periods=5).mean()

    # ---------- Volume features ----------
    df["volume_ma"] = df["Volume"].rolling(ROLL_SHORT, min_periods=5).mean()
    # Floor volume_ma at 1st percentile of nonzero volume for this stock,
    # not just EPS - illiquid penny stocks have many near-zero-volume days,
    # and dividing by a near-zero denominator was blowing VR_t up to huge
    # values, saturating downstream z-scores and the A_t sigmoid at ~1.0
    # for almost every stock regardless of real signal.
    vol_floor = max(df["Volume"][df["Volume"] > 0].quantile(
        0.01), 1.0) if (df["Volume"] > 0).any() else 1.0
    df["VR_t"] = df["Volume"] / df["volume_ma"].clip(lower=vol_floor)

    # ---------- Delivery-based accumulation features ----------
    if "Delivery_Percentage" not in df.columns:
        df["Delivery_Percentage"] = np.nan

    delv = df["Delivery_Percentage"].fillna(df["Delivery_Percentage"].median())

    # Smart Accumulation Score: real (delivery-backed) volume surge
    sas_raw = df["VR_t"] * (delv / 100.0)
    df["SAS_t"] = zscore(sas_raw, ROLL_SHORT).fillna(0.0)

    # Delivery Volume Spike: volume surge that IS delivery-backed
    df["DVS_t"] = (df["VR_t"] * (delv / 100.0)).fillna(0.0)

    # Wash Trading Proxy: volume surge with LOW delivery -> circular trading suspicion
    df["WTP_t"] = (df["VR_t"] * (1 - delv / 100.0)).fillna(0.0)
    df["WTP_z"] = zscore(df["WTP_t"], ROLL_SHORT).fillna(0.0)

    keep_cols = [
        "symbol", "Date", "Close", "Volume", "Delivery_Percentage",
        "R_t", "sigma_t", "V_t", "IF_norm", "M_t",
        "VR_t", "SAS_t", "DVS_t", "WTP_t", "WTP_z",
    ]
    return df[keep_cols]


def load_sentiment():
    sent_file = TEXTUAL_DIR / "daily_sentiment.csv"
    if not sent_file.exists():
        print(
            "[Sentiment] daily_sentiment.csv not found - all sentiment will be 0/imputed downstream")
        return pd.DataFrame(columns=["symbol", "date", "SentimentDelta_t", "NewsCount_t", "C_quality"])

    sent = pd.read_csv(sent_file)
    if sent.empty:
        print(
            "[Sentiment] daily_sentiment.csv is empty - all sentiment will be 0/imputed downstream")
        return pd.DataFrame(columns=["symbol", "date", "SentimentDelta_t", "NewsCount_t", "C_quality"])

    sent["date"] = pd.to_datetime(
        sent["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(f"[Sentiment] Loaded {len(sent)} symbol-day sentiment rows covering "
          f"{sent['symbol'].nunique()} symbols")
    return sent[["symbol", "date", "SentimentDelta_t", "NewsCount_t", "C_quality"]]


def main():
    files = sorted(PROCESSED_DIR.glob("*_cleaned.csv"))
    print(f"[Features] Found {len(files)} stock files to process")

    all_frames = []
    for f in files:
        symbol = f.stem.replace("_cleaned", "").replace("_NS", "").upper()
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
            continue
        if len(df) < ROLL_SHORT:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        feats = compute_stock_features(df, symbol)
        all_frames.append(feats)

    features = pd.concat(all_frames, ignore_index=True)
    features["date"] = pd.to_datetime(features["Date"]).dt.strftime("%Y-%m-%d")

    sentiment = load_sentiment()
    merged = features.merge(sentiment, on=["symbol", "date"], how="left")

    # missing sentiment -> explicit low-quality zeros, never NaN (NaN would break ERTS math)
    merged["SentimentDelta_t"] = merged["SentimentDelta_t"].fillna(0.0)
    merged["NewsCount_t"] = merged["NewsCount_t"].fillna(0).astype(int)
    merged["C_quality"] = merged["C_quality"].fillna(0.0)

    out_file = OUT_DIR / "all_features.csv"
    merged.to_csv(out_file, index=False)
    print(
        f"\n[Save] {len(merged)} rows, {merged['symbol'].nunique()} symbols -> {out_file}")
    print(f"[Coverage] Rows with real sentiment (C_quality > 0): "
          f"{(merged['C_quality'] > 0).sum()} / {len(merged)}")


if __name__ == "__main__":
    main()
