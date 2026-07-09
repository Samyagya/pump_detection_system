"""
build_daily_sentiment.py
-------------------------
Replaces merge_news.py + sentiment_analysis.py + earlier unify_sentiment.py.

Problem this solves: you have THREE scrapers writing files with THREE different
naming conventions and THREE different column schemas, and nothing merges +
scores them all consistently for every symbol.

Sources handled (all optional per symbol - missing ones are just skipped):
  1. data/news/{symbol}_news.csv                       -> published, source, title, url
  2. data/textual/{SYMBOL}_NS_announcements.csv        -> Date, Headline        (Screener)
  3. data/textual/{SYMBOL}_announcements/announcements.csv -> date, subject, text  (NSE)

Symbol-name mismatch problem (e.g. "eurotex" vs "EUROTEXIND"):
  Solved with prefix matching - the canonical symbol list comes from
  data/processed/*_cleaned.csv, and any news/screener/NSE file whose name
  STARTS WITH that symbol (case-insensitive) is treated as belonging to it.
  Anything that can't be matched is printed at the end so you can fix filenames
  or add an alias by hand - nothing is silently dropped.

Output:
  data/textual/all_articles_sentiment.csv  (every matched article, scored - for debugging)
  data/textual/daily_sentiment.csv         (aggregated per symbol/day - feeds compute_features.py)
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
NEWS_DIR = DATA_DIR / "news"
TEXTUAL_DIR = DATA_DIR / "textual"
PROCESSED_DIR = DATA_DIR / "processed"

SOURCE_WEIGHTS = {
    "nse": 1.0,
    "screener": 0.8,
    "news_rss": 0.6,
}
# below this, day's sentiment is zeroed (not enough evidence)
MIN_POSTS_FOR_SIGNAL = 2


# ============================================================
# STEP 1: Canonical symbol list + fuzzy file matching
# ============================================================
def get_canonical_symbols():
    files = list(PROCESSED_DIR.glob("*_cleaned.csv"))
    symbols = sorted(set(f.stem.replace("_cleaned", "").replace(
        "_NS", "").upper() for f in files))
    print(f"[Symbols] {len(symbols)} canonical symbols from data/processed/")
    return symbols


def match_file_to_symbol(filename_stem, symbols):
    """Return the canonical symbol this file belongs to, or None.
    Checks both directions - filename can be a prefix of the symbol
    (e.g. "eurotex" -> "EUROTEXIND") or the symbol a prefix of the
    filename (e.g. "ANMOLIND_extra" -> "ANMOLIND")."""
    up = filename_stem.upper()
    MIN_OVERLAP = 5  # guard against short generic prefixes false-matching
    for sym in symbols:
        if len(sym) < MIN_OVERLAP or len(up) < MIN_OVERLAP:
            if up == sym:
                return sym
            continue
        if up.startswith(sym) or sym.startswith(up):
            return sym
    return None


# ============================================================
# STEP 2: Load + standardize each source type
# ============================================================
def load_news_rss(symbols):
    files = glob.glob(str(NEWS_DIR / "*_news.csv"))
    frames = []
    unmatched = []

    for f in files:
        stem = Path(f).stem.replace("_news", "")
        if "moneycontrol" in stem.lower():
            continue  # handled separately, currently empty anyway
        sym = match_file_to_symbol(stem, symbols)
        if sym is None:
            unmatched.append(f)
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
            continue
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("published") or cols.get("date")
        title_col = cols.get("title")
        if date_col is None or title_col is None:
            print(
                f"  [WARN] {f} missing expected columns, has: {df.columns.tolist()}")
            continue

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "headline": df[title_col].fillna(""),
        })
        out["text"] = out["headline"]
        # scalar assigned AFTER frame has real rows -> broadcasts correctly
        out["symbol"] = sym
        out["source_type"] = "news_rss"
        frames.append(out.dropna(subset=["date"]))

    if unmatched:
        print(
            f"[News RSS] {len(unmatched)} files could not be matched to a symbol:")
        for u in unmatched:
            print(f"    - {u}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "date", "source_type", "headline", "text"])
    print(
        f"[News RSS] {len(combined)} rows loaded from {len(frames)} matched files")
    return combined


def load_screener(symbols):
    files = glob.glob(str(TEXTUAL_DIR / "*_NS_announcements.csv")) + \
        glob.glob(str(TEXTUAL_DIR / "*_announcements.csv"))
    # exclude anything that's actually inside an "_announcements" directory pattern
    files = [f for f in files if os.path.isfile(f)]

    frames = []
    unmatched = []

    for f in files:
        stem = Path(f).stem.replace("_NS_announcements",
                                    "").replace("_announcements", "")
        sym = match_file_to_symbol(stem, symbols)
        if sym is None:
            unmatched.append(f)
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
            continue
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date")
        head_col = cols.get("headline")
        if date_col is None or head_col is None:
            print(
                f"  [WARN] {f} missing expected columns, has: {df.columns.tolist()}")
            continue

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "headline": df[head_col].fillna(""),
        })
        out["text"] = out["headline"]
        out["symbol"] = sym
        out["source_type"] = "screener"
        frames.append(out.dropna(subset=["date"]))

    if unmatched:
        print(
            f"[Screener] {len(unmatched)} files could not be matched to a symbol:")
        for u in unmatched:
            print(f"    - {u}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "date", "source_type", "headline", "text"])
    print(
        f"[Screener] {len(combined)} rows loaded from {len(frames)} matched files")
    return combined


def load_nse_announcements(symbols):
    dirs = glob.glob(str(TEXTUAL_DIR / "*_announcements"))
    frames = []
    unmatched = []

    for d in dirs:
        if not os.path.isdir(d):
            continue
        stem = Path(d).stem.replace("_announcements", "")
        sym = match_file_to_symbol(stem, symbols)
        f = os.path.join(d, "announcements.csv")
        if not os.path.exists(f):
            continue
        if sym is None:
            unmatched.append(d)
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
            continue
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date")
        subj_col = cols.get("subject")
        text_col = cols.get("text")
        if date_col is None:
            print(
                f"  [WARN] {f} missing date column, has: {df.columns.tolist()}")
            continue

        subj = df[subj_col].fillna("").astype(
            str) if subj_col else pd.Series([""] * len(df))
        text = df[text_col].fillna("").astype(str).str.slice(
            0, 1000) if text_col else pd.Series([""] * len(df))

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "headline": subj,
        })
        out["text"] = (subj + " " + text) if text_col else subj
        out["symbol"] = sym
        out["source_type"] = "nse"
        frames.append(out.dropna(subset=["date"]))

    if unmatched:
        print(
            f"[NSE] {len(unmatched)} folders could not be matched to a symbol:")
        for u in unmatched:
            print(f"    - {u}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "date", "source_type", "headline", "text"])
    print(
        f"[NSE] {len(combined)} rows loaded from {len(frames)} matched folders")
    return combined


# ============================================================
# STEP 3: Score everything with FinBERT ONCE (consistent convention)
# ============================================================
def score_all(df):
    if df.empty:
        df["sentiment_score"] = []
        return df

    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(
        f"[FinBERT] Loading model (device={'GPU' if device == 0 else 'CPU'})...")
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                       tokenizer="ProsusAI/finbert", device=device)

    texts = df["text"].fillna("").astype(str).str.slice(0, 1000).tolist()
    print(f"[FinBERT] Scoring {len(texts)} texts...")

    scores = []
    batch_size = 32 if device == 0 else 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            results = finbert(batch, truncation=True)
        except Exception as e:
            print(f"  [WARN] batch {i} failed: {e}")
            results = [{"label": "neutral", "score": 0.0}] * len(batch)
        for r in results:
            label = r["label"].lower()
            conf = float(r["score"])
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
        if i % (batch_size * 10) == 0:
            print(f"  scored {i + len(batch)}/{len(texts)}")

    df["sentiment_score"] = scores
    return df


# ============================================================
# STEP 4: Aggregate to daily per symbol, source-weighted
# ============================================================
def aggregate_daily(df):
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "SentimentDelta_t", "NewsCount_t", "C_quality"])

    df = df.copy()
    df["weight"] = df["source_type"].map(SOURCE_WEIGHTS).fillna(0.4)
    df["w_score"] = df["weight"] * df["sentiment_score"]

    # vectorized aggregation instead of groupby().apply(Series) - avoids
    # pandas version-dependent shape bugs that dropped columns silently
    grouped = df.groupby(["symbol", "date"], as_index=False).agg(
        NewsCount_t=("sentiment_score", "count"),
        weight_sum=("weight", "sum"),
        w_score_sum=("w_score", "sum"),
        nse_count=("source_type", lambda s: int((s == "nse").sum())),
        screener_count=("source_type", lambda s: int((s == "screener").sum())),
        news_rss_count=("source_type", lambda s: int((s == "news_rss").sum())),
    )

    grouped["SentimentDelta_t"] = np.where(
        grouped["weight_sum"] > 0,
        grouped["w_score_sum"] / grouped["weight_sum"],
        0.0
    )
    grouped.loc[grouped["NewsCount_t"] <
                MIN_POSTS_FOR_SIGNAL, "SentimentDelta_t"] = 0.0
    grouped["C_quality"] = (grouped["NewsCount_t"] / 20).clip(upper=1.0)

    daily = grouped[["symbol", "date", "SentimentDelta_t", "NewsCount_t", "C_quality",
                     "nse_count", "screener_count", "news_rss_count"]]
    return daily


def main():
    symbols = get_canonical_symbols()

    news_rss = load_news_rss(symbols)
    screener = load_screener(symbols)
    nse = load_nse_announcements(symbols)

    all_articles = pd.concat([news_rss, screener, nse], ignore_index=True)
    all_articles = all_articles.dropna(subset=["date"])
    print(f"\n[Combine] {len(all_articles)} total articles across all sources, "
          f"{all_articles['symbol'].nunique()} symbols covered")

    all_articles = score_all(all_articles)

    TEXTUAL_DIR.mkdir(parents=True, exist_ok=True)
    all_articles_file = TEXTUAL_DIR / "all_articles_sentiment.csv"
    all_articles.to_csv(all_articles_file, index=False)
    print(f"[Save] -> {all_articles_file}")

    daily = aggregate_daily(all_articles)
    daily_file = TEXTUAL_DIR / "daily_sentiment.csv"
    daily.to_csv(daily_file, index=False)
    print(f"[Save] -> {daily_file}")
    print(
        f"[Done] {len(daily)} symbol-day rows, {daily['symbol'].nunique() if not daily.empty else 0} symbols")


if __name__ == "__main__":
    main()
