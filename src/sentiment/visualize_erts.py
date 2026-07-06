"""
visualize_erts.py
------------------
Interactive 3-panel linked visualizer: price (candlestick), volume, ERTS score.
All three panels share the x-axis, so zooming/panning on any one moves all three.

Color-codes candidate pump-and-dump days (from your 5-rule heuristic) against
what ERTS actually flagged, so you can eyeball agreement/disagreement:

  GREEN  - heuristic flags anomaly AND ERTS also flags it   (agreement)
  YELLOW - heuristic flags anomaly but ERTS did NOT flag it (ERTS missed it)
  RED    - ERTS flags it but heuristic does NOT              (possible false
                                                                positive - or ERTS
                                                                caught something the
                                                                heuristic can't see)

HOW TO USE:
  1. Edit SYMBOL below.
  2. Run: python visualize_erts.py
  3. A browser tab opens with the interactive chart. It's also saved as an
     HTML file under data/erts/scores/ so you can reopen it later.

NOTE on rule 3 ("no legitimate news"): this needs corporate-action/earnings
data that hasn't been built yet. It's a placeholder (always True, i.e. never
disqualifies) - wire in real data later without touching anything else here.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# EDIT THIS to switch stocks
# ============================================================
SYMBOL = "PAR"
# ============================================================

PROCESSED_DIR = Path("data/processed")
ERTS_FILE = Path("data/erts/scores/erts_scores.csv")
OUT_HTML = Path(f"data/erts/scores/{SYMBOL}_visualizer.html")

# Heuristic thresholds from your 5 rules - tune these if needed
PUMP_LOOKBACK_DAYS = 126      # ~6 months of trading days
PUMP_RETURN_THRESHOLD = 2.0   # 200% rise
DUMP_WINDOW_DAYS = 30
DUMP_DROP_THRESHOLD = 0.5     # 50% crash
VOLUME_SPIKE_MULT = 5.0
LOW_DELIVERY_PCT = 30.0

ERTS_ANOMALY_LEVELS = {"WARNING", "CRITICAL"}


def find_price_file(symbol):
    candidates = [
        PROCESSED_DIR / f"{symbol}_cleaned.csv",
        PROCESSED_DIR / f"{symbol}_NS_cleaned.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(PROCESSED_DIR.glob(f"{symbol}*_cleaned.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No processed OHLCV file found for symbol '{symbol}' in {PROCESSED_DIR}")


def compute_heuristic_anomaly(df):
    df = df.sort_values("Date").reset_index(drop=True)

    # Rule 1: price rose >200% versus its trailing 6-month low
    rolling_low = df["Close"].rolling(PUMP_LOOKBACK_DAYS, min_periods=20).min()
    pump_return = (df["Close"] / rolling_low) - 1
    df["pump_flag"] = pump_return > PUMP_RETURN_THRESHOLD

    # Rule 2: within the NEXT 30 trading days, does price crash >50% from here?
    future_min = df["Close"].iloc[::-
                                  1].rolling(DUMP_WINDOW_DAYS, min_periods=1).min().iloc[::-1]
    df["dump_flag"] = (future_min / df["Close"]) < (1 - DUMP_DROP_THRESHOLD)

    # Rule 5: volume spike vs its own 20-day average
    vol_ma = df["Volume"].rolling(20, min_periods=5).mean()
    df["volume_spike_flag"] = df["Volume"] > (VOLUME_SPIKE_MULT * vol_ma)

    # Rule 4: low delivery % (wash-trading signature)
    delv = df["Delivery_Percentage"] if "Delivery_Percentage" in df.columns else pd.Series(
        np.nan, index=df.index)
    df["low_delivery_flag"] = delv < LOW_DELIVERY_PCT

    # Rule 3: no legitimate news - PLACEHOLDER, always True (see module docstring)
    df["no_news_flag"] = True

    df["heuristic_anomaly"] = (
        df["pump_flag"] & df["dump_flag"] & df["volume_spike_flag"] &
        df["low_delivery_flag"] & df["no_news_flag"]
    )
    return df


def classify_agreement(row):
    heuristic = bool(row["heuristic_anomaly"])
    erts = row["alert_level"] in ERTS_ANOMALY_LEVELS
    if heuristic and erts:
        return "green"
    elif heuristic and not erts:
        return "yellow"
    elif erts and not heuristic:
        return "red"
    return None


def main():
    price_file = find_price_file(SYMBOL)
    print(f"[Load] Price data: {price_file}")
    price = pd.read_csv(price_file)
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price = price.dropna(subset=["Date"]).sort_values("Date")

    if not ERTS_FILE.exists():
        raise FileNotFoundError(
            f"{ERTS_FILE} not found - run compute_erts.py first")
    erts = pd.read_csv(ERTS_FILE)
    erts_sym = erts[erts["symbol"].str.upper() == SYMBOL.upper()].copy()
    if erts_sym.empty:
        raise ValueError(
            f"No ERTS scores found for symbol '{SYMBOL}' - check spelling / that it's in your 401")
    erts_sym["date"] = pd.to_datetime(erts_sym["date"], errors="coerce")

    price = compute_heuristic_anomaly(price)

    merged = price.merge(erts_sym, left_on="Date", right_on="date", how="left")
    merged["alert_level"] = merged["alert_level"].fillna("NORMAL")
    merged["agreement_color"] = merged.apply(classify_agreement, axis=1)

    print(
        f"[Heuristic] {int(merged['heuristic_anomaly'].sum())} candidate anomaly days found")
    print(
        f"[ERTS] {int((merged['alert_level'].isin(ERTS_ANOMALY_LEVELS)).sum())} days flagged WARNING/CRITICAL")
    print(f"[Agreement] green={int((merged['agreement_color'] == 'green').sum())}  "
          f"yellow={int((merged['agreement_color'] == 'yellow').sum())}  "
          f"red={int((merged['agreement_color'] == 'red').sum())}")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.5, 0.2, 0.3],
        subplot_titles=(f"{SYMBOL} - Price", "Volume", "ERTS Score")
    )

    fig.add_trace(go.Candlestick(
        x=merged["Date"], open=merged["Open"], high=merged["High"],
        low=merged["Low"], close=merged["Close"], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=merged["Date"], y=merged["Volume"], name="Volume", marker_color="steelblue"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=merged["Date"], y=merged["ERTS_t"], name="ERTS_t",
        mode="lines", line=dict(color="black", width=1.5)
    ), row=3, col=1)

    color_map = {"green": "green", "yellow": "gold", "red": "red"}
    for color_name, color_val in color_map.items():
        subset = merged[merged["agreement_color"] == color_name]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Date"], y=subset["ERTS_t"], mode="markers",
            marker=dict(color=color_val, size=10, symbol="circle",
                        line=dict(width=1, color="black")),
            name=f"{color_name} ({len(subset)})"
        ), row=3, col=1)

    fig.update_layout(
        height=900, title=f"ERTS Visualizer - {SYMBOL}",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    # this is what links zoom/pan across all 3 panels
    fig.update_xaxes(matches="x")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML))
    print(f"\n[Saved] {OUT_HTML} - reopen this file anytime in a browser")
    fig.show()


if __name__ == "__main__":
    main()
