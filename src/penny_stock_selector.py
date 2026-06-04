"""
SSEWS Project — Stage 1: Penny Stock Universe Selection
========================================================
Fetches all NSE EQ-series stocks, applies hard filters,
scores each by pump-risk, and saves a ranked CSV universe.

KEY CHANGES FROM PREVIOUS VERSION:
  - Removed unreliable yfinance marketCap filter (was dropping 1550 valid stocks)
  - Replaced marketCap gate with avg daily traded value (computed from OHLCV — reliable)
  - Raised avg_value threshold to Rs 200L (Rs 50L was too tight)
  - Lowered min_history to 45 days for SELECTION only (GARCH enforces 90 at model time)
  - Lowered dead_stock threshold to 15 days (slightly tighter)

Run time : ~25-40 minutes (2000 stocks, batched yfinance calls)
Output   : penny_stock_universe.csv
"""

import numpy as np
import time
import yfinance as yf
import pandas as pd
import requests
import io


# ══════════════════════════════════════════════════════════
#  CONFIGURATION  — only touch this block
# ══════════════════════════════════════════════════════════
CFG = {
    # ── Hard filters ──────────────────────────────────────
    "price_max_rs":        100,   # Max closing price (Rs)
                                  # Rationale: retail investors can't afford
                                  # high-priced shares in bulk; operators target
                                  # cheap stocks to create volume illusion.

    # NOTE: marketCap filter REMOVED.
    # yfinance marketCap for NSE micro-caps is wrong ~40% of the time.
    # It was dropping 1550 stocks — clearly not all large-caps.
    # We use avg daily traded value instead (computed from OHLCV = reliable).

    "avg_value_max_lakh":  200,   # Max avg daily traded value (Rs lakh)
                                  # Replaces market cap as size/liquidity gate.
                                  # Rs 200L/day is still very illiquid by NSE
                                  # standards — fully pumpable territory.
                                  # pump_risk_score will rank within this pool.

    "min_history_days":    45,    # Min trading days available — FOR SELECTION ONLY
                                  # Lowered from 90 because:
                                  #   (a) newly listed stocks are prime pump targets
                                  #   (b) circuit-hit stocks have gaps that reduce count
                                  # GARCH will re-enforce 90-day minimum at model time.
                                  # Do NOT raise this above 60 during selection.

    "dead_stock_zero_vol": 15,    # Flag as dead if >15 zero-volume days in window
                                  # Dead stocks score high on Amihud (artificially)
                                  # because near-zero volume makes |ret|/vol blow up.

    # ── Scoring parameters ────────────────────────────────
    "vol_spike_multiple":  3,     # Volume spike = day where vol > N × 20-day MA
                                  # 3× is cited in Aggarwal & Wu (2006) and
                                  # Comerton-Forde & Putniņš (2015).
                                  # Sensitivity analysis: try 2×, 3×, 5× in paper.

    # ── Runtime parameters ────────────────────────────────
    "batch_size":          50,    # Stocks per yfinance batch
    # Seconds between batches (avoids rate limits)
    "sleep_between_batch": 2,
    "max_retries":         3,     # Retry attempts per ticker on network error

    # ── Output ────────────────────────────────────────────
    "output_path":         "penny_stock_universe.csv",
}


# ══════════════════════════════════════════════════════════
#  STEP 1 — Fetch master NSE equity list
# ══════════════════════════════════════════════════════════
def fetch_nse_equity_list():
    """
    Downloads the complete NSE EQ-series equity list.
    Source: NSE public archive (no authentication needed).
    Strips warrants, rights, ETFs — keeps only regular equity (EQ series).
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }

    print("=" * 60)
    print("  SSEWS — Penny Stock Universe Selection")
    print("=" * 60)
    print("\n[Step 1] Fetching master NSE equity list...")

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df = df[df[' SERIES'].str.strip() == 'EQ']
    df['SYMBOL'] = df['SYMBOL'].str.strip()

    print(f"  ✓ Total NSE EQ-series stocks found: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════
#  STEP 2 — Download OHLCV for every symbol
# ══════════════════════════════════════════════════════════
def fetch_ticker_with_retry(ticker_str, max_retries):
    """
    Fetches 90-day OHLCV + market cap for one ticker.
    Uses exponential backoff on HTTP 429 (rate limit) errors.

    WHY retry instead of bare except-pass:
      yfinance 429s are transient — the stock is real but the server
      is throttling us. Silently dropping it makes your universe
      non-reproducible (two runs give different stocks). Retry fixes this.

    Returns: (hist_df, market_cap_inr) or (None, None) on failure.
    market_cap_inr may be None — that is handled downstream, not here.
    """
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_str)
            hist = ticker.history(period="90d", interval="1d")

            # Genuinely no data (delisted, bad symbol) — don't retry
            if hist.empty or len(hist) < 5:
                return None, None

            info = ticker.info
            market_cap = info.get('marketCap', None)

            # Fallback: price × shares outstanding if marketCap missing.
            # NOTE: sharesOutstanding from yfinance for Indian stocks often
            # includes promoter locked-in shares, so this may overestimate.
            # We store it but do NOT use it as a hard filter for this reason.
            if market_cap is None:
                shares = info.get('sharesOutstanding', None)
                if shares:
                    market_cap = float(hist['Close'].iloc[-1]) * float(shares)

            return hist, market_cap

        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in [
                                '429', 'rate', 'throttl', 'too many'])
            if is_rate_limit and attempt < max_retries - 1:
                wait = (2 ** attempt) * 3      # 3s → 6s → 12s
                print(
                    f"    [Rate limit] {ticker_str} — waiting {wait}s then retrying...")
                time.sleep(wait)
            else:
                break   # Non-retryable (bad symbol, delisted, SSL error, etc.)

    return None, None


def get_stock_metrics(symbols):
    """
    Fetches 90-day OHLCV for every symbol in batches.

    Per-stock metrics stored:
      symbol              — NSE ticker (without .NS suffix)
      last_close          — most recent closing price (Rs)
      market_cap_cr       — market cap in Rs crore (may be None — unreliable)
      market_cap_missing  — True if yfinance had no marketCap data
      avg_daily_value_L   — 60-day mean of (Close × Volume) in Rs lakh (reliable)
      days_available      — number of trading days returned
      hist                — full OHLCV DataFrame (used for scoring, dropped at save)
    """
    results = []
    missing_mcap_count = 0
    failed_count = 0
    symbols_ns = [s + ".NS" for s in symbols]
    total_batches = (len(symbols_ns) +
                     CFG["batch_size"] - 1) // CFG["batch_size"]

    print(f"\n[Step 2] Downloading data for {len(symbols_ns)} symbols...")
    print(f"  Estimated time: 25–40 minutes\n")

    for i in range(0, len(symbols_ns), CFG["batch_size"]):
        batch = symbols_ns[i: i + CFG["batch_size"]]
        batch_num = i // CFG["batch_size"] + 1
        print(
            f"  Batch {batch_num:>3}/{total_batches} — {len(batch)} stocks...", end="  ")

        batch_ok = 0
        for ticker_str in batch:
            hist, market_cap = fetch_ticker_with_retry(
                ticker_str, CFG["max_retries"])

            if hist is None:
                failed_count += 1
                continue

            last_close = float(hist['Close'].iloc[-1])
            hist['tv'] = hist['Close'] * hist['Volume']
            avg_daily_value = float(hist['tv'].tail(60).mean())

            missing_mcap = (market_cap is None)
            if missing_mcap:
                missing_mcap_count += 1

            results.append({
                'symbol':            ticker_str.replace('.NS', ''),
                'last_close':        round(last_close, 2),
                'market_cap_cr':     round(market_cap / 1e7, 2) if market_cap else None,
                'market_cap_missing': missing_mcap,
                'avg_daily_value_L': round(avg_daily_value / 1e5, 2),
                'days_available':    len(hist),
                'hist':              hist,
            })
            batch_ok += 1

        print(f"got {batch_ok}")
        time.sleep(CFG["sleep_between_batch"])

    print(f"\n  ✓ Successfully fetched : {len(results)} stocks")
    print(f"  ✗ Failed / no data     : {failed_count} stocks")
    print(f"  ⚠ Market cap missing   : {missing_mcap_count} stocks "
          f"(not used as filter — see CFG notes)")
    return results


# ══════════════════════════════════════════════════════════
#  STEP 3 — Hard quantitative filters
# ══════════════════════════════════════════════════════════
def apply_hard_filters(metrics_list):
    """
    Three hard filters — all computed from OHLCV (reliable data only).

    Filter 1 — Price ≤ Rs 100
      Why: Pump operators target low-priced shares. Retail investors
      buy more units of a cheap share, creating the volume illusion
      needed to attract momentum traders.

    Filter 2 — Avg daily traded value ≤ Rs 200 lakh  [REPLACES MARKET CAP]
      Why market cap was removed:
        yfinance marketCap dropped 1550 stocks — clearly not all large-caps.
        The field is wrong ~40% of the time for NSE micro-caps.
      Why avg traded value is better:
        It is computed directly from Close × Volume which we have reliably.
        It measures actual manipulability — a stock trading Rs 50L/day needs
        only ~Rs 2-5L of coordinated buying to move 3-5%. That is pumpable.
        A stock trading Rs 200L/day still only needs Rs 10-20L. Still pumpable.
        The pump_risk_score will rank within this pool.

    Filter 3 — At least 45 trading days of history
      Why 45 (not 90):
        90 days is needed for GARCH — but GARCH runs at model-training time,
        not at universe-selection time. Enforcing 90 here excluded many newly
        listed stocks and circuit-breaker-hit stocks that are prime pump targets.
        45 days gives enough history for the pump-risk scoring (Amihud needs ~20).
        GARCH will silently skip any stock with < 90 days when model runs.
    """
    filtered = []
    dropped = {'price': 0, 'liquidity': 0, 'history': 0}

    for s in metrics_list:

        if s['last_close'] > CFG["price_max_rs"]:
            dropped['price'] += 1
            continue

        if s['avg_daily_value_L'] > CFG["avg_value_max_lakh"]:
            dropped['liquidity'] += 1
            continue

        if s['days_available'] < CFG["min_history_days"]:
            dropped['history'] += 1
            continue

        filtered.append(s)

    print(f"\n[Step 3] Hard Filter Results")
    print(f"  Input stocks                          : {len(metrics_list)}")
    print(
        f"  Dropped — price > Rs{CFG['price_max_rs']}             : {dropped['price']}")
    print(
        f"  Dropped — avg value > Rs{CFG['avg_value_max_lakh']}L        : {dropped['liquidity']}")
    print(
        f"  Dropped — history < {CFG['min_history_days']} days          : {dropped['history']}")
    print(f"  ✓ After filters                       : {len(filtered)}")
    return filtered


# ══════════════════════════════════════════════════════════
#  STEP 4 — Remove dead / suspended stocks
# ══════════════════════════════════════════════════════════
def flag_dead_stocks(stocks):
    """
    Removes stocks that score high on illiquidity NOT because they are
    pump targets, but because they are effectively dead / suspended.

    The distinction:
      Genuine pump target  → trades sporadically, but real buyers exist.
                             Zero-volume days are scattered, not consecutive.
      Dead / suspended     → hasn't traded in weeks. Long runs of zero volume.

    Why this matters for scoring:
      The Amihud illiquidity ratio = |return| / rupee_volume.
      When volume approaches zero, this ratio blows up to very large values.
      Dead stocks would artificially dominate the top of the pump-risk ranking
      without this filter — they look maximally illiquid because they're dead,
      not because they're manipulable.

    Threshold: > 15 zero-volume days in the 90-day window.
    """
    alive = []
    dead_list = []

    for s in stocks:
        zero_vol = int((s['hist']['Volume'] == 0).sum())
        s['zero_vol_days'] = zero_vol
        s['likely_dead'] = zero_vol > CFG["dead_stock_zero_vol"]

        if s['likely_dead']:
            dead_list.append(s)
        else:
            alive.append(s)

    print(f"\n[Step 4] Dead Stock Filter")
    print(f"  Flagged dead / suspended  : {len(dead_list)}")
    print(f"  ✓ Actively traded stocks  : {len(alive)}")
    return alive


# ══════════════════════════════════════════════════════════
#  STEP 5 — Pump-risk scoring
# ══════════════════════════════════════════════════════════
def compute_universe_amihud_p95(stocks):
    """
    Computes the 95th-percentile Amihud illiquidity value across the
    entire filtered universe, used as the normalisation denominator.

    Academic basis: Amihud (2002) "Illiquidity and stock returns"
    Journal of Financial Markets — normalising by a data-driven
    percentile rather than a hardcoded constant is the standard approach.

    Interpretation:
      A stock AT the 95th percentile → Amihud sub-score ≈ 100
      The median stock              → Amihud sub-score ≈ 50
      The most liquid stock         → Amihud sub-score ≈ 0
    """
    all_amihud = []
    for s in stocks:
        h = s['hist'].copy()
        h['ret_abs'] = h['Close'].pct_change().abs()
        h['rupee_vol'] = h['Close'] * h['Volume']
        h['amihud'] = h['ret_abs'] / (h['rupee_vol'] + 1e-9)
        med = h['amihud'].median()
        if not pd.isna(med) and np.isfinite(med):
            all_amihud.append(med)

    p95 = float(np.percentile(all_amihud, 95)) if all_amihud else 1e-4
    print(f"\n[Step 5] Scoring")
    print(f"  Amihud 95th-percentile across universe: {p95:.3e}")
    return p95


def compute_pump_risk_score(stock, amihud_p95):
    """
    Scores each stock 0–100 on structural pump attractiveness.
    Higher score = more likely to be a pump target = higher monitoring priority.

    ── Component 1: Amihud Illiquidity  (weight 40%) ─────────────────────
      Formula : median( |daily_return| / (Close × Volume) )
      Meaning : price impact per rupee of trading — how easy it is for a
                small coordinated buy to move the price significantly.
      High    → stock is illiquid → easy to manipulate → HIGH RISK
      Source  : Amihud (2002), standard in market microstructure literature.
      Normalisation: divide by universe 95th percentile (data-driven, defensible).

    ── Component 2: Volume Spike Frequency  (weight 35%) ────────────────
      Formula : fraction of days where Volume > N × 20-day moving average
                (N = CFG["vol_spike_multiple"], default 3)
      Meaning : stocks that have ALREADY had unexplained volume spikes are
                statistically more likely to be repeat pump targets.
                Pump operators return to stocks they know move easily.
      High    → history of coordinated activity → HIGH RISK
      Threshold cite: Aggarwal & Wu (2006), Comerton-Forde & Putniņš (2015).

    ── Component 3: Price Acceleration  (weight 25%) ────────────────────
      Formula : standard deviation of 5-day rolling returns
      Meaning : erratic price history (large swings in both directions)
                suggests past episodes of price coordination or panic.
      High    → volatile, erratic history → MODERATE RISK signal
      Lowest weight because it can be caused by genuine news events.
      Normalised at 15% 5-day return std = score 100.

    ── Weight rationale ─────────────────────────────────────────────────
      Amihud (40%) : structural property — directly measures manipulability.
      Vol spikes (35%): behavioural evidence — observed past coordination.
      Price accel (25%): noisy signal — can be fundamental or manipulation.
    """
    h = stock['hist'].copy()

    # — Component 1: Amihud illiquidity —
    h['ret_abs'] = h['Close'].pct_change().abs()
    h['rupee_vol'] = h['Close'] * h['Volume']
    h['amihud'] = h['ret_abs'] / (h['rupee_vol'] + 1e-9)
    amihud_med = h['amihud'].median()
    amihud_med = 0.0 if (pd.isna(amihud_med) or not np.isfinite(
        amihud_med)) else float(amihud_med)
    amihud_score = min(amihud_med / (amihud_p95 + 1e-12), 1.0) * 100

    # — Component 2: Volume spike frequency —
    h['vol_ma20'] = h['Volume'].rolling(20, min_periods=5).mean()
    h['vol_ratio'] = h['Volume'] / (h['vol_ma20'] + 1)
    spike_freq = float(
        (h['vol_ratio'] > CFG["vol_spike_multiple"]).sum()) / len(h)
    vol_spike_score = min(spike_freq / 0.20, 1.0) * 100

    # — Component 3: Price acceleration —
    h['ret_5d'] = h['Close'].pct_change(5)
    price_accel = h['ret_5d'].std()
    price_accel = 0.0 if (pd.isna(price_accel) or not np.isfinite(
        price_accel)) else float(price_accel)
    price_accel_score = min(price_accel / 0.15, 1.0) * 100

    # — Weighted sum —
    score = (
        0.40 * amihud_score +
        0.35 * vol_spike_score +
        0.25 * price_accel_score
    )

    return (
        round(score, 2),
        round(amihud_score, 2),
        round(vol_spike_score, 2),
        round(price_accel_score, 2),
    )


# ══════════════════════════════════════════════════════════
#  STEP 6 — Save to CSV
# ══════════════════════════════════════════════════════════
def save_universe(stocks, output_path):
    """
    Saves the full ranked universe to CSV.

    Columns saved:
      symbol                — NSE ticker
      last_close_rs         — closing price (Rs)
      market_cap_cr         — market cap from yfinance (Rs crore, may be None)
      market_cap_missing    — flag: True = yfinance had no data
      avg_daily_value_lakh  — 60-day avg rupee turnover (Rs lakh) — our size proxy
      days_history          — trading days available in yfinance
      zero_vol_days         — days with exactly zero volume (quality indicator)
      pump_risk_score       — final composite score 0-100 (higher = more at risk)
      sub_amihud            — Amihud illiquidity sub-score 0-100
      sub_vol_spike         — volume spike frequency sub-score 0-100
      sub_price_accel       — price acceleration sub-score 0-100

    NOTE: 'hist' (the raw OHLCV DataFrame) is intentionally excluded.
    It will be re-fetched for the full 2021-2024 window in Stage 2.
    Storing it in CSV is not feasible (nested object) and not needed.
    """
    rows = []
    for s in stocks:
        rows.append({
            'symbol':               s['symbol'],
            'last_close_rs':        s['last_close'],
            'market_cap_cr':        s['market_cap_cr'],
            'market_cap_missing':   s['market_cap_missing'],
            'avg_daily_value_lakh': s['avg_daily_value_L'],
            'days_history':         s['days_available'],
            'zero_vol_days':        s['zero_vol_days'],
            'pump_risk_score':      s['pump_risk_score'],
            'sub_amihud':           s['sub_amihud'],
            'sub_vol_spike':        s['sub_vol_spike'],
            'sub_price_accel':      s['sub_price_accel'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    # ── Terminal summary ──
    total = len(df)
    top10_thresh = df['pump_risk_score'].quantile(0.90)
    top25_thresh = df['pump_risk_score'].quantile(0.75)
    median_score = df['pump_risk_score'].median()

    print(f"\n[Step 6] Saved {total} stocks → '{output_path}'")
    print(f"\n  ── Score distribution ──────────────────────")
    print(f"  Top 10% (score ≥ {top10_thresh:.1f}) : "
          f"{(df['pump_risk_score'] >= top10_thresh).sum()} stocks")
    print(f"  Top 25% (score ≥ {top25_thresh:.1f}) : "
          f"{(df['pump_risk_score'] >= top25_thresh).sum()} stocks")
    print(f"  Median score          : {median_score:.1f}")
    print(f"  Missing market cap    : {df['market_cap_missing'].sum()} stocks")

    print(f"\n  ── Summary statistics ──────────────────────")
    print(df[['last_close_rs', 'avg_daily_value_lakh',
              'pump_risk_score', 'sub_amihud',
              'sub_vol_spike',  'sub_price_accel']
             ].describe().round(2).to_string())

    return df


# ══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Fetch all NSE EQ symbols
    nse_df = fetch_nse_equity_list()
    all_symbols = nse_df['SYMBOL'].tolist()

    # 2. Download 90-day OHLCV for every symbol (slow — ~25-40 min)
    raw_metrics = get_stock_metrics(all_symbols)

    # 3. Apply hard filters (price, avg value, history)
    filtered = apply_hard_filters(raw_metrics)

    # 4. Remove dead / suspended stocks
    filtered = flag_dead_stocks(filtered)

    if len(filtered) == 0:
        print("\n  ✗ ERROR: No stocks survived filtering.")
        print("    Try raising avg_value_max_lakh or lowering min_history_days in CFG.")
        exit(1)

    # 5. Compute universe-level Amihud 95th percentile
    amihud_p95 = compute_universe_amihud_p95(filtered)

    # 6. Score every surviving stock
    print(f"  Scoring {len(filtered)} stocks...")
    for s in filtered:
        score, a, v, p = compute_pump_risk_score(s, amihud_p95)
        s['pump_risk_score'] = score
        s['sub_amihud'] = a
        s['sub_vol_spike'] = v
        s['sub_price_accel'] = p

    # 7. Sort by pump risk score (highest first)
    filtered.sort(key=lambda x: x['pump_risk_score'], reverse=True)

    # 8. Print top 30 to terminal for sanity check
    print(f"\n  ── Top 30 highest pump-risk stocks ─────────────────────────────")
    print(f"  {'Symbol':<14} {'Price(Rs)':>9} {'MCap(Cr)':>9} "
          f"{'AvgVal(L)':>9} {'Score':>7} {'Amihud':>7} {'VolSpk':>7} {'PrAcc':>6}")
    print("  " + "─" * 72)
    for s in filtered[:30]:
        mc = f"{s['market_cap_cr']:.0f}" if s['market_cap_cr'] else "N/A"
        print(f"  {s['symbol']:<14} {s['last_close']:>9.2f} {mc:>9} "
              f"{s['avg_daily_value_L']:>9.2f} "
              f"{s['pump_risk_score']:>7.1f} "
              f"{s['sub_amihud']:>7.1f} "
              f"{s['sub_vol_spike']:>7.1f} "
              f"{s['sub_price_accel']:>6.1f}")

    # 9. Save full ranked universe to CSV
    universe_df = save_universe(filtered, CFG["output_path"])

    print(f"\n{'=' * 60}")
    print(f"  DONE.")
    print(f"  {len(universe_df)} stocks saved to '{CFG['output_path']}'")
    print(f"  For your study, use the TOP 150-200 by pump_risk_score.")
    print(f"  Next: run Stage 2 (full 2021-2024 history download)")
    print(f"{'=' * 60}\n")
