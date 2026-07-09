"""
compute_erts.py
----------------
Fuses data/erts/features/all_features.csv into the final ERTS score per
symbol-day.

Components:
  A_t  Accumulation  - delivery-backed accumulation vs wash-trading signal
  B_t  Pump          - price/volatility anomaly signal
  C_t  Deception     - sentiment mismatch signal (imputed when data is thin)
  E_t  Dampener      - corporate-action discount (SKIPPED for now, defaults to 1.0
                       i.e. no dampening - add later without touching this file)

Missing-data handling (the important part):
  If C_quality (sentiment confidence, from compute_features.py) is below
  0.3, we do NOT let a missing/thin news day zero out the whole score.
  Instead C_t is imputed as a function of A_t and B_t themselves:
      C_t_imputed = 0.3 + 0.4 * A_t * B_t
  This says: "if price+volume already look manipulated, assume deception
  risk is at least moderate even without confirming news" - rather than
  silently treating "no news scraped yet" as "definitely not a pump".

Output:
  data/erts/scores/erts_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_FILE = Path("data/erts/features/all_features.csv")
OUT_DIR = Path("data/erts/scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_QUALITY_THRESHOLD = 0.3

# Percentile-based thresholds instead of fixed absolute cutoffs.
# Pump-and-dump is rare - with no labeled data yet, flag the tail of the
# distribution rather than guessing absolute numbers. Once you validate
# against the SEBI-confirmed cases (SADBHAV, SURANASOL, GOKUL, VAKRANGEE,
# PAR), replace these with fixed cutoffs tuned to catch those events.
PERCENTILE_THRESHOLDS = {
    "watch": 0.85,     # top 15%
    "warning": 0.95,   # top 5%
    "critical": 0.99,  # top 1%
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_unit_interval(z, scale=1.0):
    """Map an unbounded z-score-like signal to [0, 1] via sigmoid."""
    return sigmoid(z * scale)


def compute_A_t(df):
    """Accumulation: delivery-backed volume surge (SAS_t, M_t) penalized by
    wash-trading signal (WTP_z, volume surge with LOW delivery)."""
    raw = 0.5 * df["SAS_t"] + 0.3 * df["M_t"] + 0.4 * df["WTP_z"]
    return to_unit_interval(raw)


def compute_B_t(df):
    """Pump: price anomaly (IF_norm), persistence (M_t), volatility asymmetry (V_t)."""
    v_signal = np.tanh(
        np.log(df["V_t"].clip(lower=1e-3)))  # symmetric around 0
    raw = 0.6 * df["IF_norm"] + 0.3 * df["M_t"] + 0.2 * v_signal
    return to_unit_interval(raw)


def compute_C_t(df, A_t, B_t):
    """Deception: sentiment-based, imputed from A_t*B_t when data is thin."""
    sentiment_signal = to_unit_interval(df["SentimentDelta_t"], scale=2.0)

    imputed = 0.3 + 0.4 * (A_t * B_t)
    thin_data = df["C_quality"] < C_QUALITY_THRESHOLD

    C_t = np.where(thin_data, imputed, sentiment_signal)
    return pd.Series(C_t, index=df.index), thin_data


def classify(score, cutoffs):
    if score >= cutoffs["critical"]:
        return "CRITICAL"
    elif score >= cutoffs["warning"]:
        return "WARNING"
    elif score >= cutoffs["watch"]:
        return "WATCH"
    return "NORMAL"


def main():
    if not FEATURES_FILE.exists():
        print(
            f"[ERROR] {FEATURES_FILE} not found. Run compute_features.py first.")
        return

    df = pd.read_csv(FEATURES_FILE)
    print(
        f"[ERTS] Loaded {len(df)} feature rows, {df['symbol'].nunique()} symbols")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    A_t = compute_A_t(df)
    B_t = compute_B_t(df)
    C_t, imputed_flag = compute_C_t(df, A_t, B_t)

    # E_t dampener - not built yet (no corporate-action / circuit data wired up).
    # Defaults to 1.0 (no dampening). Add a real E_t column later and multiply
    # it in here without changing anything else.
    E_t = 1.0

    # Weighted geometric-style fusion: emphasizes agreement across all three
    # signals rather than letting one high component alone trigger an alert.
    eps = 1e-6
    fusion = (A_t.clip(lower=eps) ** 0.4) * (B_t.clip(lower=eps)
                                             ** 0.4) * ((0.3 + 0.7 * C_t).clip(lower=eps) ** 0.2)
    ERTS_t = (fusion * E_t).clip(0, 1)

    out = df[["symbol", "date"]].copy()
    out["A_t"] = A_t
    out["B_t"] = B_t
    out["C_t"] = C_t
    out["C_imputed"] = imputed_flag
    out["E_t"] = E_t
    out["ERTS_t"] = ERTS_t

    # Diagnostics BEFORE classifying - this is how you catch calibration
    # problems like saturation (values clustered at 0/1) instead of a
    # healthy spread
    print("\n[Diagnostics] Component distributions (should NOT cluster near 0 or 1):")
    print(out[["A_t", "B_t", "C_t", "ERTS_t"]].describe(
        percentiles=[.5, .85, .95, .99]))

    cutoffs = {
        "watch": ERTS_t.quantile(PERCENTILE_THRESHOLDS["watch"]),
        "warning": ERTS_t.quantile(PERCENTILE_THRESHOLDS["warning"]),
        "critical": ERTS_t.quantile(PERCENTILE_THRESHOLDS["critical"]),
    }
    print(f"\n[Calibration] Percentile-based cutoffs derived from this run's data:")
    print(f"  WATCH    >= {cutoffs['watch']:.4f}  (top 15%)")
    print(f"  WARNING  >= {cutoffs['warning']:.4f}  (top 5%)")
    print(f"  CRITICAL >= {cutoffs['critical']:.4f}  (top 1%)")

    out["alert_level"] = out["ERTS_t"].apply(lambda s: classify(s, cutoffs))

    out_file = OUT_DIR / "erts_scores.csv"
    out.to_csv(out_file, index=False)

    print(f"[Save] -> {out_file}")
    print(f"\n[Alert distribution]")
    print(out["alert_level"].value_counts())
    print(f"\n[Imputed C_t] {imputed_flag.sum()} / {len(out)} rows had thin/no sentiment data "
          f"and used the A_t*B_t imputation")

    top = out.sort_values("ERTS_t", ascending=False).head(15)
    print(f"\n[Top 15 highest ERTS scores]")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
