import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# =====================================
# CONFIG
# =====================================

START_DATE = "2023-01-01"

# =====================================
# DOWNLOAD DATA
# =====================================

print("Downloading data...")

stock = yf.download(
    "ANMOL.BO",
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

market = yf.download(
    "^NSEI",
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

# Small-cap proxy
smallcap = yf.download(
    "^NSEMDCP50",
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

print("Downloaded Successfully")

# =====================================
# EXTRACT CLOSE SERIES
# =====================================

def get_close(df):

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        return close.iloc[:, 0]

    return close

stock_close = get_close(stock)
market_close = get_close(market)
small_close = get_close(smallcap)

# =====================================
# RETURNS
# =====================================

stock_ret = stock_close.pct_change()
market_ret = market_close.pct_change()
small_ret = small_close.pct_change()

# =====================================
# BUILD FACTORS
# =====================================

df = pd.DataFrame(index=stock_ret.index)

df["Stock"] = stock_ret
df["MKT"] = market_ret

# SMB proxy
df["SMB"] = small_ret - market_ret

# HML proxy
df["HML"] = (
    market_ret.rolling(20).mean()
    - market_ret.rolling(60).mean()
)

df = df.dropna()

print("\nObservations:", len(df))

# =====================================
# FAMA FRENCH REGRESSION
# =====================================

X = df[["MKT", "SMB", "HML"]]

X = sm.add_constant(X)

y = df["Stock"]

model = sm.OLS(y, X).fit()

print("\n")
print("=" * 80)
print("ANMOL FAMA FRENCH 3 FACTOR MODEL")
print("=" * 80)

print(model.summary())

# =====================================
# SAVE FACTOR COEFFICIENTS
# =====================================

results = pd.DataFrame({
    "Factor": model.params.index,
    "Coefficient": model.params.values
})

results.to_csv(
    "ANMOL_ff3_results.csv",
    index=False
)

# =====================================
# ABNORMAL RETURNS
# =====================================

df["Predicted_Return"] = model.predict(X)

df["Abnormal_Return"] = (
    df["Stock"]
    - df["Predicted_Return"]
)

df.to_csv(
    "ANMOL_abnormal_returns.csv"
)

# =====================================
# TOP ABNORMAL DAYS
# =====================================

print("\nTop 10 Abnormal Return Days")

top_days = (
    df["Abnormal_Return"]
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

print(top_days)

print("\nFiles Saved:")
print("ANMOL_ff3_results.csv")
print("ANMOL_abnormal_returns.csv")