# 🗺️ Pump Detection System — Codebase Map

> **Purpose**: Detect pump-and-dump activity in NSE (Indian stock exchange) penny stocks by fusing quantitative OHLCV signals, GARCH volatility models, NLP sentiment, and corporate-event scoring into a unified anomaly detection pipeline.
>
> **Stack**: Python · yfinance · jugaad-data · FinBERT (HuggingFace) · Isolation Forest · XGBoost · GARCH (arch) · Optuna · Plotly · Matplotlib

---

## 📐 High-Level Architecture

The system is organized as a staged pipeline:

1. **Universe Selection** (`penny_stock_selector.py`) — rank all NSE stocks by pump-risk score
2. **OHLCV Ingestion + Cleaning** (`ingestion.py`, `data_cleaning.py`, `delivery_ingestion.py`)
3. **Text Ingestion** (NSE API, Screener.in, Google RSS, Telegram, Moneycontrol)
4. **Feature Engineering** (11 quant features + GARCH volatility features)
5. **Sentiment Scoring** (FinBERT on news/announcements → daily aggregated scores)
6. **ERTS Feature Fusion** (`compute_features.py` merges OHLCV signals + sentiment)
7. **Anomaly Detection** (Isolation Forest, XGBoost walk-forward, Fama-French, ERTS score)
8. **Evaluation & Visualisation** (metric reports, backtested anomaly overlays, interactive charts)

---

## 📁 Root-Level Files

| File | Purpose |
|---|---|
| `penny_stock_universe.csv` | Final ranked output of `penny_stock_selector.py` — the master list of ~400 at-risk stocks |
| `stratified_training_universe.csv` | A stratified subset used as the official training universe for model stages |
| `commands.md` | Canonical reference of every CLI invocation in execution order — the operator's manual |
| `collaborations.md` | Project collaboration notes between team members |

---

## 📁 `src/data_pipeline/` — Data Ingestion & Preparation

### `penny_stock_selector.py`
**Stage 0 — Universe construction.** Downloads the full NSE EQ-series list, batch-fetches 90-day OHLCV from Yahoo Finance, applies three hard filters (price ≤ ₹100, avg traded value ≤ ₹200L, ≥45 trading days), removes dead stocks, and scores survivors 0–100 on a composite pump-risk metric (40% Amihud illiquidity + 35% volume-spike frequency + 25% price acceleration). Saves the ranked universe to `penny_stock_universe.csv` — **the entry point of the entire system**.

### `ingestion.py`
**Stage 1 — Historical OHLCV download.** Reads tickers from `penny_stock_universe.csv`, fetches 3 years of daily OHLCV from Yahoo Finance per ticker with exponential-backoff rate-limit handling. Saves one `*_raw.csv` per stock to `data/raw/`.

### `data_cleaning.py`
**Stage 1b — Data sanitization.** Reads every `*_raw.csv` from `data/raw/`, applies timezone de-localization, forward-fills prices on non-trading days, zero-fills volume, drops `Dividends`/`Stock_Splits` columns, enforces 4-decimal precision. Writes cleaned files to `data/processed/`. Provides `clean_financial_data()` — consumed by the batch `main()`.

### `delivery_ingestion.py`
**Stage 1c — Real NSE delivery data injection.** Iterates over all `*_cleaned.csv` files, fetches official NSE Bhavcopy delivery-percentage data via `jugaad_data.nse.stock_df`, merges it onto the Yahoo Finance OHLCV with forward-fill, overwrites processed files in-place.

### `nse_announcement2.py`
**Lightweight NSE corporate announcement fetcher.** Opens an authenticated `requests.Session` against the NSE API (`/api/corporate-announcements`) and extracts structured metadata (date, category, summary, PDF link) for symbol `ANMOL`. Saves to `data/textual/ANMOL_announcements/nse_announcements_clean.csv`. Simpler, metadata-only version of `nse_announcements.py` — no PDF text extraction.

### `nse_announcements.py`
**Full NSE announcement scraper with PDF text extraction.** Same session setup as `nse_announcement2.py` but additionally downloads each attached PDF, extracts raw text using PyMuPDF (`fitz`), and stores full extracted text alongside metadata. The rich text content feeds `sentiment_analysis.py`.

### `screener_data.py`
**Screener.in announcement scraper.** Sends a browser-mimicking HTTP request to `screener.in/company/{symbol}/`, parses HTML with BeautifulSoup, extracts clean headlines + dates using a subtractive text technique to avoid duplication. Exports to `data/textual/{symbol}_announcements.csv`. Feeds `build_daily_sentiment.py`.

### `stocks_config.py`
**Multi-stock Google News RSS fetcher.** Defines a config dict for 4 target stocks (ANMOLIND, ARTNIRMAN, GENUSPAPER, KEEPLEARN) with search queries and aliases. Queries Google News RSS via `feedparser`, alias-filters headlines, and saves per-ticker CSVs to `data/news/`. Feeds `batch_finbert.py` and `build_daily_sentiment.py`.

### `web_scrap.py`
**Moneycontrol news article scraper.** Uses the `newspaper` library to crawl and parse up to 30 stock-section articles from Moneycontrol, extracting title, body, URL, and publish date. Saves to `data/news/moneycontrol_news.csv`. Standalone proof-of-concept; noted as "currently empty anyway" in `build_daily_sentiment.py`.

### `merge_news.py`
**Early single-stock news merger (Eurotex-specific).** Loads Google RSS news (`eurotex_news.csv`) and Screener.in CSV, normalizes column names to a common schema, concatenates, deduplicates by `(date, title)`, and saves to `data/news/eurotex_master_news.csv`. Superseded at scale by `build_daily_sentiment.py`.

### `telegram_historical_scraper.py`
**Historical Telegram channel bulk scraper.** Uses Telethon async API to scrape ~20 configured stock-tip Telegram channels for all text messages from Jan 2021 to Dec 2024, handling FloodWaitError rate limits. Saves full message corpus to `master_raw_telegram_2021_2024.csv`. Not yet wired into the main ERTS pipeline — intended as a future social-media signal layer.

### `event.py`
**Early standalone NSE event scorer (variant).** An earlier standalone version of the event-scoring logic that reads from `ANMOL_announcements/nse_announcements_clean.csv`. Contains an identical `EVENT_MAP` and scoring logic, writes `daily_event_features.csv` directly to the `ANMOL_announcements/` folder. A predecessor to `src/features/event_features.py`.

---

## 📁 `src/features/` — Feature Engineering

### `feature_engineering.py`
**Core quantitative feature factory.** Reads every `*_cleaned.csv` from `data/processed/`, computes 11 pump-detection features per trading day: Log Return, Volume Shock Ratio (vs 20-day MA), Normalized High-Low Spread, Amihud Illiquidity Ratio, Delivery-Volume Divergence, Volatility Squeeze (10D vs 90D), Consecutive Positive Streak, Return Skewness, Gap-Up Momentum, Volume Gini Coefficient, and OBV Acceleration. Also computes the 20-day forward max drawdown as the supervised target. Saves `*_features.csv` to `data/features/`; consumed by `isolation_forest.py` / `inference.py`.

### `garch_features.py`
**GARCH(1,1) rolling volatility feature generator.** Reads cleaned OHLCV, fits a rolling GARCH(1,1) model (refitted every 20 days) to produce 1-day-ahead conditional volatility forecasts. Derives four output features: `GARCH_Vol_Forecast`, `GARCH_Omega/Alpha/Beta` (model parameters), `GARCH_Vol_Surprise` (|actual return| / forecast), `GARCH_Surprise_Score` (percentile rank). Outputs `*_garch.csv` to `data/garch/`; merged in by `isolation_forest.py`.

### `event_features.py`
**Rule-based NSE event scorer.** Reads `nse_announcements_clean.csv` for ANMOL, maps each `announcement_type` via a hand-crafted `EVENT_MAP` to a category and integer score (e.g. "Price movement" → +3, "Resignation of Independent director" → −3). Aggregates per day to count, total, and avg event score. Saves to `data/textual/ANMOL_announcements/daily_event_features.csv`. Feeds `visualize_event_scores.py`.

### `sentiment_analysis.py`
**FinBERT NSE-announcement sentiment scorer (multi-stock CLI).** Loads HuggingFace `ProsusAI/finbert`, scores each NSE announcement PDF text, converts labels to a signed confidence score (positive → +conf, negative → −conf), and aggregates to daily `avg_sentiment`, `max_sentiment`, `min_sentiment`. Supports `--symbol ALL` to batch-process every stock in `data/textual/`. Writes `{symbol}_daily_nse_sentiment.csv` to `data/sentiment/`.

### `batch_finbert.py`
**Batch FinBERT news-article scorer.** Loads FinBERT via the lower-level `AutoTokenizer` / `AutoModelForSequenceClassification` API. Iterates every `data/news/*_news.csv`, scores each article title, and saves both article-level scores and daily aggregated sentiment (article count, avg sentiment, positive/negative article counts) to `data/sentiment/`.

### `finbert_load.py`
**Single-stock FinBERT news scorer (Eurotex prototype).** Hardcoded for `eurotex_master_news.csv`. Runs FinBERT on `title + content`, aggregates to daily, saves `eurotex_article_sentiment.csv` and `eurotex_daily_sentiment.csv` to `data/sentiment/`. An early single-stock prototype superseded by the batch scripts.

---

## 📁 `src/models/` — Anomaly Detection & Predictive Models

### `inference.py`
**Main Isolation Forest inference orchestrator.** Imports `load_master_matrix()` and `find_best_parameters()` from `isolation_forest.py` to: (1) build the merged feature matrix (core + GARCH), (2) run Optuna hyperparameter search (25 trials), (3) fit the tuned IsolationForest on the full universe. Exports `data/results/full_universe_scores.csv` (all stocks + `Anomaly_Score`) and `data/results/pump_anomaly_targets.csv` (flagged rows). The **primary model output** consumed by evaluation scripts and `xgboost_predictor.py`.

### `isolation_forest.py`
**Isolation Forest loader + Optuna tuner library module.** Provides two reusable functions imported by `inference.py`: `load_master_matrix()` — assembles a global feature matrix by inner-joining `data/features/*_features.csv` with `data/garch/*_garch.csv` on date; `find_best_parameters()` — runs an Optuna study maximizing F1 of anomaly classification against the 20-day drawdown ground truth.

### `xgboost_predictor.py`
**XGBoost walk-forward regressor.** Reads `data/results/full_universe_scores.csv`, groups by stock, and runs an expanding-window walk-forward loop per stock: builds a "flattened" feature matrix (T0 through T-5 lags for every feature), trains XGBoost to predict tomorrow's `Anomaly_Score`, and logs predicted vs. actual. Saves `*_xgb_preds.csv` per stock to `data/predictions/xgboost/`. The lag-flattening gives XGBoost temporal memory. Consumed by `evaluate_xgboost.py` and `inspect_xgboost_anomalies.py`.

### `fama_french.py`
**Fama-French 3-Factor abnormal return calculator.** Downloads ANMOL (BSE `.BO`), Nifty 50, and Nifty Midcap 50 from Yahoo Finance, constructs MKT/SMB/HML factors, fits OLS regression via statsmodels, and computes daily abnormal returns (residuals). Saves factor coefficients and abnormal-return series to `data/results/`. A classical finance benchmark alongside the ML models.

---

## 📁 `src/sentiment/` — ERTS (Equity Risk Threat Score) Unified Pipeline

> This module is the **newer, production-grade pipeline** that supersedes several early one-off scripts. It fuses all signal sources into a single composite risk score.

### `build_daily_sentiment.py`
**Master multi-source sentiment aggregator.** Replaces `merge_news.py` + earlier standalone sentiment scripts. Derives a canonical symbol list from `data/processed/`, fuzzy-matches news files from all three sources (Google RSS, Screener CSVs, NSE announcement folders) by prefix, normalizes their schemas, runs all text through FinBERT in GPU-accelerated batches (32/batch GPU, 16/batch CPU), applies per-source confidence weights (NSE=1.0, Screener=0.8, RSS=0.6), and writes source-weighted daily sentiment to `data/textual/daily_sentiment.csv`. Feeds `compute_features.py`.

### `compute_features.py`
**ERTS input feature factory.** Reads all `data/processed/*_cleaned.csv`, computes ERTS-specific features: `R_t` (log return), `sigma_t` (rolling std), `V_t` (downside/upside vol asymmetry), `IF_norm` (z-scored return via tanh), `M_t` (EWMA anomaly persistence), `VR_t` (volume ratio), `SAS_t` (Smart Accumulation Score = delivery-backed volume surge z-score), `DVS_t` (delivery volume spike), `WTP_z` (wash-trading proxy z-score). Left-joins daily sentiment from `daily_sentiment.csv`, fills missing with zeros. Outputs `data/erts/features/all_features.csv`.

### `compute_erts.py`
**ERTS composite score engine.** Reads `all_features.csv` and computes the final score per symbol-day via three components: **A_t** (Accumulation — delivery-backed buy surge), **B_t** (Pump — price/vol anomaly persistence), **C_t** (Deception — sentiment mismatch, imputed from A_t×B_t when news is thin). Fuses via a weighted geometric mean (A^0.4 × B^0.4 × C^0.2). Classifies each row as NORMAL / WATCH / WARNING / CRITICAL using data-driven percentile cutoffs (top 15% / 5% / 1%). Saves `data/erts/scores/erts_scores.csv`.

### `visualize_erts.py`
**Interactive 3-panel ERTS visualizer.** For a configurable `SYMBOL`, loads OHLCV and ERTS scores, applies a 5-rule heuristic (200% pump, 50% crash within 30 days, 5× volume spike, <30% delivery, no news flag) to identify candidate P&D days, then cross-compares with ERTS WARNING/CRITICAL flags. Renders an interactive Plotly chart with 3 linked panels (candlestick price, volume, ERTS score) and color-coded agreement markers (green=both agree, yellow=heuristic only, red=ERTS only). Saves an HTML file to `data/erts/scores/`.

---

## 📁 `src/evaluation/` — Model Evaluation & Visualisation

### `evaluate_xgboost.py`
**XGBoost regression + classification evaluator.** Loads all `*_xgb_preds.csv` from `data/predictions/xgboost/`, concatenates into a master frame, and reports regression metrics (MAE, RMSE) and classification metrics (accuracy, precision, recall, F1, confusion matrix) using a configurable threshold (default −2.0) to label anomaly vs. normal days.

### `inspect_xgboost_anomalies.py`
**Severe anomaly inspector.** Scans all XGBoost prediction files, filters rows where `Actual_IF_Score ≤ −0.35` (the most extreme anomalies), and reports whether XGBoost's `Residual_Delta` also caught each one (True Positive vs. False Negative). A targeted diagnostic for the worst-case market events.

### `automated_verification.py`
**Backtested anomaly precision checker.** For each flagged date in `pump_anomaly_targets.csv`, looks 20 trading days forward in the price data and checks if a peak-to-trough drawdown >15% occurred. Prints per-flag status (VERIFIED / FAILED) and a summary precision hit rate. The text-only analogue of `visualize_anomalies.py`.

### `visualize_anomalies.py`
**Visual anomaly overlay chart.** For a given stock, plots Close Price and Volume with vertical dashed lines at each anomaly date. Lines are green if the anomaly was verified (>15% drawdown within 20 days) and red if unverified. Reads from `data/processed/*_cleaned.csv` and `data/results/pump_anomaly_targets.csv`.

### `visualize_event_scores.py`
**Event-score time-series plotter.** Loads `daily_event_features.csv` for ANMOL, plots `total_event_score` over time via Matplotlib, and saves the chart to `data/results/event_score_plot.png`.

### `visualize_news_sentiment.py`
**Dual-axis sentiment vs. news volume chart.** Loads `eurotex_daily_sentiment.csv`, filters to records from May 2026 onward, and plots average daily FinBERT sentiment (left y-axis) vs. article count as bars (right y-axis). Saves to `data/results/eurotex_sentiment_volume.png`.

---

## 📁 `scratch/` — Exploratory / Prototype Scripts

> These are throwaway experiments, **not part of the production pipeline**. They are preserved for reference only.

| File | Purpose |
|---|---|
| `ggoglerss.py` | One-off Google News RSS pull for "Eurotex Industries" via `feedparser` — prototype for `stocks_config.py` |
| `scarping_anmol.py` | Identical RSS pull prototype for "Anmol India Ltd." |
| `googlescrap.py` | PoC using `googlesearch` library to find Eurotex URLs on trusted financial domains |
| `strt.py` | Minimal NewsAPI client test for Vodafone (API key hardcoded); not integrated |
| `strt2.py` | EventRegistry API prototype for Vodafone Idea news 2021–2024; not integrated |

---

## 📁 `data/` — Directory Inventory

| Sub-directory | Contents / Produced By |
|---|---|
| `data/raw/` | Raw `*_raw.csv` OHLCV per stock → `ingestion.py` |
| `data/processed/` | Cleaned `*_cleaned.csv` with delivery data → `data_cleaning.py` + `delivery_ingestion.py` |
| `data/features/` | `*_features.csv` with 11 quant features → `feature_engineering.py` |
| `data/garch/` | `*_garch.csv` with GARCH volatility features → `garch_features.py` |
| `data/news/` | Per-stock Google RSS + Moneycontrol scrapes → `stocks_config.py`, `web_scrap.py` |
| `data/textual/` | NSE PDFs, Screener CSVs, merged master news, `daily_sentiment.csv` → various scrapers + `build_daily_sentiment.py` |
| `data/sentiment/` | FinBERT-scored article-level and daily sentiment CSVs → `batch_finbert.py`, `sentiment_analysis.py`, `finbert_load.py` |
| `data/results/` | `full_universe_scores.csv`, `pump_anomaly_targets.csv`, FF3 results, static charts → `inference.py`, `fama_french.py` |
| `data/predictions/xgboost/` | Per-stock `*_xgb_preds.csv` → `xgboost_predictor.py` |
| `data/erts/features/` | `all_features.csv` — merged ERTS feature matrix → `compute_features.py` |
| `data/erts/scores/` | `erts_scores.csv` + interactive HTML visualizers → `compute_erts.py`, `visualize_erts.py` |
| `data/logs/` | `garch_skipped.txt` — stocks skipped due to insufficient data → `garch_features.py` |

---

## 🔗 Complete Data-Flow Summary

```
penny_stock_selector.py   ──► penny_stock_universe.csv
ingestion.py              ──► data/raw/*_raw.csv
data_cleaning.py          ──► data/processed/*_cleaned.csv
delivery_ingestion.py     ──► data/processed/*_cleaned.csv  (enriched in-place)

nse_announcements.py      ──► data/textual/{sym}_announcements/announcements.csv
screener_data.py          ──► data/textual/{sym}_NS_announcements.csv
stocks_config.py          ──► data/news/{sym}_news.csv
web_scrap.py              ──► data/news/moneycontrol_news.csv

feature_engineering.py    ──► data/features/*_features.csv
garch_features.py         ──► data/garch/*_garch.csv
event_features.py         ──► data/textual/ANMOL_announcements/daily_event_features.csv

build_daily_sentiment.py  ──► data/textual/daily_sentiment.csv
compute_features.py       ──► data/erts/features/all_features.csv
compute_erts.py           ──► data/erts/scores/erts_scores.csv

inference.py              ──► data/results/full_universe_scores.csv
                          ──► data/results/pump_anomaly_targets.csv
xgboost_predictor.py      ──► data/predictions/xgboost/*_xgb_preds.csv
fama_french.py            ──► data/results/ANMOL_ff3_results.csv

evaluate_xgboost.py       ──► console report
automated_verification.py ──► console report
visualize_anomalies.py    ──► matplotlib chart (price + anomaly overlays)
visualize_erts.py         ──► data/erts/scores/{sym}_visualizer.html
```

---

*Generated by Antigravity on 2026-07-09. All 32 functional source files analyzed.*
