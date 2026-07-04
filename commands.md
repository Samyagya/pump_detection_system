# 💻 Pump Detection System: Command Execution Guide

This document lists all the command-line execution statements required to run the various pipeline scripts in the project.

---

## 🛠️ 1. Environment Activation

Before running any commands, make sure the virtual environment is activated in your terminal.

- **PowerShell (Recommended for Windows)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

---

## 📥 2. Ingestion & Data Pipeline

These commands scrape, download, clean, and merge financial price history and textual news sources.

### A. Price and Volume Ingestion
1. **Select Penny Stock Universe** (fetches EQ list, filters, and ranks):
   ```bash
   python -m src.data_pipeline.penny_stock_selector
   ```
2. **Ingest 3-Year Historical OHLCV data**:
   ```bash
   python -m src.data_pipeline.ingestion
   ```
3. **Clean Financial Price Data**:
   ```bash
   python -m src.data_pipeline.data_cleaning
   ```
4. **Inject Official NSE Delivery Data** (merges real delivery percentages):
   ```bash
   python -m src.data_pipeline.delivery_ingestion
   ```

### B. Text and News Ingestion
5. **Download Announcements & Filings PDFs** (NSE API crawler):
   ```bash
   python -m src.data_pipeline.nse_announcements
   ```
6. **Scrape Corporate Filings from Screener.in** (scrapes Eurotex or target company):
   ```bash
   python -m src.data_pipeline.screener_data
   ```
7. **Scrape Google News RSS Feed**:
   ```bash
   python -m src.data_pipeline.stocks_config
   ```
8. **Scrape Moneycontrol News Headlines**:
   ```bash
   python -m src.data_pipeline.web_scrap
   ```
9. **Merge Google News & Screener Filings**:
    ```bash
    python -m src.data_pipeline.merge_news
    ```
10. **Scrape Historical Telegram Channel logs**:
    ```bash
    python -m src.data_pipeline.telegram_historical_scraper
    ```

---

## 📊 3. Feature Engineering & Sentiment Pipelines

Extract signals, expected volatilities, rule-based event features, and neural sentiment scores.

1. **Calculate Core Quantitative Features** (11 variables + drawdown targets):
   ```bash
   python -m src.features.feature_engineering
   ```
2. **Calculate GARCH(1,1) Expected Volatility and Surprise Metrics**:
   ```bash
   python -m src.features.garch_features --ohlcv_dir data/processed --output_dir data/garch --min_rows 60
   ```
3. **Calculate Rule-Based Event Scores on Corporate Announcements**:
   ```bash
   python -m src.features.event_features
   ```
4. **Calculate FinBERT Sentiment on Corporate Announcements**:
   - **For a Specific Stock** (e.g. `ANMOL`):
     ```bash
     python -m src.features.sentiment_analysis --symbol ANMOL
     ```
   - **For All Available Stocks** (scans directory and batch processes):
     ```bash
     python -m src.features.sentiment_analysis --symbol ALL
     ```
5. **Calculate FinBERT Sentiment on News Files**:
   - **For Eurotex news**:
     ```bash
     python -m src.features.finbert_load
     ```
   - **For all news files** (batch runner):
     ```bash
     python -m src.features.batch_finbert
     ```

---

## 🤖 4. Model Training & Anomaly Predictors

Train models to identify pricing anomalies and forecast future structural crashes.

1. **Isolation Forest Tuning (Optuna) and Inference**:
   ```bash
   python -m src.models.inference
   ```
2. **Fama-French 3-Factor Model & Abnormal Return Calculator**:
   ```bash
   python -m src.models.fama_french
   ```
3. **XGBoost Walk-Forward Regressor** (forecasts tomorrow's anomaly score):
   ```bash
   python -m src.models.xgboost_predictor --input_file data/results/full_universe_scores.csv --output_dir data/predictions/xgboost --target_col Anomaly_Score --lookback 5 --min_train 90
   ```

---

## 📈 5. Diagnostics, Validation & Visualizers

Assess prediction performance and visualize model predictions against actual market movements.

1. **Evaluate XGBoost Predictor** (Regression & Classification Metrics):
   ```bash
   python -m src.evaluation.evaluate_xgboost --preds_dir data/predictions/xgboost --threshold -2.0
   ```
2. **Inspect Severe XGBoost/IF Anomalies** (target score $\le -0.35$):
   ```bash
   python -m src.evaluation.inspect_xgboost_anomalies
   ```
3. **Verify Anomalies Precision** (checks peak-to-trough drops in subsequent 20D windows):
   ```bash
   python -m src.evaluation.automated_verification
   ```
4. **Plot Anomaly Price & Volume Overlay charts** (Green = verified, Red = unverified):
   ```bash
   python -m src.evaluation.visualize_anomalies
   ```
5. **Plot Announcement Event Scores Over Time**:
   ```bash
   python -m src.evaluation.visualize_event_scores
   ```
6. **Plot News Sentiment vs. News Volume**:
   ```bash
   python -m src.evaluation.visualize_news_sentiment
   ```
