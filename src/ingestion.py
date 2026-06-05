import os
import sys
import datetime
import time
import pandas as pd
import numpy as np
import yfinance as yf

def get_ticker_universe(csv_path="stratified_training_universe.csv"):
    """
    Dynamically loads the stratified training universe CSV generated in Stage 2.
    Extracts tickers from the 'symbol' column and formats them with the required '.NS' suffix.
    """
    print(f"[INFO] Loading target training universe from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Critical universe file missing: '{csv_path}' not found in the current directory.")
        print("[ERROR] Please verify that 'Filtering_penny_stock.py' ran successfully.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(csv_path)
        if "symbol" not in df.columns:
            print("[ERROR] Corrupted CSV format: 'symbol' column missing from universe data.")
            sys.exit(1)
            
        # Clean whitespaces and extract distinct tickers
        symbols = df["symbol"].dropna().astype(str).str.strip().unique().tolist()
        
        # Append .NS suffix for Yahoo Finance NSE compatibility
        tickers = [f"{sym}.NS" if not sym.endswith(".NS") else sym for sym in symbols]
        print(f"[SUCCESS] Parsed {float(len(tickers)):.4f} unique tickers from universe matrix.")
        return tickers
    except Exception as e:
        print(f"[ERROR] Failed to read universe CSV asset: {str(e)}")
        sys.exit(1)

def fetch_historical_ohlcv(ticker, start_date, end_date, max_retries=3):
    """
    Fetches daily OHLCV historical time-series data from Yahoo Finance for a given ticker.
    Implements transient network failure retries and rate limit handling.
    """
    print(f"[INFO] Fetching 3-year historical data for {ticker}...")
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty or len(df) < 5:
                print(f"[WARNING] Empty data matrix or insufficient history returned for: {ticker}")
                return None
            return df
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(k in err_msg for k in ["429", "rate", "throttl", "too many"])
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = float((2 ** attempt) * 4)
                print(f"    [Rate Limit] {ticker} throttling detected. Pausing for {wait_seconds:.4f} seconds before retry...")
                time.sleep(wait_seconds)
            else:
                print(f"[ERROR] Complete pipeline failure fetching {ticker} on attempt {float(attempt + 1):.4f}: {str(e)}")
                break
    return None

def inject_delivery_data(df):
    """
    Simulates daily delivery volume percentages matching standard 
    Indian market structural behavior to support downstream features.
    Returns values strictly adhering to the 4-decimal place specification.
    """
    np.random.seed(42)
    low_bound = 0.1000
    high_bound = 0.7000
    delivery_pct = np.random.uniform(low_bound, high_bound, size=len(df))
    df["Delivery_Percentage"] = np.round(delivery_pct, 4)
    return df

def main():
    # Define exact time horizons (3 years of structural historical data)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=3 * 365)
    
    # Dynamically extract tickers from the stratified training universe file
    tickers = get_ticker_universe("stratified_training_universe.csv")
    
    # Configure output target paths
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    successful_fetches = 0.0000
    total_tickers = float(len(tickers))
    
    print(f"\n[START] Initializing massive data download pipeline for {total_tickers:.4f} assets...")
    
    for idx, ticker in enumerate(tickers):
        df = fetch_historical_ohlcv(ticker, start_date, end_date)
        
        if df is not None:
            # Reformat index and normalize column naming schema
            df = df.reset_index()
            df.columns = [col.replace(" ", "_") for col in df.columns]
            
            # Inject synthetic delivery metrics for microstructure feature mapping
            df = inject_delivery_data(df)
            
            # Enforce systematic type definitions across numerical data columns
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Standardize output filenames
            safe_filename = ticker.replace(".", "_") + "_raw.csv"
            output_path = os.path.join(raw_data_dir, safe_filename)
            df.to_csv(output_path, index=False)
            
            successful_fetches += 1.0000
            print(f"[SUCCESS] [{float(idx + 1):.4f}/{total_tickers:.4f}] Saved data array to {output_path}")
        else:
            print(f"[FAILED] Skipping asset node: {ticker}")
            
        # Throttling safety guard to preserve API stability across 181 continuous calls
        time.sleep(1.0000)
            
    success_rate = (successful_fetches / total_tickers) * 100.0000
    print("\n" + "="*60)
    print("DATA INGESTION PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Successfully Processed Tickers : {successful_fetches:.4f} / {total_tickers:.4f}")
    print(f"Pipeline Download Success Rate : {success_rate:.4f}%")
    print(f"Output Target Destination Directory  : {raw_data_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()