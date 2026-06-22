import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# main problem is the "amnesia" of XGBoost - it only sees the current row, not the historical context. We can fix that by creating lagged features that represent the past few days of data. This way, XGBoost can learn from the temporal patterns in the data. So  T-0 (current day) to T-N (N days ago) columns for every feature. This is the "flattening" process that gives XGBoost its memory and allows it to make informed predictions based on historical trends.
def create_flattened_dataset(df: pd.DataFrame, target_col: str, lookback_days: int = 5) -> pd.DataFrame:
    df_flat = df[['Date']].copy()
    
    # Identify the features (everything except Date, Ticker/Symbol, and the Target IF Score)
    exclude_cols = ['Date', target_col, 'Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 1. Create the Target (Tomorrow's IF Score)
    df_flat['Target_Tomorrow'] = df[target_col].shift(-1)
    
    # 2. Build the Lagged Features (The Memory)
    for col in feature_cols:
        df_flat[f'{col}_T0'] = df[col]
        for lag in range(1, lookback_days + 1):
            df_flat[f'{col}_T{lag}'] = df[col].shift(lag)
            
    # Drop the NaN rows created by the shifting
    df_flat = df_flat.dropna().reset_index(drop=True)
    return df_flat


# impllement the daily train-predict-expand loop. Rebuild the XGBoost decision trees from scratch every single day. This is the "walk-forward" approach that simulates how the model would perform in a real-world scenario, where it only has access to past data when making predictions for the next day.
def run_walk_forward_xgboost(df_flat: pd.DataFrame, min_train_days: int = 90) -> pd.DataFrame:
    feature_cols = [c for c in df_flat.columns if c not in ['Date', 'Target_Tomorrow']]
    
    predictions = []
    actuals = []
    dates = []
    
    # Initialize the XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, # shallow trees to prevent overfitting on small training sets
        random_state=42,
        n_jobs=-1
    )
    
    # The Expanding Window Loop
    for t in range(min_train_days, len(df_flat)):
        train_df = df_flat.iloc[:t]
        X_train = train_df[feature_cols]
        y_train = train_df['Target_Tomorrow']
        
        test_df = df_flat.iloc[t:t+1]
        X_test = test_df[feature_cols]
        y_test = test_df['Target_Tomorrow'].values[0] 
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        predictions.append(pred)
        actuals.append(y_test)
        dates.append(test_df['Date'].values[0])
        
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual_IF_Score': actuals,
        'Predicted_IF_Score': predictions
    })
    
    results_df['Residual_Delta'] = results_df['Actual_IF_Score'] - results_df['Predicted_IF_Score']
    
    for col in ['Actual_IF_Score', 'Predicted_IF_Score', 'Residual_Delta']:
        results_df[col] = results_df[col].apply(lambda x: f"{float(x):.4f}")
        
    return results_df

def process_master_file(file_path: Path, output_dir: Path, target_col: str, lookback: int, min_train: int):
    log.info(f"Loading master dataset: {file_path.name}")
    try:
        df_master = pd.read_csv(file_path)
    except Exception as e:
        log.error(f"Cannot read {file_path.name}: {e}")
        return

    if target_col not in df_master.columns:
        log.error(f"Missing target column '{target_col}'. Please verify the exact column name.")
        log.info(f"Columns found: {list(df_master.columns)}")
        return

    # Dynamically find the column containing the stock tickers
    symbol_col = next((col for col in ["Symbol", "symbol", "Ticker", "ticker", "Stock", "stock"] if col in df_master.columns), None)
    
    if not symbol_col:
        log.error(f"Could not find a stock identifier column (Ticker/Symbol) in {file_path.name}.")
        return

    # Fix the jumbled data: Ensure Date is a datetime object so we can sort properly
    if 'Date' in df_master.columns:
        df_master['Date'] = pd.to_datetime(df_master['Date'])

    succeeded = 0.0000
    failed = 0.0000
    
    # Group the jumbled master file by individual stock ticker
    grouped = df_master.groupby(symbol_col)
    log.info(f"Separated data into {float(len(grouped)):.4f} individual stock timelines. Commencing Walk-Forward...")
    
    for ticker, df_stock in grouped:
        # Sort each stock's timeline chronologically
        df_stock = df_stock.sort_values(by='Date').reset_index(drop=True)
        # Convert Date back to string for clean output
        df_stock['Date'] = df_stock['Date'].dt.strftime('%Y-%m-%d')
        
        df_flat = create_flattened_dataset(df_stock, target_col=target_col, lookback_days=lookback)
        
        if len(df_flat) <= min_train:
            log.debug(f"  {ticker}: Not enough rows after flattening ({len(df_flat)}). Need > {min_train}.")
            failed += 1.0000
            continue
            
        results_df = run_walk_forward_xgboost(df_flat, min_train_days=min_train)
        
        out_path = output_dir / f"{ticker}_xgb_preds.csv"
        results_df.to_csv(out_path, index=False)
        succeeded += 1.0000

    log.info("─" * 60)
    log.info(f"Done.  ✓ {succeeded:.4f} models trained   ✗ {failed:.4f} skipped (insufficient data)")

def main():
    parser = argparse.ArgumentParser(description="XGBoost Walk-Forward Predictor")
    
    # Hardcoded the exact path to your specific master file
    parser.add_argument("--input_file", type=str, default="data/results/full_universe_scores.csv", help="Path to the master CSV")
    parser.add_argument("--output_dir", type=str, default="data/predictions/xgboost")
    
    # Hardcoded the exact spelling of your target column
    parser.add_argument("--target_col", type=str, default="Anomaly_Score", help="The exact name of your IF Score column")
    
    parser.add_argument("--lookback", type=int, default=5, help="Number of days to flatten into memory")
    parser.add_argument("--min_train", type=int, default=90, help="Initial training window size")
    
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        log.error(f"Cannot find input file: {input_file}")
        sys.exit(1)

    process_master_file(input_file, output_dir, args.target_col, args.lookback, args.min_train)

if __name__ == "__main__":
    main()