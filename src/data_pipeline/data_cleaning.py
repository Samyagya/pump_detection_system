import os
import pandas as pd
import numpy as np


# The raw data extracted from yfinance needs to be cleaned. This function cleans it
def clean_financial_data(df):

    # 1. Timezone De-localization
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # 2. Handling Suspensions and Circuit Locks (0 trading days)

    # Forward-fill prices to maintain the last known valuation on non-trading days
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            
    # Zero-fill volume metrics to prevent artificial accumulation illusions
    vol_cols = ['Volume', 'Delivery_Percentage']
    for col in vol_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0000)

    # 3. Redundant Feature Pruning
    cols_to_drop = ['Dividends', 'Stock_Splits']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 4. Precision Locking and Type Unification
    for col in df.columns:
        if col != 'Date':
            # Convert to float first, then lock to exactly 4 decimal places
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    return df


def main():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    
    # Ensure the processed directory exists before saving
    os.makedirs(processed_dir, exist_ok=True)
    
    for filename in os.listdir(raw_dir):
        if filename.endswith("_raw.csv"):
            filepath = os.path.join(raw_dir, filename)
            df = pd.read_csv(filepath)
            
            # Apply the cleaning pipeline
            df = clean_financial_data(df)
            
            # Save the clean data matrix to the processed directory
            safe_out_name = filename.replace("_raw", "_cleaned")
            out_path = os.path.join(processed_dir, safe_out_name)
            df.to_csv(out_path, index=False)
            print(f"[SUCCESS] Cleaned and formatted data saved to: {out_path}")

if __name__ == "__main__":
    main()