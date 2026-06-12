import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from jugaad_data.nse import stock_df
import warnings

warnings.filterwarnings('ignore')

def fetch_nse_delivery(symbol_with_ns, min_date, max_date):
    """
    Fetches official historical delivery data directly from NSE archives.
    Uses updated column headers to match the current NSE Bhavcopy format.
    """
    clean_symbol = symbol_with_ns.replace("_NS", "")
    
    try:
        # Fetch raw NSE data
        raw_data = stock_df(
            symbol=clean_symbol, 
            from_date=min_date, 
            to_date=max_date, 
            series="EQ"
        )
        
        df_nse = pd.DataFrame(raw_data)
        
        if df_nse.empty:
            print(f"    [WARNING] NSE returned empty data for {clean_symbol}. Skipping.")
            return pd.DataFrame()
            
        # UPDATE: Looking for 'DELIVERY %' instead of '%Deliverble'
        if 'DATE' not in df_nse.columns or 'DELIVERY %' not in df_nse.columns:
            print(f"    [WARNING] Missing expected columns for {clean_symbol}. Found: {list(df_nse.columns)}")
            return pd.DataFrame()
        
        # Format the date to match your existing Yahoo Finance pipeline
        df_nse['Date'] = pd.to_datetime(df_nse['DATE']).dt.strftime('%Y-%m-%d')
        
        # Extract and format the delivery percentage using the new header
        df_nse['Delivery_Percentage'] = pd.to_numeric(df_nse['DELIVERY %'], errors='coerce') / 100.0000
        
        return df_nse[['Date', 'Delivery_Percentage']]
        
    except Exception as e:
        print(f"    [ERROR] Connection or fetch failure for {clean_symbol}: {str(e)}")
        time.sleep(5.0000) 
        return pd.DataFrame()

def replace_mocked_delivery():
    """
    Loops through the existing OHLCV dataset, downloads the real delivery data, 
    and perfectly merges it with the Yahoo Finance price data.
    """
    processed_dir = os.path.join("data", "processed")
    total_files = len([f for f in os.listdir(processed_dir) if f.endswith("_cleaned.csv")])
    processed_count = 0.0000
    
    print(f"\n[INFO] Initiating real NSE Delivery injection for {float(total_files):.4f} stocks...")
    
    for filename in os.listdir(processed_dir):
        if filename.endswith("_cleaned.csv"):
            filepath = os.path.join(processed_dir, filename)
            df_yfinance = pd.read_csv(filepath)
            
            # Identify the date boundaries required for the NSE API
            df_yfinance['Date'] = pd.to_datetime(df_yfinance['Date'])
            min_date = df_yfinance['Date'].min().date()
            max_date = df_yfinance['Date'].max().date()
            
            symbol = filename.replace("_cleaned.csv", "")
            print(f"Processing {symbol}...")
            
            # Fetch the real delivery data
            df_delivery = fetch_nse_delivery(symbol, min_date, max_date)
            
            if not df_delivery.empty:
                # Convert Yahoo Finance dates back to string for a clean merge
                df_yfinance['Date'] = df_yfinance['Date'].dt.strftime('%Y-%m-%d')
                
                # Drop the mocked 'Delivery_Percentage' column you generated earlier
                if 'Delivery_Percentage' in df_yfinance.columns:
                    df_yfinance = df_yfinance.drop(columns=['Delivery_Percentage'])
                
                # Merge the real NSE delivery data onto the Yahoo Finance OHLCV data
                df_merged = pd.merge(df_yfinance, df_delivery, on='Date', how='left')
                
                # Forward-fill any NaN delivery values caused by API glitches or missing NSE data,
                # then enforce strictly 4 decimal places
                df_merged['Delivery_Percentage'] = df_merged['Delivery_Percentage'].ffill()
                df_merged['Delivery_Percentage'] = df_merged['Delivery_Percentage'].apply(lambda x: f"{float(x):.4f}")
                
                # Overwrite the file with the real data
                df_merged.to_csv(filepath, index=False)
                processed_count += 1.0000
            
            # Sleep briefly to avoid overwhelming the NSE servers and getting IP banned
            time.sleep(3)

    print("\n" + "="*60)
    print("DELIVERY DATA INJECTION COMPLETE")
    print("="*60)
    print(f"Successfully Updated Matrices : {processed_count:.4f} / {float(total_files):.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    replace_mocked_delivery()