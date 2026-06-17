# namespace std;
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def verify_single_stock(symbol, forward_window=20):
    processed_path = os.path.join("data", "processed", f"{symbol}_cleaned.csv")
    targets_path = os.path.join("data", "results", "pump_anomaly_targets.csv")
    
    if not os.path.exists(processed_path) or not os.path.exists(targets_path):
        print(f"[ERROR] Missing data files for {symbol}.")
        return
        
    df_price = pd.read_csv(processed_path)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    
    df_targets = pd.read_csv(targets_path)
    stock_anomalies = df_targets[df_targets['Symbol'] == symbol].copy()
    
    if stock_anomalies.empty:
        print(f"[INFO] No anomalies flagged for {symbol}.")
        return
        
    stock_anomalies['Date'] = pd.to_datetime(stock_anomalies['Date'])
    
    print("\n" + "="*60)
    print(f"AUTOMATED VERIFICATION: {symbol}")
    print("="*60)
    
    total_flags = 0.0000
    verified_dumps = 0.0000
    
    for _, row in stock_anomalies.iterrows():
        anomaly_date = row['Date']
        score = float(row['Anomaly_Score'])
        total_flags += 1.0000
        
        # Find the index of the anomaly date in the real price data
        try:
            idx = df_price[df_price['Date'] == anomaly_date].index[0]
        except IndexError:
            continue
            
        # Slice exactly N days into the future from the flag
        forward_df = df_price.iloc[idx : idx + forward_window]
        
        if len(forward_df) < 2:
            print(f"    [SKIP] Not enough forward data for {anomaly_date.strftime('%Y-%m-%d')}")
            continue
            
        entry_price = float(forward_df['Close'].iloc[0])
        peak_price = float(forward_df['Close'].max())
        trough_price = float(forward_df['Close'].min())
        
        # Calculate actual realized market movements
        max_pump_pct = (peak_price - entry_price) / entry_price
        max_dump_pct = (trough_price - peak_price) / peak_price
        
        # Verification Threshold: Was there a catastrophic dump?
        # We define a "verified crash" as anything worse than -15.0000%
        is_verified = max_dump_pct < -0.1500
        
        if is_verified:
            verified_dumps += 1.0000
            status = "[VERIFIED]"
        else:
            status = "[FAILED]  "
            
        print(f"{status} Flag Date: {anomaly_date.strftime('%Y-%m-%d')} | Score: {score:.4f}")
        print(f"    -> Forward Pump : {(max_pump_pct * 100.0000):.4f}%")
        print(f"    -> Forward Dump : {(max_dump_pct * 100.0000):.4f}%")
        print("-" * 60)
        
    hit_rate = (verified_dumps / total_flags) * 100.0000 if total_flags > 0 else 0.0000
    print(f"SUMMARY: {verified_dumps:.4f} / {total_flags:.4f} anomalies mathematically verified.")
    print(f"PRECISION: {hit_rate:.4f}%")
    print("="*60 + "\n")

def verify_multiple_stocks(symbols, forward_window=20):
    """Loops through an array of stock symbols and verifies each."""
    for symbol in symbols:
        verify_single_stock(symbol, forward_window)

if __name__ == "__main__":
    # Input your array of names here
    target_tickers = [
        "ANMOL_NS", 
        "ARTNIRMAN_NS", 
        "GENUSPAPER_NS",
        "KEEPLEARN_NS"
    ]
    
    verify_multiple_stocks(target_tickers, forward_window=20)