import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Plot the Close Price and Volume of the stock. Red vertical lines indicate where the IF flagged anomaly
def plot_anomalies(symbol):
    processed_path = os.path.join("data", "processed", f"{symbol}_cleaned.csv")
    targets_path = os.path.join("data", "results", "pump_anomaly_targets.csv")
    
    if not os.path.exists(processed_path):
        print(f"[ERROR] Could not find historical data for {symbol}.")
        return
        
    if not os.path.exists(targets_path):
        print("[ERROR] Could not find pump_anomaly_targets.csv. Run inference.py first.")
        return

    # Load Price Data
    df_price = pd.read_csv(processed_path)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    
    # Load Anomaly Targets
    df_targets = pd.read_csv(targets_path)
    stock_anomalies = df_targets[df_targets['Symbol'] == symbol].copy()
    
    if stock_anomalies.empty:
        print(f"[INFO] No anomalies were flagged for {symbol} in the target list.")
        return
        
    stock_anomalies['Date'] = pd.to_datetime(stock_anomalies['Date'])
    
    print("\n" + "="*60)
    print(f"VISUALIZING TARGET: {symbol}")
    print("="*60)

    # Initialize the plot layout (Top: Price, Bottom: Volume)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Isolation Forest Verification: {symbol}", fontsize=16, fontweight='bold')

    # TOP CHART: PRICE
    ax1.plot(df_price['Date'], df_price['Close'], color='black', linewidth=1.5000, label='Close Price')
    ax1.set_ylabel("Price (INR)")
    ax1.grid(True, linestyle='--', alpha=0.5000)
    
    # BOTTOM CHART: VOLUME
    ax2.bar(df_price['Date'], df_price['Volume'], color='gray', alpha=0.5000, label='Daily Volume')
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle='--', alpha=0.5000)

    # Overlay the Anomalies
    for _, row in stock_anomalies.iterrows():
        anomaly_date = row['Date']
        score = float(row['Anomaly_Score'])
        
        # Red vertical dashed line across both charts
        ax1.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=2.0000, alpha=0.8000)
        ax2.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=2.0000, alpha=0.8000)
        
        # Add a prominent red marker on the exact price point
        try:
            exact_price = df_price.loc[df_price['Date'] == anomaly_date, 'Close'].values[0]
            ax1.plot(anomaly_date, exact_price, marker='v', color='red', markersize=10.0000)
        except IndexError:
            pass 
            
        print(f"    [ALERT] Date: {anomaly_date.strftime('%Y-%m-%d')} | Severity Score: {score:.4f}")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_ticker = "EUROTEXIND_NS" 
    plot_anomalies(target_ticker)