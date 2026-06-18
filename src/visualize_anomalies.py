# namespace std;
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def plot_anomalies(symbol, forward_window=20):
    """
    Plots the Close Price and Volume for a given stock.
    Overlays vertical lines and markers:
    - GREEN: Verified anomalies (forward max drawdown < -15.0000%)
    - RED: Unverified anomalies
    """
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
    print(f"VISUALIZING AND VERIFYING TARGET: {symbol}")
    print("="*60)

    # Initialize the plot layout (Top: Price, Bottom: Volume)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Isolation Forest Verification & Backtest: {symbol}", fontsize=16, fontweight='bold')

    # --- TOP CHART: PRICE ---
    ax1.plot(df_price['Date'], df_price['Close'], color='black', linewidth=1.5000, label='Close Price')
    ax1.set_ylabel("Price (INR)")
    ax1.grid(True, linestyle='--', alpha=0.5000)
    
    # --- BOTTOM CHART: VOLUME ---
    ax2.bar(df_price['Date'], df_price['Volume'], color='gray', alpha=0.5000, label='Daily Volume')
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle='--', alpha=0.5000)

    # Tracking labels to avoid duplicating them in the chart legend
    added_verified_legend = False
    added_unverified_legend = False

    # Overlay the Anomalies with Look-Ahead Validation
    for _, row in stock_anomalies.iterrows():
        anomaly_date = row['Date']
        score = float(row['Anomaly_Score'])
        
        # Determine index to calculate forward window
        try:
            idx = df_price[df_price['Date'] == anomaly_date].index[0]
        except IndexError:
            continue
            
        # Extract the look-ahead window
        forward_df = df_price.iloc[idx : idx + forward_window]
        
        if len(forward_df) < 2:
            continue
            
        peak_price = float(forward_df['Close'].max())
        trough_price = float(forward_df['Close'].min())
        max_dump_pct = (trough_price - peak_price) / peak_price
        
        # Mathematical verification threshold (-15.0000%)
        is_verified = max_dump_pct < -0.1500
        
        # Dynamic color assigning
        if is_verified:
            marker_color = 'green'
            label_text = 'Verified Anomaly (Crash > 15%)' if not added_verified_legend else ""
            added_verified_legend = True
            status = "[VERIFIED]"
        else:
            marker_color = 'red'
            label_text = 'Unverified Anomaly' if not added_unverified_legend else ""
            added_unverified_legend = True
            status = "[UNVERIFIED]"

        # Plot vertical dashed lines across both charts
        ax1.axvline(x=anomaly_date, color=marker_color, linestyle='--', linewidth=1.8000, alpha=0.7000, label=label_text)
        ax2.axvline(x=anomaly_date, color=marker_color, linestyle='--', linewidth=1.8000, alpha=0.7000)
        
        # Add prominent marker on the exact price point
        try:
            exact_price = df_price.loc[df_price['Date'] == anomaly_date, 'Close'].values[0]
            ax1.plot(anomaly_date, exact_price, marker='v', color=marker_color, markersize=10.0000)
        except IndexError:
            pass 
            
        print(f"    {status} Date: {anomaly_date.strftime('%Y-%m-%d')} | Score: {score:.4f} | Max Drawdown: {(max_dump_pct * 100.0000):.4f}%")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_multiple_anomalies(symbols, forward_window=20):
    """Loops through an array of stock symbols and plots each."""
    for symbol in symbols:
        plot_anomalies(symbol, forward_window)

if __name__ == "__main__":
    # Input your array of names here
    target_tickers = [
        "EUROTEXIND_NS"
    ]
    
    plot_multiple_anomalies(target_tickers, forward_window=20)