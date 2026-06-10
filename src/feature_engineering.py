import os
import pandas as pd
import numpy as np
import warnings

# Suppress pandas fragmentation and rolling calculation warnings
warnings.filterwarnings('ignore')

def calculate_gini(v):
    """
    Calculates the Gini Coefficient for a rolling window of volumes.
    A score near 1.0000 indicates extreme volume concentration (manipulation).
    """
    v = np.sort(v)
    n = len(v)
    if n == 0 or np.sum(v) == 0: 
        return 0.0000
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * v)) / (n * np.sum(v))

def calculate_optimized_features(df):
    """
    Computes the 11 strictly orthogonal features designed to compress 
    time-series mechanics into static spatial coordinates for the Isolation Forest.
    """
    # 1. The Foundation: Daily Logarithmic Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. The Activity Signal: Volume Shock Ratio (20-Day)
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Shock_Ratio'] = df['Volume'] / (df['Vol_MA20'] + 1e-9)
    
    # 3. The Intraday Battle: Normalized High-Low Spread
    df['Norm_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # 4. The Operator Footprint: Amihud Illiquidity Ratio
    df['Rupee_Volume'] = df['Close'] * df['Volume']
    df['Amihud_Ratio'] = df['Log_Return'].abs() / (df['Rupee_Volume'] + 1e-9)
    
    # 5. The Wash Trade Trap: Delivery-Volume Divergence
    df['Delivery_Divergence'] = df['Vol_Shock_Ratio'] * (1.0000 - df['Delivery_Percentage'])
    
    # 6. The Temporal Springboard: Volatility Squeeze (10D vs 90D)
    df['Vol_10D'] = df['Log_Return'].rolling(window=10).std()
    df['Vol_90D'] = df['Log_Return'].rolling(window=90).std()
    df['Volatility_Squeeze'] = df['Vol_10D'] / (df['Vol_90D'] + 1e-9)
    
    # 7. The Unnatural Climb: Consecutive Positive Streak
    is_positive = (df['Log_Return'] > 0.0000).astype(int)
    # Blocks increment every time we hit a day that is NOT positive
    blocks = (~(df['Log_Return'] > 0.0000)).cumsum()
    df['Positive_Streak'] = is_positive.groupby(blocks).cumsum()
    
    # 8. The Asymmetric Tail: Rolling Return Skewness (20-Day)
    df['Return_Skewness'] = df['Log_Return'].rolling(window=20).skew()
    
    # 9. The Overnight Trap: Gap-Up Momentum
    df['Gap_Up_Momentum'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # 10. The Coordinated Blast: Volume Gini Coefficient (20-Day)
    df['Volume_Gini_20D'] = df['Volume'].rolling(window=20).apply(calculate_gini, raw=True)
    
    # 11. The Hidden Accumulation: OBV Acceleration
    # Calculate daily OBV flow
    obv_flow = np.sign(df['Log_Return']) * df['Volume']
    df['OBV'] = obv_flow.cumsum()
    # Calculate acceleration (Z-score of OBV)
    df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
    df['OBV_Std20'] = df['OBV'].rolling(window=20).std()
    df['OBV_Acceleration'] = (df['OBV'] - df['OBV_MA20']) / (df['OBV_Std20'] + 1e-9)

    # ---------------------------------------------------------
    # VALIDATION TARGET (For Optuna Only)
    # ---------------------------------------------------------
    df['Forward_Min_20D'] = df['Close'].shift(-20).rolling(window=20).min()
    df['Max_Drawdown_20D'] = df['Forward_Min_20D'] / df['Close'] - 1.0000

    # ---------------------------------------------------------
    # CLEANUP & FORMATTING
    # ---------------------------------------------------------
    # Define the exact columns to keep
    features_to_keep = [
        'Date', 'Close', 'Log_Return', 'Vol_Shock_Ratio', 'Norm_Spread', 
        'Amihud_Ratio', 'Delivery_Divergence', 'Volatility_Squeeze', 
        'Positive_Streak', 'Return_Skewness', 'Gap_Up_Momentum', 
        'Volume_Gini_20D', 'OBV_Acceleration', 'Max_Drawdown_20D'
    ]
    df = df[features_to_keep].copy()
    
    # Drop rows with NaNs (The first 90 days will drop due to Volatility Squeeze)
    df = df.dropna().reset_index(drop=True)
    
    # Enforce strict 4-decimal formatting for all numerical outputs
    for col in df.columns:
        if col not in ['Date', 'Positive_Streak']:
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}")
        elif col == 'Positive_Streak':
            # Keep streak as a clean float with 4 decimal places for matrix consistency
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}")
            
    return df

def main():
    cleaned_dir = os.path.join("data", "processed")
    features_dir = os.path.join("data", "features")
    
    os.makedirs(features_dir, exist_ok=True)
    
    total_files = len([f for f in os.listdir(cleaned_dir) if f.endswith("_cleaned.csv")])
    processed_count = 0.0000
    
    print(f"\n[INFO] Starting core feature generation (11 variables) for {float(total_files):.4f} stocks...")
    
    for filename in os.listdir(cleaned_dir):
        if filename.endswith("_cleaned.csv"):
            filepath = os.path.join(cleaned_dir, filename)
            df = pd.read_csv(filepath)
            
            # Ensure proper typing before calculations
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Delivery_Percentage']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Execute feature engineering
            df_features = calculate_optimized_features(df)
            
            # Output to features directory
            safe_out_name = filename.replace("_cleaned", "_features")
            out_path = os.path.join(features_dir, safe_out_name)
            df_features.to_csv(out_path, index=False)
            
            processed_count += 1.0000
            
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Successfully Generated Feature Matrices : {processed_count:.4f} / {float(total_files):.4f}")
    print(f"Output Target Destination Directory   : {features_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()