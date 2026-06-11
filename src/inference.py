import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

def load_master_matrix(features_dir):
    """Loads the global feature matrix."""
    all_data = []
    for filename in os.listdir(features_dir):
        if filename.endswith("_features.csv"):
            filepath = os.path.join(features_dir, filename)
            df = pd.read_csv(filepath)
            for col in df.columns:
                if col != 'Date':
                    df[col] = df[col].astype(float)
            df['Symbol'] = filename.split("_features")[0]
            all_data.append(df)
            
    master_df = pd.concat(all_data, ignore_index=True)
    return master_df.dropna().reset_index(drop=True)

def main():
    features_dir = os.path.join("data", "features")
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    master_df = load_master_matrix(features_dir)
    
    feature_cols = [
        'Log_Return', 'Vol_Shock_Ratio', 'Norm_Spread', 
        'Amihud_Ratio', 'Delivery_Divergence', 'Volatility_Squeeze', 
        'Positive_Streak', 'Return_Skewness', 'Gap_Up_Momentum', 
        'Volume_Gini_20D', 'OBV_Acceleration'
    ]
    
    print("\n" + "="*60)
    print("DEPLOYING OPTIMIZED ISOLATION FOREST")
    print("="*60)
    
    # 1. Initialize with Optuna's exact findings
    model = IsolationForest(
        contamination=0.1150,
        n_estimators=220,
        max_samples=0.2500,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Fit the model and predict anomalies
    X = master_df[feature_cols].copy()
    print("[INFO] Building 220.0000 isolation trees and scoring universe...")
    master_df['Anomaly_Flag'] = model.fit_predict(X)
    
    # 3. Calculate Anomaly Severity Score for ALL rows
    master_df['Anomaly_Score'] = model.decision_function(X)
    
    # Sort everything so the most dangerous scores (negative) are at the top, 
    # and normal scores (positive) are at the bottom.
    master_df = master_df.sort_values(by='Anomaly_Score', ascending=True)
    
    # Reorder columns for readability
    front_cols = ['Symbol', 'Date', 'Anomaly_Score', 'Max_Drawdown_20D']
    back_cols = [c for c in master_df.columns if c not in front_cols and c != 'Anomaly_Flag']
    master_df = master_df[front_cols + back_cols]
    
    # 4. Format all outputs to exactly 4.0000 decimal places
    for col in master_df.columns:
        if col not in ['Date', 'Symbol']:
            master_df[col] = master_df[col].apply(lambda x: f"{float(x):.4f}")
    
    # 5. Create the two distinct CSV files
    # File 1: The full universe (What you just requested)
    full_out_path = os.path.join(results_dir, "full_universe_scores.csv")
    master_df.to_csv(full_out_path, index=False)
    
    # File 2: The filtered targeted hit-list (Score < 0.0000)
    # Since we formatted the columns to strings, we need to convert Anomaly_Score back to float to filter
    anomalies_df = master_df[master_df['Anomaly_Score'].astype(float) < 0.0000].copy()
    hitlist_out_path = os.path.join(results_dir, "pump_anomaly_targets.csv")
    anomalies_df.to_csv(hitlist_out_path, index=False)
    
    print(f"[SUCCESS] Exported FULL universe scores ({float(len(master_df)):.4f} rows) to: {full_out_path}")
    print(f"[SUCCESS] Exported TARGETED hit-list ({float(len(anomalies_df)):.4f} rows) to: {hitlist_out_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()