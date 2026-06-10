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
    
    # 3. Calculate Anomaly Severity Score
    # decision_function returns lower/negative values for highly anomalous points
    master_df['Anomaly_Score'] = model.decision_function(X)
    
    # 4. Filter down to ONLY the days flagged as manipulated (-1.0000)
    anomalies_df = master_df[master_df['Anomaly_Flag'] == -1.0000].copy()
    
    # Sort so the absolute most extreme, dangerous anomalies are at the very top
    anomalies_df = anomalies_df.sort_values(by='Anomaly_Score', ascending=True)
    
    # Reorder columns for readability
    front_cols = ['Symbol', 'Date', 'Anomaly_Score', 'Max_Drawdown_20D']
    back_cols = [c for c in anomalies_df.columns if c not in front_cols and c != 'Anomaly_Flag']
    anomalies_df = anomalies_df[front_cols + back_cols]
    
    # 5. Format outputs to exactly 4 decimal places
    for col in anomalies_df.columns:
        if col not in ['Date', 'Symbol']:
            anomalies_df[col] = anomalies_df[col].apply(lambda x: f"{float(x):.4f}")
    
    # 6. Save the Hit-List
    out_path = os.path.join(results_dir, "pump_anomaly_targets.csv")
    anomalies_df.to_csv(out_path, index=False)
    
    print(f"[SUCCESS] Isolated {float(len(anomalies_df)):.4f} total anomalous events.")
    print(f"[SUCCESS] Exported ranked target list to: {out_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()