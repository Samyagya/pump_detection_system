import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings

# Import the optimization engine directly from your sibling file
from isolation_forest import load_master_matrix, find_best_parameters

warnings.filterwarnings('ignore')

def main():
    features_dir = os.path.join("data", "features")
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load the unified data matrix
    master_df = load_master_matrix(features_dir)
    
    feature_cols = [
        'Log_Return', 'Vol_Shock_Ratio', 'Norm_Spread', 
        'Amihud_Ratio', 'Delivery_Divergence', 'Volatility_Squeeze', 
        'Positive_Streak', 'Return_Skewness', 'Gap_Up_Momentum', 
        'Volume_Gini_20D', 'OBV_Acceleration'
    ]
    
    # 2. Run Optuna search behind the scenes to capture live optimal hyperparameters
    best_params = find_best_parameters(master_df, feature_cols)
    
    print("\n" + "="*60)
    print("DEPLOYING MODEL WITH LIVE TUNED PARAMETERS")
    print("="*60)
    
    # 3. Initialize model using the dynamic parameter dictionary unpacker
    model = IsolationForest(
        contamination=best_params['contamination'],
        n_estimators=int(best_params['n_estimators']),
        max_samples=best_params['max_samples'],
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Fit the model and extract continuous scoring metrics
    X = master_df[feature_cols].copy()
    print(f"[INFO] Building {float(best_params['n_estimators']):.4f} isolation trees across universe...")
    master_df['Anomaly_Flag'] = model.fit_predict(X)
    master_df['Anomaly_Score'] = model.decision_function(X)
    
    # Sort with highest anomaly profile at the top
    master_df = master_df.sort_values(by='Anomaly_Score', ascending=True)
    
    # Structure presentation view
    front_cols = ['Symbol', 'Date', 'Anomaly_Score', 'Max_Drawdown_20D']
    back_cols = [c for c in master_df.columns if c not in front_cols and c != 'Anomaly_Flag']
    master_df = master_df[front_cols + back_cols]
    
    # Format precision to 4 decimal points
    for col in master_df.columns:
        if col not in ['Date', 'Symbol']:
            master_df[col] = master_df[col].apply(lambda x: f"{float(x):.4f}")
    
    # 5. Export structural datasets
    full_out_path = os.path.join(results_dir, "full_universe_scores.csv")
    master_df.to_csv(full_out_path, index=False)
    
    anomalies_df = master_df[master_df['Anomaly_Score'].astype(float) < 0.0000].copy()
    hitlist_out_path = os.path.join(results_dir, "pump_anomaly_targets.csv")
    anomalies_df.to_csv(hitlist_out_path, index=False)
    
    print(f"[SUCCESS] Exported FULL universe scores ({float(len(master_df)):.4f} rows) to: {full_out_path}")
    print(f"[SUCCESS] Exported TARGETED hit-list ({float(len(anomalies_df)):.4f} rows) to: {hitlist_out_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()