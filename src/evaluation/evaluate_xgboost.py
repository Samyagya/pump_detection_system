import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def evaluate_predictions(preds_dir: Path, threshold: float = -2.0000):
    csv_files = list(preds_dir.glob("*_xgb_preds.csv"))
    if not csv_files:
        log.error(f"No prediction files found in {preds_dir}.")
        return

    log.info(f"Scanning and loading predictions from {float(len(csv_files)):.4f} files...")
    
    df_list = []
    skipped_files = 0.0000

    required_cols = ['Actual_IF_Score', 'Predicted_IF_Score', 'Residual_Delta']

    for file in csv_files:
        try:
            # Load file and immediately clean column names from any hidden spaces
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            # Verify if this specific file has all the required walk-forward outputs
            if not all(col in df.columns for col in required_cols):
                log.warning(f"Skipping outdated or malformed file: {file.name} (Columns found: {list(df.columns)})")
                skipped_files += 1.0000
                continue
                
            df_list.append(df)
        except Exception as e:
            skipped_files += 1.0000
            continue
            
    if not df_list:
        log.error("No valid prediction files with 'Residual_Delta' columns could be loaded.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    log.info(f"Successfully aggregated {float(len(df_list)):.4f} valid stock datasets. (Skipped {skipped_files:.4f} stale files).")
    
    # 1. REGRESSION METRICS (How close were the raw numbers?)
    actuals = master_df['Actual_IF_Score'].astype(float)
    preds = master_df['Predicted_IF_Score'].astype(float)
    
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    # 2. CLASSIFICATION METRICS (Did we catch the anomalies?) Ground Truth: Was the actual day highly anomalous?
    master_df['Actual_Pump'] = (actuals <= threshold).astype(int)
    
    # Model Signal: Did the model's Residual "Surprise" drop below threshold?
    master_df['Model_Signal'] = (master_df['Residual_Delta'].astype(float) <= threshold).astype(int)
    
    y_true = master_df['Actual_Pump']
    y_pred = master_df['Model_Signal']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    

    print("\n" + "="*50)
    print(f"XGBOOST WALK-FORWARD EVALUATION REPORT")
    print(f"Total Trading Rows Evaluated : {float(len(master_df)):.4f}")
    print(f"Anomaly Cutoff Threshold     : {threshold:.4f}")
    print("="*50)
    
    print("\n--- REGRESSION PERFORMANCE ---")
    print(f"Mean Absolute Error (MAE)    : {mae:.4f}")
    print(f"Root Mean Squared (RMSE)     : {rmse:.4f}")
    
    print("\n--- CLASSIFICATION PERFORMANCE ---")
    print(f"Accuracy                     : {accuracy:.4f}")
    print(f"Precision                    : {precision:.4f}")
    print(f"Recall                       : {recall:.4f}")
    print(f"F1-Score                     : {f1:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Negatives (Normal days correctly ignored) : {float(cm[0][0]):.4f}")
    print(f"False Positives (False Alarms)                 : {float(cm[0][1]):.4f}")
    print(f"False Negatives (Missed Pumps)                 : {float(cm[1][0]):.4f}")
    print(f"True Positives (Caught Pumps)                  : {float(cm[1][1]):.4f}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost Predictions")
    parser.add_argument("--preds_dir", type=str, default="data/predictions/xgboost", help="Folder with XGBoost output CSVs")
    parser.add_argument("--threshold", type=float, default=-2.0000, help="The cutoff score to classify a day as an anomaly")
    
    args = parser.parse_args()
    evaluate_predictions(Path(args.preds_dir), args.threshold)

if __name__ == "__main__":
    main()