import pandas as pd
from pathlib import Path

def extract_the_seven(preds_dir: Path, threshold: float = -0.3500):
    csv_files = list(preds_dir.glob("*_xgb_preds.csv"))
    
    print(f"Scanning {float(len(csv_files)):.4f} files for severe anomalies...\n")
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            # Extract the stock ticker from the file name
            ticker = file.name.replace("_xgb_preds.csv", "")
            df['Ticker'] = ticker
            
            df_list.append(df)
        except Exception:
            continue
            
    master_df = pd.concat(df_list, ignore_index=True)
    
    # Force numeric types to prevent filtering errors
    master_df['Actual_IF_Score'] = pd.to_numeric(master_df['Actual_IF_Score'], errors='coerce')
    master_df['Residual_Delta'] = pd.to_numeric(master_df['Residual_Delta'], errors='coerce')
    
    # 1. Isolate the 7 actual anomalies
    anomalies_df = master_df[master_df['Actual_IF_Score'] <= threshold].copy()
    
    # 2. Add a column to see if XGBoost caught it (True Positive) or missed it (False Negative)
    anomalies_df['XGBoost_Caught_It'] = anomalies_df['Residual_Delta'] <= threshold
    
    # Sort them from most extreme pump to least extreme
    anomalies_df = anomalies_df.sort_values(by='Actual_IF_Score', ascending=True)
    
    # Enforce strictly 4 decimal places for the console output
    for col in ['Actual_IF_Score', 'Predicted_IF_Score', 'Residual_Delta']:
        anomalies_df[col] = anomalies_df[col].apply(lambda x: f"{float(x):.4f}")
        
    # Reorder columns for a clean visual report
    report_df = anomalies_df[['Date', 'Ticker', 'Actual_IF_Score', 'Residual_Delta', 'XGBoost_Caught_It']]
    
    print("="*60)
    print("THE 7 SEVERE MARKET ANOMALIES (Threshold: -0.3500)")
    print("="*60)
    print(report_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    preds_folder = Path("data/predictions/xgboost")
    extract_the_seven(preds_folder, threshold=-0.3500)