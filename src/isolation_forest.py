import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import optuna
import warnings

# Suppress warnings for clean terminal output during trial iterations
warnings.filterwarnings('ignore')

def load_master_matrix(features_dir):
    """
    Loads all 181 individual stock feature matrices and concatenates them 
    into a single global training dataset for the Isolation Forest.
    """
    print(f"[INFO] Assembling global feature matrix from: {features_dir}")
    all_data = []
    
    for filename in os.listdir(features_dir):
        if filename.endswith("_features.csv"):
            filepath = os.path.join(features_dir, filename)
            df = pd.read_csv(filepath)
            
            # Ensure proper float typing
            for col in df.columns:
                if col != 'Date':
                    df[col] = df[col].astype(float)
                    
            df['Symbol'] = filename.split("_features")[0]
            all_data.append(df)
            
    master_df = pd.concat(all_data, ignore_index=True)
    # Drop any lingering NaNs to prevent sklearn matrix failures
    master_df = master_df.dropna().reset_index(drop=True)
    
    print(f"[SUCCESS] Global matrix assembled. Total active trading days: {float(len(master_df)):.4f}")
    return master_df

def objective(trial, master_df, feature_cols):
    """
    The Optuna objective function. Trains the Isolation Forest with trial 
    hyperparameters and evaluates success against the 20-day drawdown validation loop.
    """
    # 1. Define the hyperparameter search space
    contamination = trial.suggest_float("contamination", 0.0100, 0.1200, step=0.0050)
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=10)
    max_samples = trial.suggest_float("max_samples", 0.2000, 0.9000, step=0.0500)
    
    # 2. Initialize the spatial model
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Fit the model and extract anomaly flags (-1 = Anomaly, 1 = Normal)
    X = master_df[feature_cols].copy()
    master_df['Prediction'] = model.fit_predict(X)
    
    # 4. The 20-Day Drawdown Validation Loop
    # Target criteria: The stock must crash by at least 25.0000% within the next 20.0000 days
    actual_crashes = master_df['Max_Drawdown_20D'] <= -0.2500
    model_flags = master_df['Prediction'] == -1
    
    # 5. Compile Performance Metrics
    true_positives = float((model_flags & actual_crashes).sum())
    false_positives = float((model_flags & ~actual_crashes).sum())
    false_negatives = float((~model_flags & actual_crashes).sum())
    
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    
    # Calculate the F1-Score (The harmonious balance of Precision and Recall)
    f1_score = (2.0000 * precision * recall) / (precision + recall + 1e-9)
    
    return f1_score

def main():
    features_dir = os.path.join("data", "features")
    if not os.path.exists(features_dir):
        print(f"[ERROR] Directory not found: {features_dir}")
        return
        
    # 1. Load the dataset and define the exact 11.0000 orthogonal features
    master_df = load_master_matrix(features_dir)
    
    feature_cols = [
        'Log_Return', 'Vol_Shock_Ratio', 'Norm_Spread', 
        'Amihud_Ratio', 'Delivery_Divergence', 'Volatility_Squeeze', 
        'Positive_Streak', 'Return_Skewness', 'Gap_Up_Momentum', 
        'Volume_Gini_20D', 'OBV_Acceleration'
    ]
    
    # 2. Initialize and execute the Optuna Hyperparameter Study
    print("\n" + "="*60)
    print("INITIALIZING OPTUNA HYPERPARAMETER SEARCH")
    print("="*60)
    
    # We want to maximize the F1-Score
    study = optuna.create_study(direction="maximize")
    # Execute 25.0000 distinct trial combinations
    study.optimize(lambda trial: objective(trial, master_df, feature_cols), n_trials=25)
    
    # 3. Output the mathematically optimal configuration
    best_trial = study.best_trial
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE: OPTIMAL PIPELINE PARAMETERS")
    print("="*60)
    print(f"Maximized F1-Score   : {float(best_trial.value):.4f}")
    print(f"Contamination Rate   : {float(best_trial.params['contamination']):.4f}")
    print(f"Number of Estimators : {float(best_trial.params['n_estimators']):.4f}")
    print(f"Max Samples Fraction : {float(best_trial.params['max_samples']):.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()