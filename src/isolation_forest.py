import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import optuna
import warnings

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
            
            for col in df.columns:
                if col != 'Date':
                    df[col] = df[col].astype(float)
                    
            df['Symbol'] = filename.split("_features")[0]
            all_data.append(df)
            
    master_df = pd.concat(all_data, ignore_index=True)
    master_df = master_df.dropna().reset_index(drop=True)
    print(f"[SUCCESS] Global matrix assembled. Total active trading days: {float(len(master_df)):.4f}")
    return master_df

def objective(trial, master_df, feature_cols):
    """The Optuna objective function evaluating success against the 20-day drawdown."""
    contamination = trial.suggest_float("contamination", 0.0100, 0.1200, step=0.0050)
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=10)
    max_samples = trial.suggest_float("max_samples", 0.2000, 0.9000, step=0.0500)
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    X = master_df[feature_cols].copy()
    master_df['Prediction'] = model.fit_predict(X)
    
    actual_crashes = master_df['Max_Drawdown_20D'] <= -0.2500
    model_flags = master_df['Prediction'] == -1
    
    true_positives = float((model_flags & actual_crashes).sum())
    false_positives = float((model_flags & ~actual_crashes).sum())
    false_negatives = float((~model_flags & actual_crashes).sum())
    
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = (2.0000 * precision * recall) / (precision + recall + 1e-9)
    
    return f1_score

def find_best_parameters(master_df, feature_cols):
    """
    Executes the Optuna study and returns the optimal hyperparameter dictionary
    directly to whichever script requests it.
    """
    print("\n" + "="*60)
    print("INITIALIZING DYNAMIC OPTUNA HYPERPARAMETER SEARCH")
    print("="*60)
    
    # Turning off Optuna's verbose multi-line logging for a clean runtime display
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, master_df, feature_cols), n_trials=25)
    
    best_trial = study.best_trial
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE: PARAMETERS FOUND")
    print("="*60)
    print(f"Maximized F1-Score   : {float(best_trial.value):.4f}")
    print(f"Contamination Rate   : {float(best_trial.params['contamination']):.4f}")
    print(f"Number of Estimators : {float(best_trial.params['n_estimators']):.4f}")
    print(f"Max Samples Fraction : {float(best_trial.params['max_samples']):.4f}")
    print("="*60 + "\n")
    
    return best_trial.params

if __name__ == "__main__":
    features_dir = os.path.join("data", "features")
    if os.path.exists(features_dir):
        df = load_master_matrix(features_dir)
        features = [
            'Log_Return', 'Vol_Shock_Ratio', 'Norm_Spread', 'Amihud_Ratio', 
            'Delivery_Divergence', 'Volatility_Squeeze', 'Positive_Streak', 
            'Return_Skewness', 'Gap_Up_Momentum', 'Volume_Gini_20D', 'OBV_Acceleration'
        ]
        find_best_parameters(df, features)