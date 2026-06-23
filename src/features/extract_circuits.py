"""
src/features/extract_circuits.py
===================================
Extracts institutional-grade Circuit Lock metrics from 
daily OHLC data using strict Exchange Band Clustering.
Saves output to a newly isolated 'circuit_forensics' directory.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_circuit_streaks(series):
    """Calculates consecutive days of hitting upper circuits."""
    streak = []
    current_streak = 0
    for val in series:
        if val == 1:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)
    return streak


def extract_all_circuits():
    # 1. Map input directory
    input_dir = os.path.join("data", "processed")

    # 2. CREATE THE NEW ISOLATED FOLDER
    output_dir = os.path.join("data", "circuit_forensics")
    # This safely creates the new folder
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "circuit_features_all.csv")

    # 3. Find all your cleaned 4-year daily datasets
    csv_files = glob.glob(os.path.join(input_dir, "*_cleaned.csv"))
    if not csv_files:
        logging.error(f"No cleaned files found in {input_dir}.")
        return

    logging.info(
        f"Processing {len(csv_files)} stocks for Circuit Forensics...")

    all_circuit_records = []

    # 4. Process each stock
    for file in csv_files:
        ticker = os.path.basename(file).replace("_cleaned.csv", "")
        df = pd.read_csv(file)

        # Ensure we have enough data
        if len(df) < 20:
            continue

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # --- THE ROBUST CIRCUIT MATH ---

        # A. Calculate Daily Return
        df['Return'] = df['Close'].pct_change()

        # B. Strict Exchange Band Clustering
        # We only look for returns that perfectly match the Indian exchange limits
        is_near_band = (
            df['Return'].between(0.019, 0.021) |  # 2% Circuit Limit
            df['Return'].between(0.048, 0.052) |  # 5% Circuit Limit
            df['Return'].between(0.098, 0.102) |  # 10% Circuit Limit
            df['Return'].between(0.198, 0.202)    # 20% Circuit Limit
        )

        # C. The Ultimate Operator Lock Flag (Close must equal High AND hit a strict band)
        df['Is_Circuit'] = ((df['Close'] == df['High'])
                            & is_near_band).astype(int)

        # D. Feature Engineering: 20-Day Rolling Frequency & Streaks
        df['Circuit_Freq_20d'] = df['Is_Circuit'].rolling(
            window=20, min_periods=1).sum()
        df['Circuit_Streak'] = calculate_circuit_streaks(df['Is_Circuit'])

        # E. Format for output
        df['Ticker'] = ticker

        # Only keep the columns we need for the final machine learning matrix
        final_cols = ['Date', 'Ticker', 'Close', 'Return',
                      'Is_Circuit', 'Circuit_Freq_20d', 'Circuit_Streak']
        all_circuit_records.append(df[final_cols].dropna())

    # 5. Save the compiled results
    if all_circuit_records:
        master_df = pd.concat(all_circuit_records, ignore_index=True)
        master_df.to_csv(output_file, index=False)

        total_circuits = master_df['Is_Circuit'].sum()

        print("\n" + "="*60)
        logging.info("CIRCUIT EXTRACTION COMPLETE")
        logging.info(
            f"Total Circuit Locks Detected Across All Stocks: {total_circuits}")
        logging.info(
            f"Created new directory and saved firmly to: {output_file}")
        print("="*60 + "\n")
    else:
        logging.warning("No data processed.")


if __name__ == "__main__":
    extract_all_circuits()
