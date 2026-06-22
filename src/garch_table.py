import os
import pandas as pd


def generate_garch_summary():
    # Because this script is in 'src', we go up one level to reach 'data'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(base_dir, 'data', 'garch')

    parameter_results = []

    if not os.path.exists(data_folder):
        print(f"Error: Could not find folder '{data_folder}'.")
        return

    print(f"Scanning files in {data_folder}...")

    # Loop through all the CSVs in the garch folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            try:
                df = pd.read_csv(file_path)

                # Check if your specific GARCH columns exist
                required_cols = ['GARCH_Omega', 'GARCH_Alpha', 'GARCH_Beta']
                if all(col in df.columns for col in required_cols):

                    # Drop rows from the warmup period where parameters are empty
                    valid_data = df[required_cols].dropna()

                    if not valid_data.empty:
                        # Since you used a rolling window, a single stock has multiple rows of parameters.
                        # We take the mean of these rows to get one representing value per stock.
                        stock_omega = valid_data['GARCH_Omega'].mean()
                        stock_alpha = valid_data['GARCH_Alpha'].mean()
                        stock_beta = valid_data['GARCH_Beta'].mean()
                        stock_alpha_beta = stock_alpha + stock_beta

                        parameter_results.append({
                            'Stock': filename,
                            'omega': stock_omega,
                            'alpha': stock_alpha,
                            'beta': stock_beta,
                            'alpha_plus_beta': stock_alpha_beta
                        })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert the collected data into a DataFrame
    if parameter_results:
        results_df = pd.DataFrame(parameter_results)

        # Calculate cross-sectional summary statistics across all 167 stocks
        summary = results_df[['omega', 'alpha', 'beta', 'alpha_plus_beta']].describe(
            percentiles=[0.25, 0.50, 0.75])

        # Format the table to match your LaTeX report
        final_table = summary.loc[['mean', '50%', '25%', '75%']].T
        final_table.columns = ['Mean', 'Median',
                               '25th Percentile', '75th Percentile']

        print("\n" + "="*80)
        print(" CROSS-SECTIONAL GARCH PARAMETER SUMMARY (Update LaTeX Table 4.1 with this)")
        print("="*80)

        # Format floating points nicely (scientific notation for omega)
        formatters = {
            'Mean': '{:.6f}'.format,
            'Median': '{:.6f}'.format,
            '25th Percentile': '{:.6f}'.format,
            '75th Percentile': '{:.6f}'.format
        }
        print(final_table.to_string(formatters=formatters))
        print("="*80)
        print(f"\nSuccessfully processed {len(results_df)} stocks.")
    else:
        print("No valid GARCH parameter data found in the CSV files.")


if __name__ == "__main__":
    generate_garch_summary()
