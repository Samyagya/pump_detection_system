import pandas as pd

# 1. Load the full 400-stock universe you saved from Phase 1
df = pd.read_csv("./penny_stock_universe.csv")

# Ensure it's sorted by risk score just in case
df = df.sort_values(by='pump_risk_score',
                    ascending=False).reset_index(drop=True)

# 2. Build the Stratified Universe (80 / 50 / 50)
top_80 = df.iloc[:80]  # The highly suspicious haystacks (Rows 0 to 79)

# The baseline: Randomly sample 50 from the middle section (Rows 80 to 250)
middle_50 = df.iloc[80:250].sample(n=50, random_state=42)

# The boring normal: Randomly sample 50 from the bottom section (Rows 250 onwards)
bottom_50 = df.iloc[250:].sample(n=50, random_state=42)

# 3. Combine them into your final 180-stock training universe
final_universe = pd.concat(
    [top_80, middle_50, bottom_50]).reset_index(drop=True)

# Save this so you never have to do it again
final_universe.to_csv("stratified_training_universe.csv", index=False)

print(f"Final Universe Size: {len(final_universe)}")
print("Average Risk Score by Bucket:")
print(f"Top 80: {top_80['pump_risk_score'].mean():.2f}")
print(f"Mid 50: {middle_50['pump_risk_score'].mean():.2f}")
print(f"Bot 50: {bottom_50['pump_risk_score'].mean():.2f}")
