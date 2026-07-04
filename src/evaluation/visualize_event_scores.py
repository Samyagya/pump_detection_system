import os
import pandas as pd
import matplotlib.pyplot as plt

# Load event features
df = pd.read_csv(
    os.path.join("data", "textual", "ANMOL_announcements", "daily_event_features.csv")
)

df["date"] = pd.to_datetime(df["date"])

# Sort
df = df.sort_values("date")

# Plot
plt.figure(figsize=(14,6))

plt.plot(
    df["date"],
    df["total_event_score"],
    marker="o"
)

plt.title("ANMOL - Daily NSE Event Score")
plt.xlabel("Date")
plt.ylabel("Event Score")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join("data", "results", "event_score_plot.png"),
    dpi=300
)

plt.show()

print("Saved: data/results/event_score_plot.png")