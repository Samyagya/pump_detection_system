import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eurotex_daily_sentiment.csv")

df["date"] = pd.to_datetime(df["date"])
df = df[
    df["date"] >= "2026-05-01"
]
fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(
    df["date"],
    df["avg_sentiment"],
    marker="o",
    label="Sentiment"
)

ax1.set_ylabel("Average Sentiment")

ax2 = ax1.twinx()

ax2.bar(
    df["date"],
    df["article_count"],
    alpha=0.3,
    label="Articles"
)

ax2.set_ylabel("Article Count")

plt.title("Eurotex Daily Sentiment vs News Volume")

plt.tight_layout()

plt.savefig(
    "eurotex_sentiment_volume.png",
    dpi=300
)

plt.show()

