import feedparser
from urllib.parse import quote
import pandas as pd

query = quote("Anmol india ltd")

url = f"https://news.google.com/rss/search?q={query}"

feed = feedparser.parse(url)

rows = []

rows = []

for entry in feed.entries:

    rows.append({
        "title": entry.title,
        "published": entry.published,
        "source": entry.source.title,
        "url": entry.link
    })

df = pd.DataFrame(rows)

print(df.head())

df.to_csv(
    "anmol_news.csv",
    index=False
)