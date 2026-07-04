import os
import pandas as pd

# =========================
# LOAD GOOGLE RSS NEWS
# =========================

rss = pd.read_csv(os.path.join("data", "news", "eurotex_news.csv"))

rss = rss.rename(columns={
    "published": "date"
})

rss["content"] = rss["title"]

rss = rss[
    ["date", "source", "title", "content", "url"]
]

# =========================
# LOAD SCREENER NEWS
# =========================

screener = pd.read_csv(
    "data/textual/EUROTEXIND_NS_announcements.csv"
)

screener = screener.rename(columns={
    "Date": "date",
    "Headline": "title"
})

screener["source"] = "Screener"

screener["content"] = screener["title"]

screener["url"] = None

screener = screener[
    ["date", "source", "title", "content", "url"]
]

# =========================
# DATE CLEANING
# =========================

rss["date"] = pd.to_datetime(
    rss["date"],
    errors="coerce"
)

screener["date"] = pd.to_datetime(
    screener["date"],
    errors="coerce"
)

# =========================
# MERGE
# =========================

master = pd.concat(
    [rss, screener],
    ignore_index=True
)

master = master.sort_values(
    "date"
)

master = master.drop_duplicates(
    subset=["date", "title"]
)

# =========================
# SAVE
# =========================

master.to_csv(
    os.path.join("data", "news", "eurotex_master_news.csv"),
    index=False
)

print("\nMaster Dataset Created")
print(master.shape)

print("\nSample Data:")
print(master.head())