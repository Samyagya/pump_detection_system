import feedparser
import pandas as pd
import os

from urllib.parse import quote

# ====================================
# STOCK CONFIG
# ====================================

STOCKS = {
    "ANMOLIND": {
        "query": "Anmol India Limited",
        "aliases": [
            "anmol india",
            "anmol india limited",
            "anmolind"
        ]
    },

    "ARTNIRMAN": {
        "query": "Art Nirman Limited",
        "aliases": [
            "art nirman",
            "art nirman limited",
            "artnirman"
        ]
    },

    "GENUSPAPER": {
        "query": "Genus Paper & Boards Limited",
        "aliases": [
            "genus paper",
            "genus paper and boards",
            "genus paper & boards",
            "genuspaper"
        ]
    },

    "KEEPLEARN": {
        "query": "DSJ Keep Learning Limited",
        "aliases": [
            "keep learning",
            "dsj keep learning",
            "keeplearn"
        ]
    }
}

# ====================================
# CREATE OUTPUT FOLDER
# ====================================

os.makedirs("data/news", exist_ok=True)

# ====================================
# COLLECT NEWS
# ====================================

for ticker, info in STOCKS.items():

    print("\n" + "="*50)
    print("COLLECTING:", ticker)
    print("="*50)

    query = quote(info["query"])

    url = f"https://news.google.com/rss/search?q={query}"

    feed = feedparser.parse(url)

    rows = []

    for entry in feed.entries:

        source = ""

        try:
            source = entry.source.title
        except:
            source = "Unknown"

        rows.append({
            "date": entry.published,
            "source": source,
            "title": entry.title,
            "url": entry.link
        })

    df = pd.DataFrame(rows)

    print("Before Filter:", len(df))

    # ====================================
    # ALIAS FILTER
    # ====================================

    aliases = [
        alias.lower()
        for alias in info["aliases"]
    ]

    if len(df) > 0:

        df = df[
            df["title"]
            .str.lower()
            .apply(
                lambda x:
                any(
                    alias in str(x)
                    for alias in aliases
                )
            )
        ]

    print("After Filter:", len(df))

    # ====================================
    # SAVE
    # ====================================

    output_file = f"data/news/{ticker}_news.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("Saved:", output_file)

print("\nDONE")