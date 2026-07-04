from googlesearch import search
import pandas as pd
import re

stock_name = "Eurotex Industries"
year = "2024"

query = f'"{stock_name}" OR EUROTEXIND after:2024-01-01 before:2025-01-01'

trusted_domains = [
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com"
]

rows = []

for url in search(query, num_results=100):

    for domain in trusted_domains:

        if domain in url:

            rows.append({
                "url": url,
                "source": domain
            })

            break

df = pd.DataFrame(rows)

df.drop_duplicates(inplace=True)

print(df.head())

print("\nTotal URLs:", len(df))

df.to_csv("eurotex_urls_2024.csv", index=False)