from eventregistry import *
import pandas as pd

er = EventRegistry(apiKey="9b3aa7c7-8903-4a1c-a491-52dd9be63080")

query = QueryArticlesIter(
    keywords=QueryItems.OR(["Vodafone Idea", "IDEA", "Vi"]),
    lang="eng",
    dateStart="2021-01-01",
    dateEnd="2024-12-31"
)

articles = []

for article in query.execQuery(
        er,
        sortBy="date",
        maxItems=1000):

    articles.append({
        "date": article.get("date"),
        "title": article.get("title"),
        "source": article.get("source", {}).get("title"),
        "url": article.get("url"),
        "body": article.get("body")
    })

df = pd.DataFrame(articles)

print(df.head())
print("Total Articles:", len(df))

df.to_csv("vodafone_idea_news_2021_2024.csv", index=False)