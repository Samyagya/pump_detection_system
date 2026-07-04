from newspaper import build, Article
import pandas as pd

paper = build(
    "https://www.moneycontrol.com/news/business/"
)

rows = []

for article_obj in paper.articles[:30]:

    try:
        url = article_obj.url

        if "mmb.moneycontrol.com" in url:
            continue

        if "/news/business/" not in url:
            continue

        if "/news/business/stocks/" not in url:
            continue

        article = Article(url)

        article.download()
        article.parse()

        rows.append({
            "source": "moneycontrol",
            "url": url,
            "title": article.title,
            "body": article.text,
            "date": article.publish_date
        })

        print("Collected:", article.title)

    except Exception as e:

        print("Failed:", url)
        print(e)

df = pd.DataFrame(rows)

print(df.shape)

df.to_csv(
    os.path.join("data", "news", "moneycontrol_news.csv"),
    index=False
)