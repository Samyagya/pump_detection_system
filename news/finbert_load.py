from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# =========================
# LOAD MASTER NEWS DATA
# =========================

df = pd.read_csv("eurotex_master_news.csv")

print("Rows Loaded:", len(df))
print("\nColumns:")
print(df.columns)

# =========================
# LOAD FINBERT
# =========================

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# =========================
# FINBERT FUNCTION
# =========================

def finbert_sentiment(text):

    text = str(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )[0]

    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2])
    }

# =========================
# ARTICLE LEVEL SENTIMENT
# =========================

article_scores = []
positive_probs = []
negative_probs = []
neutral_probs = []

for _, row in df.iterrows():

    text = (
        str(row["title"]) +
        " " +
        str(row["content"])
    )

    result = finbert_sentiment(text)

    score = (
        result["positive"]
        - result["negative"]
    )

    article_scores.append(score)

    positive_probs.append(result["positive"])
    negative_probs.append(result["negative"])
    neutral_probs.append(result["neutral"])

# =========================
# ADD RESULTS TO DATAFRAME
# =========================

df["positive"] = positive_probs
df["negative"] = negative_probs
df["neutral"] = neutral_probs
df["sentiment_score"] = article_scores

# =========================
# SAVE ARTICLE LEVEL FILE
# =========================

df.to_csv(
    "eurotex_article_sentiment.csv",
    index=False
)

print("\nSaved: eurotex_article_sentiment.csv")

# =========================
# DATE CLEANING
# =========================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(subset=["date"])

df["date"] = df["date"].dt.date

# =========================
# DAILY AGGREGATION
# =========================

daily = (
    df.groupby("date")
      .agg(
          article_count=("title", "count"),
          avg_sentiment=("sentiment_score", "mean"),
          positive_articles=(
              "sentiment_score",
              lambda x: (x > 0.20).sum()
          ),
          negative_articles=(
              "sentiment_score",
              lambda x: (x < -0.20).sum()
          )
      )
      .reset_index()
)

# =========================
# SAVE DAILY FILE
# =========================

daily.to_csv(
    "eurotex_daily_sentiment.csv",
    index=False
)

print("\nSaved: eurotex_daily_sentiment.csv")

print("\nDaily Sentiment Summary:")
print(daily.head())

print("\nOverall Average Sentiment:")
print(round(daily["avg_sentiment"].mean(), 4))