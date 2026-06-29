from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import glob
import os

# ====================================
# LOAD FINBERT
# ====================================

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ====================================
# FINBERT FUNCTION
# ====================================

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

# ====================================
# CREATE OUTPUT FOLDER
# ====================================

os.makedirs(
    "data/sentiment",
    exist_ok=True
)

# ====================================
# PROCESS ALL STOCK FILES
# ====================================

files = glob.glob(
    "data/news/*_news.csv"
)

for file in files:

    ticker = (
        os.path.basename(file)
        .replace("_news.csv", "")
    )

    print("\n" + "="*60)
    print("PROCESSING:", ticker)
    print("="*60)

    df = pd.read_csv(file)

    article_scores = []
    positive_probs = []
    negative_probs = []
    neutral_probs = []

    for _, row in df.iterrows():

        text = str(row["title"])

        result = finbert_sentiment(text)

        score = (
            result["positive"]
            - result["negative"]
        )

        article_scores.append(score)

        positive_probs.append(
            result["positive"]
        )

        negative_probs.append(
            result["negative"]
        )

        neutral_probs.append(
            result["neutral"]
        )

    df["positive"] = positive_probs
    df["negative"] = negative_probs
    df["neutral"] = neutral_probs
    df["sentiment_score"] = article_scores

    # =====================
    # SAVE ARTICLE LEVEL
    # =====================

    article_file = (
        f"data/sentiment/"
        f"{ticker}_article_sentiment.csv"
    )

    df.to_csv(
        article_file,
        index=False
    )

    # =====================
    # DAILY AGGREGATION
    # =====================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    df["date"] = df["date"].dt.date

    daily = (
        df.groupby("date")
        .agg(
            article_count=("title", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_articles=(
                "sentiment_score",
                lambda x: (x > 0.2).sum()
            ),
            negative_articles=(
                "sentiment_score",
                lambda x: (x < -0.2).sum()
            )
        )
        .reset_index()
    )

    daily_file = (
        f"data/sentiment/"
        f"{ticker}_daily_sentiment.csv"
    )

    daily.to_csv(
        daily_file,
        index=False
    )

    print(
        f"Articles: {len(df)}"
    )

    print(
        f"Saved: {article_file}"
    )

    print(
        f"Saved: {daily_file}"
    )

print("\nDONE")