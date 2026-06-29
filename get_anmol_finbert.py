import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
INPUT_FILE = "ANMOL_announcements/announcements.csv"
OUTPUT_FILE = "anmol_daily_nse_sentiment.csv"

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["text"])

print(f"Loaded {len(df)} announcements")

# ==========================
# FINBERT INPUT
# Subject + first part of filing
# ==========================
df["finbert_text"] = (
    df["subject"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str).str[:1000]
)

# ==========================
# LOAD FINBERT
# ==========================
print("Loading FinBERT...")

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

# ==========================
# TEST FIRST RECORD
# ==========================
print("\n" + "="*80)
print("FIRST ANNOUNCEMENT SUBJECT:")
print(df["subject"].iloc[0])

print("\nFINBERT SAMPLE TEST:")

sample_text = df["finbert_text"].iloc[0][:1000]

try:
    sample_result = finbert(sample_text)[0]
    print(sample_result)
except Exception as e:
    print("SAMPLE ERROR:", e)

print("="*80 + "\n")

# ==========================
# SCORE FUNCTION
# ==========================
def get_score(text):

    try:

        text = str(text)[:1000]

        result = finbert(text)[0]

        label = result["label"].lower()
        confidence = float(result["score"])

        if label == "positive":
            return confidence

        elif label == "negative":
            return -confidence

        elif label == "neutral":
            return 0.0

        return 0.0

    except Exception as e:
        print("ERROR:", e)
        return 0.0

# ==========================
# SCORE ANNOUNCEMENTS
# ==========================
print("Running FinBERT...")

tqdm.pandas()

df["sentiment_score"] = df["finbert_text"].progress_apply(get_score)

# ==========================
# CHECK DISTRIBUTION
# ==========================
print("\nScore Distribution:")
print(df["sentiment_score"].describe())

print("\nUnique Scores Sample:")
print(df["sentiment_score"].head(20).tolist())

# ==========================
# DATE CLEANING
# ==========================
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("\nMissing Dates:", df["date"].isna().sum())

df = df.dropna(subset=["date"])

df["date"] = df["date"].dt.date

# ==========================
# DAILY AGGREGATION
# ==========================
daily = (
    df.groupby("date")
      .agg(
          announcement_count=("sentiment_score", "count"),
          avg_sentiment=("sentiment_score", "mean"),
          max_sentiment=("sentiment_score", "max"),
          min_sentiment=("sentiment_score", "min")
      )
      .reset_index()
)

# ==========================
# DAILY LABEL
# ==========================
def classify(score):

    if score >= 0.20:
        return "Positive"

    elif score <= -0.20:
        return "Negative"

    return "Neutral"

daily["label"] = daily["avg_sentiment"].apply(classify)

daily = daily.sort_values("date")

# ==========================
# SAVE
# ==========================
daily.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)

print("\nDaily Sample:")
print(daily.head(20))