import os
import argparse
import glob
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def process_sentiment_for_stock(symbol, input_dir, output_dir, finbert):
    clean_symbol = symbol.upper().replace("_NS", "").replace(".NS", "").strip()
    print(f"\n" + "="*60)
    print(f"PROCESSING SENTIMENT FOR STOCK: {clean_symbol}")
    print("="*60)
    
    input_file = os.path.join(input_dir, f"{clean_symbol}_announcements", "announcements.csv")
    output_file = os.path.join(output_dir, f"{clean_symbol.lower()}_daily_nse_sentiment.csv")
    
    if not os.path.exists(input_file):
        print(f"[ERROR] Announcements file not found at {input_file}. Please fetch announcements first.")
        return False
        
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to read {input_file}: {e}")
        return False
        
    if "text" not in df.columns or "subject" not in df.columns:
        print(f"[ERROR] Required columns ('text', 'subject') missing from {input_file}.")
        return False
        
    df = df.dropna(subset=["text"])
    if len(df) == 0:
        print(f"[WARNING] No announcements with text found for {clean_symbol}. Skipping.")
        return False
        
    print(f"Loaded {len(df)} announcements for {clean_symbol}")
    
    df["finbert_text"] = (
        df["subject"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str).str[:1000]
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
            print(f"ERROR: {e}")
            return 0.0
            
    print(f"Running FinBERT scoring for {clean_symbol}...")
    tqdm.pandas(desc=f"Scoring {clean_symbol}")
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    
    print("\nDaily Sample:")
    print(daily.head(20))
    return True

def main():
    parser = argparse.ArgumentParser(description="FinBERT Sentiment Analysis on Corporate Announcements")
    parser.add_argument("--symbol", type=str, default="ANMOL", help="Stock symbol to process, or 'ALL' to process all available stocks")
    parser.add_argument("--input_dir", type=str, default=os.path.join("data", "textual"), help="Directory containing announcements")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "sentiment"), help="Directory to save sentiment CSV files")
    args = parser.parse_args()
    
    print("Loading FinBERT...")
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )
    
    symbol_arg = args.symbol.strip()
    
    if symbol_arg.upper() == "ALL":
        print(f"Scanning for announcements in: {args.input_dir}")
        pattern = os.path.join(args.input_dir, "*_announcements")
        announcement_dirs = glob.glob(pattern)
        
        symbols_to_process = []
        for d in announcement_dirs:
            if os.path.isdir(d):
                dirname = os.path.basename(d)
                stock_symbol = dirname.replace("_announcements", "")
                symbols_to_process.append(stock_symbol)
                
        if not symbols_to_process:
            print(f"[ERROR] No announcements directories found in {args.input_dir}.")
            return
            
        print(f"Found {len(symbols_to_process)} stocks to process: {symbols_to_process}")
        
        success_count = 0
        for s in symbols_to_process:
            success = process_sentiment_for_stock(s, args.input_dir, args.output_dir, finbert)
            if success:
                success_count += 1
                
        print(f"\n[SUMMARY] Successfully processed {success_count}/{len(symbols_to_process)} stocks.")
    else:
        success = process_sentiment_for_stock(symbol_arg, args.input_dir, args.output_dir, finbert)
        if not success:
            exit(1)

if __name__ == "__main__":
    main()