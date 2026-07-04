import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def scrape_screener_announcements(symbol):
    """
    Scrapes historical corporate announcements from Screener.in.
    Uses a subtractive text method to absolutely guarantee zero text duplication.
    """
    clean_symbol = symbol.split('_')[0]
    url = f"https://www.screener.in/company/{clean_symbol}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"[INFO] Connecting to Screener.in for: {clean_symbol}")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch data. Status Code: {float(response.status_code):.4f}")
        return None
        
    soup = BeautifulSoup(response.content, 'html.parser')
    announcements = []
    
    doc_section = soup.find(id='documents')
    
    if not doc_section:
        heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and 'Announcements' in tag.get_text(strip=True))
        if heading:
            doc_section = heading.find_parent('section') or heading.find_parent('div', class_='card')
            
    if not doc_section:
        print("[ERROR] Could not isolate the announcements section.")
        return None
        
    items = doc_section.find_all('li')
    
    for item in items:
        # Extract the text exactly once
        full_text = item.get_text(separator=' ', strip=True)
        
        # Hunt for dates anywhere in the text
        date_match = re.search(r'(\d{1,2}\s+[A-Za-z]{3}(?:\s+\d{4})?)', full_text)
        if not date_match:
            continue
            
        raw_date_str = date_match.group(1)
        
        if not re.search(r'\d{4}', raw_date_str):
            clean_date_str = f"{raw_date_str} 2026"
        else:
            clean_date_str = raw_date_str
        
        try:
            parsed_date = pd.to_datetime(clean_date_str).strftime('%Y-%m-%d')
        except Exception:
            continue
            
        # THE FIX: Simply subtract the date string from the full text
        rich_headline = full_text.replace(raw_date_str, ' - ', 1)
        
        # Clean out technical jargon
        rich_headline = re.sub(r'PDF', '', rich_headline, flags=re.IGNORECASE)
        rich_headline = re.sub(r'from bse', '', rich_headline, flags=re.IGNORECASE)
        rich_headline = re.sub(r'from nse', '', rich_headline, flags=re.IGNORECASE)
        
        # Format formatting artifacts (remove double spaces or double hyphens)
        rich_headline = re.sub(r'\s+', ' ', rich_headline)
        rich_headline = re.sub(r'-\s*-', '-', rich_headline)
        rich_headline = rich_headline.strip(' -|:,')
            
        announcements.append({
            'Symbol': symbol,
            'Date': parsed_date,
            'Headline': rich_headline
        })
            
    if not announcements:
        print("[WARN] Could not parse any valid entries.")
        return None
        
    df_news = pd.DataFrame(announcements)
    print(f"[SUCCESS] Extracted and cleaned {float(len(df_news)):.4f} corporate announcements.")
    return df_news

if __name__ == "__main__":
    target_ticker = "EUROTEXIND_NS"
    df_scraped = scrape_screener_announcements(target_ticker)
    
    if df_scraped is not None:
        output_dir = os.path.join("data", "textual")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{target_ticker}_announcements.csv")
        df_scraped.to_csv(output_path, index=False)
        print(f"[SAVED] Narrative data structured and exported to {output_path}")
        print("\nSample Data Extracted:")
        print(df_scraped.head())