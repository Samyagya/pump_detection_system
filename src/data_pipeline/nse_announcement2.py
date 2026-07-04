import requests
import pandas as pd
import os

# ==========================
# CONFIG
# ==========================
SYMBOL = "ANMOL"

OUTPUT_DIR = os.path.join("data", "textual", f"{SYMBOL}_announcements")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# NSE SESSION
# ==========================
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

# Get NSE cookies
session.get(
    "https://www.nseindia.com",
    headers=headers,
    timeout=20
)

# ==========================
# FETCH ANNOUNCEMENTS
# ==========================
url = (
    f"https://www.nseindia.com/api/corporate-announcements"
    f"?index=equities&symbol={SYMBOL}"
)

response = session.get(
    url,
    headers=headers,
    timeout=30
)

response.raise_for_status()

announcements = response.json()

print(f"Found {len(announcements)} announcements")

# ==========================
# EXTRACT DATA
# ==========================
rows = []

for item in announcements:

    rows.append({

        "symbol": item.get("symbol"),

        # best date field
        "date": item.get("sort_date"),

        # NSE category / headline
        "announcement_type": item.get("desc"),

        # short NSE description
        "announcement_summary": item.get("attchmntText"),

        # raw date if needed
        "an_dt": item.get("an_dt"),

        # PDF link
        "pdf_link": item.get("attchmntFile")

    })

# ==========================
# SAVE CSV
# ==========================
df = pd.DataFrame(rows)

output_file = os.path.join(OUTPUT_DIR, "nse_announcements_clean.csv")

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved: {output_file}")
print("\nSample:")
print(df.head())