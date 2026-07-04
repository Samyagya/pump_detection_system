import requests
import pandas as pd
import fitz  # PyMuPDF
import os
import io
import time

# =========================
# CONFIG
# =========================
SYMBOL = "ANMOL"
OUTPUT_DIR = os.path.join("data", "textual", f"{SYMBOL}_announcements")

os.makedirs(os.path.join(OUTPUT_DIR, "pdfs"), exist_ok=True)

# =========================
# NSE SESSION
# =========================
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

# Get cookies
session.get(
    "https://www.nseindia.com",
    headers=headers,
    timeout=20
)

# =========================
# FETCH ANNOUNCEMENTS
# =========================
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
print(announcements[0])

print(f"Found {len(announcements)} announcements")

rows = []

# =========================
# PROCESS EACH ANNOUNCEMENT
# =========================
for i, item in enumerate(announcements):

    date = item.get("sort_date")
    subject = item.get("subject")

    pdf_link = (
        item.get("attchmntFile")
        or item.get("attachment")
        or item.get("fileLink")
    )

    pdf_text = ""

    try:
        if pdf_link:

            if not pdf_link.startswith("http"):
                pdf_link = "https://www.nseindia.com" + pdf_link

            pdf_response = session.get(
                pdf_link,
                headers=headers,
                timeout=60
            )

            pdf_name = f"{OUTPUT_DIR}/pdfs/{i}.pdf"

            with open(pdf_name, "wb") as f:
                f.write(pdf_response.content)

            doc = fitz.open(
                stream=pdf_response.content,
                filetype="pdf"
            )

            pdf_text = ""

            for page in doc:
                pdf_text += page.get_text()

            doc.close()

            time.sleep(0.5)

    except Exception as e:
        print("PDF Error:", e)

    rows.append({
        "symbol": SYMBOL,
        "date": date,
        "subject": subject,
        "pdf_link": pdf_link,
        "text": pdf_text
    })

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(rows)

csv_file = os.path.join(OUTPUT_DIR, "announcements.csv")

df.to_csv(
    csv_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved {len(df)} announcements")
print(f"CSV: {csv_file}")