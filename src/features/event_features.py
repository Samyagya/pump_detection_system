import os
import pandas as pd

# Load NSE data
df = pd.read_csv(
    os.path.join("data", "textual", "ANMOL_announcements", "nse_announcements_clean.csv")
)

# ==========================
# EVENT MAPPING
# ==========================
EVENT_MAP = {

    "Price movement": {
        "category": "Price Alert",
        "score": 3
    },

    "Outcome of Board Meeting": {
        "category": "Corporate Action",
        "score": 2
    },

    "Change in Management": {
        "category": "Management Change",
        "score": -2
    },

    "Resignation of Independent director": {
        "category": "Management Change",
        "score": -3
    },

    "Demise": {
        "category": "Management Change",
        "score": -4
    },

    "Disclosure under SEBI Takeover Regulations": {
        "category": "Ownership Change",
        "score": 2
    },

    "General Updates": {
        "category": "General",
        "score": 0
    },

    "Trading Window": {
        "category": "Compliance",
        "score": 0
    },

    "Structural Digital Database": {
        "category": "Compliance",
        "score": 0
    },

    "Certificate under SEBI (Depositories and Participants) Regulations": {
        "category": "Compliance",
        "score": 0
    },

    "Shareholders meeting": {
        "category": "Corporate Governance",
        "score": 1
    }
}

# ==========================
# FEATURE GENERATION
# ==========================
def get_category(event):

    event = str(event)

    for key in EVENT_MAP:
        if key.lower() in event.lower():
            return EVENT_MAP[key]["category"]

    return "Other"

def get_score(event):

    event = str(event)

    for key in EVENT_MAP:
        if key.lower() in event.lower():
            return EVENT_MAP[key]["score"]

    return 0

df["event_category"] = (
    df["announcement_type"]
      .apply(get_category)
)

df["event_score"] = (
    df["announcement_type"]
      .apply(get_score)
)

# Date
df["date"] = pd.to_datetime(df["date"]).dt.date

# Daily aggregation
daily = (
    df.groupby("date")
      .agg(
          announcement_count=("event_score","count"),
          total_event_score=("event_score","sum"),
          avg_event_score=("event_score","mean")
      )
      .reset_index()
)

daily.to_csv(
    os.path.join("data", "textual", "ANMOL_announcements", "daily_event_features.csv"),
    index=False
)

print(daily.head())