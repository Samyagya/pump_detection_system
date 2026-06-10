import asyncio
import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from datetime import datetime, timezone

# ==========================================
# 1. CREDENTIALS (EDIT THESE)
# ==========================================
API_ID = 12345  # MUST be an integer (no quotes)
# MUST be a string (inside quotes)
API_HASH = '12345'
SESSION_NAME = 'stock_anomaly_session'

# ==========================================
# 2. TARGET CHANNELS (EDIT THESE)
# ==========================================

TARGET_CHANNELS = [
    'hiddenmultibaggerstocks_Devendra',  # Replace with your actual target usernames
    'eqwires',
    'Stockizenofficial',
    'TraderInvestor',
    'CryptoCognizance',
    'mastermindforu',
    'spikeitupequity',
    'stockbabaofficial',
    'SKSTOCKTALKS100',
    'fantasticnifty',
    'officialmarketmaestroo',
    # ... add all 10 of your channels here
]

# ==========================================
# 3. DATE RANGE (SET TO JAN 2021 - DEC 2024)
# ==========================================
# Timezone MUST be UTC because Telegram stores timestamps in UTC
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


async def scrape_all_channels():
    # Initialize the client
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()

    all_messages_data = []

    print(
        f"Starting Historical Scrape: {START_DATE.date()} to {END_DATE.date()}\n")

    for channel in TARGET_CHANNELS:
        print(f"[{channel}] Connecting...")
        message_count = 0

        try:
            # iter_messages without a limit pulls the entire history, newest first
            async for message in client.iter_messages(channel):

                # Skip messages newer than Dec 31, 2024
                if message.date > END_DATE:
                    continue

                # Stop searching this channel if we hit data older than Jan 1, 2021
                if message.date < START_DATE:
                    print(
                        f"[{channel}] Reached 2020 data. Stopping search for this channel.")
                    break

                # If it's a text message inside our date window, save it
                if message.text:
                    all_messages_data.append({
                        'channel_name': channel,
                        'message_id': message.id,
                        'timestamp': message.date,
                        'text_content': message.text,
                        'views': message.views if message.views else 0
                    })
                    message_count += 1

                    # Print progress every 1000 messages so you know it hasn't frozen
                    if message_count % 1000 == 0:
                        print(
                            f"[{channel}] Downloaded {message_count} messages so far...")

            print(f"[{channel}] ✓ Finished. Total saved: {message_count}")
            # Safety pause before hitting the next channel
            await asyncio.sleep(3)

        except FloodWaitError as e:
            print(
                f"\n[WARNING] Telegram API rate limit hit! Sleeping for {e.seconds} seconds...")
            await asyncio.sleep(e.seconds)
            print("Resuming...\n")
        except Exception as e:
            print(f"[{channel}] ❌ Could not scrape. Error: {e}")

    # Convert the collected data into a Pandas DataFrame
    print("\nCompiling data...")
    df = pd.DataFrame(all_messages_data)

    # Save to a CSV file
    output_filename = "master_raw_telegram_2021_2024.csv"
    df.to_csv(output_filename, index=False)

    print(
        f"🎉 Scraping Complete! Saved {len(df)} total messages to {output_filename}")

    await client.disconnect()

if __name__ == '__main__':
    # Run the async loop
    asyncio.run(scrape_all_channels())
