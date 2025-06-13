#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
from pymongo import MongoClient, UpdateOne
import certifi
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables from .env file to get MongoDB credentials.
# This allows secure and flexible configuration of database access.
# -----------------------------------------------------------------------------
root = Path(__file__).parent.parent
env_path = root / "phase1_data_extraction" / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"No .env found at {env_path}")
load_dotenv(env_path)

MONGODB_URI   = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fear_index_db")
if not MONGODB_URI:
    raise RuntimeError("Please set MONGODB_URI in your .env")

# -----------------------------------------------------------------------------
# Connect to MongoDB and select the collection for storing S&P 500 daily returns.
# This collection will store one document per trading day with open, close, and return.
# -----------------------------------------------------------------------------
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db     = client[MONGO_DB_NAME]
coll   = db["sp500_daily_returns"]

def find_last_trading_date(target: date) -> date:
    """
    Find the most recent trading day on or before the given date.
    This function steps backwards from the target date until it finds a day
    for which yfinance returns valid S&P 500 data (i.e., a trading day).
    """
    day = target
    while True:
        # yfinance treats `end` as exclusive, so we add one day to the end date.
        df = yf.download("^GSPC",
                         start=day.strftime("%Y-%m-%d"),
                         end=(day + timedelta(days=1)).strftime("%Y-%m-%d"),
                         progress=False,
                         auto_adjust=False)
        if not df.empty:
            return day
        day -= timedelta(days=1)

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Main execution: Download and store the most recent S&P 500 daily return.
    # -----------------------------------------------------------------------------

    # Determine the most recent trading day (usually "yesterday").
    today_local     = date.today()
    candidate       = today_local - timedelta(days=1)
    trading_day     = find_last_trading_date(candidate)
    trading_day_str = trading_day.strftime("%Y-%m-%d")

    print(f"Fetching S&P 500 for trading day: {trading_day_str}")

    # Download S&P 500 data for the identified trading day using yfinance.
    df = yf.download("^GSPC",
                     start=trading_day_str,
                     end=(trading_day + timedelta(days=1)).strftime("%Y-%m-%d"),
                     progress=False,
                     auto_adjust=False)
    if df.empty:
        print("No data for that trading day—something’s wrong. Exiting.")
        sys.exit(1)

    # If the DataFrame has a MultiIndex (sometimes happens), flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Extract the row for the trading day.
    row = df.iloc[0]
    if pd.isna(row["Open"]) or pd.isna(row["Close"]):
        print(f"Missing Open/Close for {trading_day_str}. Exiting.")
        sys.exit(1)

    # Calculate the intraday return: (Close - Open) / Open.
    intraday_ret = (row["Close"] - row["Open"]) / row["Open"]
    doc = {
        "Date":   trading_day_str,
        "Open":   float(row["Open"]),
        "Close":  float(row["Close"]),
        "Return": intraday_ret
    }

    # Insert the result into MongoDB if it doesn't already exist for this date.
    # Uses upsert to avoid duplicates: only inserts if the date is not present.
    try:
        result = coll.update_one(
            {"Date": trading_day_str},
            {"$setOnInsert": doc},
            upsert=True
        )
        if result.upserted_id:
            print(f"Inserted new document for {trading_day_str}.")
        else:
            print(f"No insert needed; {trading_day_str} already in DB.")
    except Exception as e:
        print("Mongo upsert failed:", e)
        sys.exit(1)
