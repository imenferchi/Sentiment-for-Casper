#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# STEP 1: Locate and load environment variables from .env file
# This allows secure and flexible configuration of database credentials.
# -----------------------------------------------------------------------------
root = Path(__file__).parent  # project root directory
env_path = root / "phase1_data_extraction" / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"No .env found at {env_path}")
load_dotenv(env_path)

MONGODB_URI   = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fear_index_db")
if not MONGODB_URI:
    raise RuntimeError("Please set MONGODB_URI in your .env")

# -----------------------------------------------------------------------------
# STEP 2: Connect to MongoDB and select relevant collections
# NEWS_COLL: stores all news articles
# DAILY_MATCH_COLL: stores daily matches between sentiment and S&P 500 returns
# -----------------------------------------------------------------------------
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db     = client[MONGO_DB_NAME]

NEWS_COLL        = db["financial_news"]
DAILY_MATCH_COLL = db["daily_sentiment_return_match"]

# -----------------------------------------------------------------------------
# STEP 3: Compute average number of articles per day
# Uses MongoDB aggregation to group articles by date and calculate the average.
# -----------------------------------------------------------------------------
pipeline = [
    {"$project": {"date": {"$substr": ["$publishedAt", 0, 10]}}},
    {"$group": {"_id": "$date", "count": {"$sum": 1}}},
    {"$group": {"_id": None, "avgCount": {"$avg": "$count"}, "days": {"$sum": 1}}}
]
res = list(NEWS_COLL.aggregate(pipeline))
if res:
    avg_articles = res[0]["avgCount"]
    total_days   = int(res[0]["days"])
else:
    print("[ERROR] Could not compute average articles per day.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# STEP 4: Compute total number of news articles in the database
# -----------------------------------------------------------------------------
total_articles = NEWS_COLL.count_documents({})

# -----------------------------------------------------------------------------
# STEP 5: Compute the match rate between sentiment and S&P 500 returns
# The match rate is the percentage of days where the sentiment and S&P 500 return
# had the same sign (both positive or both negative).
# -----------------------------------------------------------------------------
total_matches = DAILY_MATCH_COLL.count_documents({})
if total_matches > 0:
    matched_count = DAILY_MATCH_COLL.count_documents({"match": 1})
    match_pct     = (matched_count / total_matches) * 100
else:
    matched_count = 0
    match_pct     = 0.0

# -----------------------------------------------------------------------------
# STEP 6: Print summary statistics to the console
# This provides a quick overview of the data coverage and model alignment.
# -----------------------------------------------------------------------------
print("=== Model Statistics ===")
print(f"Average articles per day  : {round(avg_articles)} over {total_days} days")
print(f"Total number of articles  : {total_articles}")
print(f"Match rate (S&P500 vs sentiment): {match_pct:.1f}%")

