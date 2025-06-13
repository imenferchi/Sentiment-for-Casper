#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# STEP 1: Load environment variables from .env file for MongoDB connection.
# This allows secure and flexible configuration of database credentials.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
env_path     = PROJECT_ROOT / "phase1_data_extraction" / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"No .env found at {env_path}")
load_dotenv(env_path)

MONGODB_URI   = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fear_index_db")
if not MONGODB_URI:
    raise RuntimeError("Please set MONGODB_URI in your .env")

# -----------------------------------------------------------------------------
# STEP 2: Connect to MongoDB and define all relevant collections.
# These collections store news, sentiment, daily averages, S&P 500 returns, and match results.
# -----------------------------------------------------------------------------
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db     = client[MONGO_DB_NAME]

NEWS_COLL          = db["financial_news"]
SENTIMENT_COLL     = db["financial_news_sentiment"]
DAILY_SENT_COLL    = db["daily_sentiment_average"]
SP500_RETURNS_COLL = db["sp500_daily_returns"]
DAILY_MATCH_COLL   = db["daily_sentiment_return_match"]

# -----------------------------------------------------------------------------
# STEP 3: Set up logging for status and error messages.
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# STEP 4: Fetch all sentiment-labeled articles from MongoDB.
# Each document contains an article's _id and its sentiment analysis results.
# -----------------------------------------------------------------------------
logger.info("Fetching all sentiment-labeled articles...")
all_sent_docs = list(SENTIMENT_COLL.find({}))
if not all_sent_docs:
    logger.warning("No documents found in 'financial_news_sentiment'. Exiting.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# STEP 5: Group adjusted sentiment scores by article publish date.
# For each sentiment document, join with the original news article to get the published date.
# This enables aggregation of sentiment scores by day.
# -----------------------------------------------------------------------------
daily_scores = defaultdict(list)
for sent_doc in all_sent_docs:
    art_id    = sent_doc.get("_id")
    adj_score = sent_doc.get("adjusted_score")
    if adj_score is None:
        continue

    orig = NEWS_COLL.find_one({"_id": art_id}, {"publishedAt": 1})
    if not orig or "publishedAt" not in orig:
        continue

    iso = orig["publishedAt"]  # e.g. "2025-06-04T14:22:00Z"
    date_part = iso.split("T")[0]
    daily_scores[date_part].append(adj_score)

if not daily_scores:
    logger.warning("After joining, no (date → adjusted_score) pairs found. Exiting.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# STEP 6: Compute the average adjusted sentiment score for each date.
# This gives a daily sentiment index summarizing all news for that day.
# -----------------------------------------------------------------------------
logger.info("Computing daily average of adjusted_score for each date...")
daily_avg = {date: float(np.mean(scores)) for date, scores in daily_scores.items()}

# -----------------------------------------------------------------------------
# STEP 7: Calculate quantile thresholds for sentiment buckets.
# The 33rd and 66th percentiles are used to classify each day as
# "negative_overall", "neutral_overall", or "positive_overall".
# -----------------------------------------------------------------------------
all_vals = np.array(list(daily_avg.values()), dtype=float)
p33      = float(np.percentile(all_vals, 33))
p66      = float(np.percentile(all_vals, 66))
logger.info(f"QUANTILE thresholds: 33rd_pct = {p33:.4f}, 66th_pct = {p66:.4f}")

def classify_quantile(avg: float, low: float, high: float) -> str:
    """
    Assign a quantile-based label to a daily average sentiment score.
    - Below 33rd percentile: "negative_overall"
    - Above 66th percentile: "positive_overall"
    - Otherwise: "neutral_overall"
    """
    if avg < low:
        return "negative_overall"
    elif avg > high:
        return "positive_overall"
    else:
        return "neutral_overall"

# -----------------------------------------------------------------------------
# STEP 8: Store daily sentiment averages and quantile labels in MongoDB.
# Each document contains the date, average adjusted score, and quantile label.
# -----------------------------------------------------------------------------
logger.info("Writing out to collection 'daily_sentiment_average'...")
DAILY_SENT_COLL.delete_many({})
daily_docs = []
for date, avg in daily_avg.items():
    doc = {
        "_id":               date,
        "avg_adjusted_score": round(avg, 6),
        "label_quant":        classify_quantile(avg, p33, p66)
    }
    daily_docs.append(doc)

if daily_docs:
    DAILY_SENT_COLL.insert_many(daily_docs)
    logger.info(f"Inserted {len(daily_docs)} docs into 'daily_sentiment_average'.")
else:
    logger.warning("No docs to insert into 'daily_sentiment_average'.")

# -----------------------------------------------------------------------------
# STEP 9: Visualize the distribution of daily sentiment averages by quantile bucket.
# This violin plot shows how sentiment varies across "negative", "neutral", and
# "positive" days, helping to understand the spread and central tendency of sentiment.
# -----------------------------------------------------------------------------
logger.info("Fetching daily averages for plotting...")
df = pd.DataFrame(list(
    DAILY_SENT_COLL.find(
        {},
        {"_id": 0, "avg_adjusted_score": 1, "label_quant": 1}
    )
))

if not df.empty:
    # Map quantile labels to short bucket names for plotting
    df["bucket"] = df["label_quant"].map({
        "negative_overall": "neg",
        "neutral_overall":  "neu",
        "positive_overall": "pos"
    })

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="bucket",
        y="avg_adjusted_score",
        data=df,
        inner="quartile",
        palette={
            "neg": "#EABE1C",
            "neu": "#E5722A",
            "pos": "#E63946"
        }
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Daily Overall Sentiment")
    plt.xlabel("Bucket")
    plt.ylabel("daily avg adjusted score")
    plt.tight_layout()
    plt.show()
else:
    logger.warning("No daily averages found for plotting.")

# -----------------------------------------------------------------------------
# STEP 10: Match daily sentiment with S&P 500 returns and store results.
# For each date, check if the sign of the sentiment average matches the sign
# of the S&P 500 return (both positive or both negative). Store the results
# in a new collection for further analysis.
# -----------------------------------------------------------------------------
logger.info("Fetching S&P 500 returns from 'sp500_daily_returns'...")
sp_docs = list(SP500_RETURNS_COLL.find({}, {"Date":1, "Return":1}))
if not sp_docs:
    logger.warning("No documents found in 'sp500_daily_returns'. Skipping match step.")
else:
    sp_returns = {d["Date"]: float(d["Return"]) for d in sp_docs if d.get("Date") and d.get("Return") is not None}
    daily_map  = {d["_id"]: d["avg_adjusted_score"] for d in DAILY_SENT_COLL.find({}, {"avg_adjusted_score":1})}

    match_docs = []
    total = matched = 0

    for date, sp_ret in sp_returns.items():
        if date not in daily_map:
            continue
        sent_avg = daily_map[date]
        # A match occurs if both sentiment and return are positive or both are negative
        match    = 1 if (sent_avg > 0 and sp_ret > 0) or (sent_avg < 0 and sp_ret < 0) else 0
        match_docs.append({
            "_id":            date,
            "sentiment_avg":  round(sent_avg, 6),
            "sp500_return":   round(sp_ret,   6),
            "match":          match
        })
        total   += 1
        matched += match

    # Store the match results in MongoDB for further analysis
    DAILY_MATCH_COLL.delete_many({})
    if match_docs:
        DAILY_MATCH_COLL.insert_many(match_docs)
        pct = 100.0 * matched / total if total else 0
        logger.info(f"Inserted {len(match_docs)} docs into 'daily_sentiment_return_match'.")
        print(f"\n→ Matched sentiment vs return on {matched}/{total} dates ({pct:.1f}%).")
    else:
        logger.warning("No match docs to insert; no overlapping dates found.")
