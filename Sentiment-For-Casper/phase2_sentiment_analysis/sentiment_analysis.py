#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
)

# -----------------------------------------------------------------------------
# Add phase1_data_extraction directory to sys.path for imports.
# This allows importing shared modules (like config and database) from phase 1.
# -----------------------------------------------------------------------------
THIS_FILE   = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent
PHASE1_DIR  = PROJECT_DIR / "phase1_data_extraction"
sys.path.insert(0, str(PHASE1_DIR))

import config                   # Loads environment variables and settings
from database import db         # MongoDB client and database object

# -----------------------------------------------------------------------------
# Set up logging for status and error messages.
# This helps track the script's progress and diagnose issues.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()    # Suppress transformers library warnings

# -----------------------------------------------------------------------------
# Load FinBERT model and tokenizer for financial sentiment analysis.
# FinBERT is a BERT-based model fine-tuned for financial text sentiment.
# -----------------------------------------------------------------------------
FINBERT_MODEL = "yiyanghkust/finbert-tone"
try:
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    # Uncomment below to use GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
except Exception:
    logger.error("Could not load FinBERT. Check your network and the transformers installation.")
    raise

LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

def fetch_all_articles(collection_name: str):
    """
    Retrieve all articles from the specified MongoDB collection.
    Returns a list of (ObjectId, combined_text, publishedAt).
    Combines title, description, and content for sentiment analysis.
    """
    coll = db[collection_name]
    docs = coll.find({}, {"title":1, "description":1, "content":1, "publishedAt":1})
    out = []
    for d in docs:
        title   = d.get("title") or ""
        desc    = d.get("description") or ""
        content = d.get("content") or ""
        text    = " ".join([title, desc, content]).strip()
        out.append((d["_id"], text, d.get("publishedAt")))
    return out

@torch.no_grad()
def compute_finbert_scores(texts: list[str]) -> list[tuple[float,float,float]]:
    """
    Run FinBERT on a list of texts and return sentiment probabilities.
    Each result is a tuple: (prob_negative, prob_neutral, prob_positive).
    """
    all_probs  = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        out   = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().tolist()
        all_probs.extend([tuple(p) for p in probs])
    return all_probs

def label_from_adjusted(score: float, threshold: float = 0.02) -> str:
    """
    Assign sentiment label based on adjusted score and threshold.
    - If score > threshold: "positive"
    - If score < -threshold: "negative"
    - Otherwise: "neutral"
    """
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Main execution: Run sentiment analysis on all articles and store results.
    # -----------------------------------------------------------------------------
    SOURCE_COLL    = "financial_news"
    SENTIMENT_COLL = "financial_news_sentiment"

    logger.info(f"Loading articles from '{SOURCE_COLL}'…")
    articles = fetch_all_articles(SOURCE_COLL)
    if not articles:
        logger.info("No articles found; exiting.")
        sys.exit(0)

    ids, texts, dates = zip(*articles)
    logger.info(f"Running FinBERT on {len(texts)} articles…")
    probs = compute_finbert_scores(list(texts))

    # Calculate raw sentiment scores and their mean for normalization
    raw_scores = [p[2] - p[0] for p in probs]  # positive − negative
    mean_raw   = sum(raw_scores) / len(raw_scores)
    logger.info(f"Mean raw score: {mean_raw:.4f}")

    # Build sentiment documents for MongoDB
    sentiment_docs = []
    counts = {"positive":0,"neutral":0,"negative":0}

    for idx, (p_neg, p_neu, p_pos) in enumerate(probs):
        raw      = p_pos - p_neg
        adjusted = raw - mean_raw
        label    = label_from_adjusted(adjusted)

        counts[label] += 1
        sentiment_docs.append({
            "_id":            ids[idx],
            "date":           (dates[idx] or "").split("T")[0],
            "prob_neg":       p_neg,
            "prob_neu":       p_neu,
            "prob_pos":       p_pos,
            "raw_score":      raw,
            "adjusted_score": adjusted,
            "sentiment":      label,
        })

    # Replace old sentiment collection with new results
    db.drop_collection(SENTIMENT_COLL)
    if sentiment_docs:
        db[SENTIMENT_COLL].insert_many(sentiment_docs)
    logger.info(
        "Inserted %d docs: %d positive, %d neutral, %d negative",
        len(sentiment_docs),
        counts["positive"],
        counts["neutral"],
        counts["negative"]
    )
