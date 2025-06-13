#!/usr/bin/env python3
import os
import sys
import webbrowser
from threading import Timer
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from pymongo import MongoClient, UpdateOne
import certifi
import pytz
from dotenv import load_dotenv
from gnews import GNews
from urllib.parse import urlparse

# -----------------------------------------------------------------------------
# Load environment variables for MongoDB connection.
# This allows secure and flexible configuration of database credentials.
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
# Connect to MongoDB and define collections for dashboard data.
# These collections store sentiment, market, prediction, and news data.
# -----------------------------------------------------------------------------
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db     = client[MONGO_DB_NAME]

daily_sentiment_collection = db['daily_sentiment_average']      # Daily sentiment scores
sp500_collection           = db['sp500_daily_returns']          # S&P 500 daily returns
predictions_collection     = db['market_predictions']           # Model predictions
news_collection            = db['financial_news']               # News articles
match_collection           = db['daily_sentiment_return_match'] # Sentiment/return matches

# -----------------------------------------------------------------------------
# Set up Flask web server and enable CORS for API endpoints.
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Render the main dashboard HTML page."""
    return render_template('dashboard.html')

@app.route('/api/latest_sentiment')
def get_latest_sentiment():
    """
    Return yesterday's sentiment score and label.
    Used for displaying the most recent sentiment index on the dashboard.
    """
    try:
        utc = pytz.UTC
        yesterday_str = (datetime.now(utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        doc = daily_sentiment_collection.find_one({"_id": yesterday_str})
        if not doc:
            return jsonify(success=False, message=f'No sentiment data for {yesterday_str}')
        return jsonify(success=True, data={
            'score': doc.get('avg_adjusted_score', 0),
            'label': doc.get('label_quant', 'neutral'),
            'date':  yesterday_str
        })
    except Exception as e:
        app.logger.exception("Error in get_latest_sentiment")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/yesterday_sp500')
def get_yesterday_sp500():
    """
    Return yesterday's S&P 500 return.
    Used for showing the latest market movement on the dashboard.
    """
    try:
        utc = pytz.UTC
        yesterday_str = (datetime.now(utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        doc = sp500_collection.find_one({"Date": yesterday_str})
        if not doc:
            return jsonify(success=False, message=f'No S&P 500 data for {yesterday_str}')
        return jsonify(success=True, data={
            'return': doc.get('Return', 0),
            'date':   yesterday_str
        })
    except Exception as e:
        app.logger.exception("Error in get_yesterday_sp500")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/yesterday_prediction')
def get_yesterday_prediction():
    """
    Return the model's prediction for yesterday (actually made two days ago).
    Used for advanced analytics and comparing prediction to actual market movement.
    """
    try:
        utc = pytz.UTC
        # Query date: two days ago (prediction made for yesterday)
        target_date_dt = datetime.now(utc) - timedelta(days=2)
        target_date = target_date_dt.strftime('%Y-%m-%d')

        cursor = predictions_collection.find(
            {'Date': target_date}
        ).sort('Confidence', -1).limit(1)
        pred = next(cursor, None)
        if not pred:
            return jsonify(success=False, message=f'No prediction found for {target_date}'), 404

        # Display date: one day after target (i.e. yesterday)
        display_date = (target_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify(success=True, data={
            'direction':  pred.get('Predicted_Direction', 'unknown'),
            'confidence': pred.get('Confidence', 0),
            'date':       display_date
        })
    except Exception as e:
        app.logger.exception("Error in get_yesterday_prediction")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/today_prediction')
def get_today_prediction():
    """
    Return the model's prediction for today (made yesterday).
    Used for displaying the most recent actionable prediction.
    """
    try:
        utc = pytz.UTC
        target_date = (datetime.now(utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        cursor = predictions_collection.find(
            {'Date': target_date}
        ).sort('Confidence', -1).limit(1)
        pred = next(cursor, None)
        if not pred:
            return jsonify(success=False, message=f'No prediction found for {target_date}'), 404
        return jsonify(success=True, data={
            'direction':  pred.get('Predicted_Direction', 'unknown'),
            'confidence': pred.get('Confidence', 0),
            'date':       target_date
        })
    except Exception as e:
        app.logger.exception("Error in get_today_prediction")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/tomorrow_prediction')
def get_tomorrow_prediction():
    """
    Return the model's prediction for tomorrow (made today).
    Used for displaying the latest available forecast.
    """
    try:
        utc = pytz.UTC
        target_date = datetime.now(utc).strftime('%Y-%m-%d')
        cursor = predictions_collection.find(
            {'Date': target_date}
        ).sort('Confidence', -1).limit(1)
        pred = next(cursor, None)
        if not pred:
            return jsonify(success=False, message=f'No prediction found for {target_date}'), 404
        return jsonify(success=True, data={
            'direction':  pred.get('Predicted_Direction', 'unknown'),
            'confidence': pred.get('Confidence', 0),
            'date':       target_date
        })
    except Exception as e:
        app.logger.exception("Error in get_tomorrow_prediction")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/historical_data')
def get_historical_data():
    """
    Return historical sentiment scores and S&P 500 returns for the last ~60 days.
    Used for plotting time series charts on the dashboard.
    """
    try:
        utc = pytz.UTC
        end_date   = datetime.now(utc)
        start_date = end_date - timedelta(days=60)
        sd_str = start_date.strftime('%Y-%m-%d')
        ed_str = end_date.strftime('%Y-%m-%d')

        # Fetch sentiment data for the date range
        sentiment_data = list(daily_sentiment_collection.find(
            {"_id": {"$gte": sd_str, "$lte": ed_str}},
            sort=[("_id", 1)]
        ))

        # Fetch S&P 500 returns for the date range
        sp500_data = list(sp500_collection.find(
            {"Date": {"$gte": sd_str, "$lte": ed_str}},
            sort=[("Date", 1)]
        ))
        if not sp500_data:
            # fallback to datetime range if string keys fail
            start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=utc)
            end_dt   = datetime(end_date.year,   end_date.month,   end_date.day,   tzinfo=utc)
            sp500_data = list(sp500_collection.find(
                {"Date": {"$gte": start_dt, "$lte": end_dt}},
                sort=[("Date", 1)]
            ))

        # Build a mapping of date to S&P 500 return
        sp500_map = {}
        for item in sp500_data:
            dt = item['Date']
            key = dt if isinstance(dt, str) else dt.strftime('%Y-%m-%d')
            sp500_map[key] = item.get('Return', 0)

        # Intersect sentiment and S&P 500 data by date
        labels = []
        sentiment_scores = []
        sp500_returns   = []
        for doc in sentiment_data:
            date_str = doc['_id']
            if date_str in sp500_map:
                labels.append(date_str)
                sentiment_scores.append(doc.get('avg_adjusted_score', 0))
                sp500_returns.append(sp500_map[date_str] * 100)

        return jsonify(success=True, data={
            'labels': labels,
            'sentiment_scores': sentiment_scores,
            'sp500_returns': sp500_returns
        })
    except Exception as e:
        app.logger.exception("Error in get_historical_data")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/today_news')
def get_today_news():
    """
    Fetch and return the top 3 macro-economic news headlines for today using the GNews API.
    Used to display current news context on the dashboard.
    """
    try:
        news_client = GNews(
            language='en',
            country='US',
            period='1d',
            max_results=6
        )
        query = 'Federal Reserve OR inflation OR GDP OR unemployment OR "macro-economics"'
        raw    = news_client.get_news(query)[:3]

        articles = []
        for art in raw:
            pub = art.get('publisher')
            if not pub or not isinstance(pub, str):
                net = urlparse(art.get('url','')).netloc.replace('www.', '')
                pub = net or 'Unknown Source'
            articles.append({
                'title':  art.get('title','No Title'),
                'url':    art.get('url','#'),
                'source': pub
            })
        return jsonify(success=True, articles=articles)
    except Exception:
        app.logger.exception("Failed to fetch macro-economic news via GNews")
        return jsonify(success=False, message='Unable to retrieve news at this time.'), 500

@app.route('/api/statistics')
def get_statistics():
    """
    Return dashboard statistics including:
      - Average number of news articles per day
      - Total number of articles in the database
      - Match rate: percentage of days where sentiment and S&P 500 return had the same sign
    """
    try:
        # Calculate average articles per day using aggregation
        pipeline = [
            {"$project": {"date": {"$substr": ["$publishedAt", 0, 10]}}},
            {"$group": {"_id": "$date", "count": {"$sum": 1}}},
            {"$group": {"_id": None, "avgCount": {"$avg": "$count"}}}
        ]
        res = list(news_collection.aggregate(pipeline))
        avg_articles  = round(res[0]['avgCount']) if res else 0
        total_articles= news_collection.count_documents({})

        total   = match_collection.count_documents({})
        matched = match_collection.count_documents({'match': 1}) if total>0 else 0
        match_rate = (matched/total*100) if total>0 else 0.0

        return jsonify(success=True, data={
            'avg_articles':   avg_articles,
            'total_articles': total_articles,
            'match_rate':     match_rate
        })
    except Exception as e:
        app.logger.exception("Error in get_statistics")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/debug')
def debug_data():
    """
    Debug endpoint to check sample documents and date formats in the database.
    Useful for troubleshooting data issues during development.
    """
    try:
        sample_sent = daily_sentiment_collection.find_one()
        sample_sp   = sp500_collection.find_one()

        utc = pytz.UTC
        yesterday = (datetime.now(utc) - timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify(success=True, data={
            'yesterday_date': yesterday,
            'sample_sentiment': {
                '_id': str(sample_sent.get('_id')) if sample_sent else None,
                'fields': list(sample_sent.keys()) if sample_sent else []
            },
            'sample_sp500': {
                'Date':      str(sample_sp.get('Date')) if sample_sp else None,
                'Date_type': type(sample_sp.get('Date')).__name__ if sample_sp else None,
                'fields':    list(sample_sp.keys()) if sample_sp else []
            }
        })
    except Exception as e:
        app.logger.exception("Error in debug_data")
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/refresh')
def refresh_data():
    """
    Stub endpoint for a front-end “Refresh” button.
    Returns the current server timestamp.
    """
    return jsonify(success=True, timestamp=datetime.now().isoformat())

def open_browser():
    """Automatically open the dashboard in the user's default web browser."""
    webbrowser.open('http://localhost:5000/')

if __name__ == '__main__':
    print("\nStarting Sentiment For-Casper Dashboard...")
    print("Dashboard will open automatically in your browser!")
    print("Press CTRL+C to stop the server\n")
    Timer(1.5, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)
