import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone

import sys
import os
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Add phase1_data_extraction directory to sys.path for importing shared modules.
# This allows us to reuse database utility functions and configuration.
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1_data_extraction')))
from database import insert_prediction
import config

# -----------------------------------------------------------------------------
# Load environment variables for MongoDB connection.
# This enables secure and flexible configuration of database credentials.
# -----------------------------------------------------------------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fear_index_db")

# -----------------------------------------------------------------------------
# Set up MongoDB client and database handle for storing and retrieving data.
# -----------------------------------------------------------------------------
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB_NAME]

# -----------------------------------------------------------------------------
# Fetch daily average sentiment scores from the database.
# This data is used as an input feature for the forecasting model.
# -----------------------------------------------------------------------------
def get_daily_sentiment_average():
    """
    Retrieve daily average sentiment scores from the database.
    Returns a DataFrame indexed by date, containing the average sentiment for each day.
    """
    daily_sent = list(db["daily_sentiment_average"].find({}, {"_id": 1, "avg_adjusted_score": 1}))
    df = pd.DataFrame(daily_sent)
    df.rename(columns={"_id": "Date", "avg_adjusted_score": "daily_sentiment_avg"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

# -----------------------------------------------------------------------------
# Download historical daily market data for S&P 500 and selected stocks using yfinance.
# This data forms the basis for feature engineering and model training.
# -----------------------------------------------------------------------------
def get_market_data(start_date="1990-01-01"):
    """
    Download historical daily market data for S&P 500 and selected stocks using yfinance.
    Returns a DataFrame with open, high, low, close, and volume for each ticker.
    """
    tickers = ['^GSPC', 'MSFT', 'JNJ', 'JPM', 'XOM', 'WMT']
    data = yf.download(tickers, start=start_date, group_by='ticker')

    processed = {}
    for ticker in tickers:
        if ticker in data:
            df = data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            processed[ticker] = df

    combined = pd.concat(processed.values(), axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined = combined.reset_index().rename(columns={'index': 'Date'})
    return combined

# -----------------------------------------------------------------------------
# Compute the Relative Strength Index (RSI) for a given price series.
# RSI is a momentum indicator used in technical analysis to measure the speed and change of price movements.
# -----------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# -----------------------------------------------------------------------------
# Add technical indicators and engineered features to the market data.
# Features include daily and weekly returns, moving averages, RSI, and volume averages for each ticker.
# These features help the model capture market trends and patterns.
# -----------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    tickers = set(col.split('_')[0] for col in df.columns if '_' in col)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col not in df.columns:
            continue

        df[f"{ticker}_Returns_1d"] = df[close_col].pct_change()
        df[f"{ticker}_Returns_5d"] = df[close_col].pct_change(5)
        df[f"{ticker}_MA_5"] = df[close_col].rolling(5).mean()
        df[f"{ticker}_MA_20"] = df[close_col].rolling(20).mean()
        df[f"{ticker}_RSI_14"] = compute_rsi(df[close_col], 14)

        if f"{ticker}_Volume" in df.columns:
            df[f"{ticker}_Volume_MA_5"] = df[f"{ticker}_Volume"].rolling(5).mean()

    return df.dropna()

# -----------------------------------------------------------------------------
# Select features that are not highly correlated with each other to avoid redundancy.
# Uses mutual information to rank features by their predictive power for the target variable.
# This step helps improve model generalization and interpretability.
# -----------------------------------------------------------------------------
def select_uncorrelated_features(df, target, max_correlation=0.8):
    try:
        corr_matrix = df.drop(columns=['Date']).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > max_correlation)]

        remaining_features = df.drop(columns=to_drop + [target, 'Date']).columns
        if len(remaining_features) < 10:
            corr_with_target = df.drop(columns=['Date']).corr()[target].abs().sort_values(ascending=False)
            remaining_features = corr_with_target.index[1:11]

        mi = mutual_info_classif(df[remaining_features], df[target], random_state=42)
        features = pd.Series(mi, index=remaining_features).sort_values(ascending=False)
        return features.index.tolist()
    except Exception as e:
        print(f"Feature selection error: {e}")
        basic_features = ['^GSPC_Close', '^GSPC_MA_5', '^GSPC_MA_20',
                          '^GSPC_RSI_14', '^GSPC_Volume_MA_5',
                          'MSFT_Close', 'JPM_Close', 'XOM_Close']
        return [f for f in basic_features if f in df.columns]

# -----------------------------------------------------------------------------
# Simulate a rolling-window backtest for the prediction model.
# Trains a Random Forest on a moving window and tests on the next period.
# Returns a DataFrame with predictions, probabilities, and actual outcomes.
# This approach mimics how the model would perform in real-time trading.
# -----------------------------------------------------------------------------
def backtest(df, features, window_size=1000, retrain_every=30):
    df = df.copy()
    predictions = []

    for start in range(window_size, len(df) - 1, retrain_every):
        end = min(start + retrain_every, len(df) - 1)
        train_df = df.iloc[start - window_size:start].copy()
        test_df = df.iloc[start:end].copy()

        X_train, y_train = train_df[features], train_df["Target"]
        X_test, y_test = test_df[features], test_df["Target"]

        # Compute class weights to handle class imbalance in the target variable
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = {0: weights[0], 1: weights[1]}

        # Train Random Forest classifier on the training window
        model = RandomForestClassifier(n_estimators=200, class_weight=weight_dict, random_state=42,
                                       max_depth=5, min_samples_leaf=10, max_features='sqrt')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        for i in range(len(y_pred)):
            predictions.append({
                "Date": test_df.iloc[i]['Date'],
                "True": y_test.iloc[i],
                "Pred": y_pred[i],
                "Proba_UP": y_proba[i],
                "Close": test_df.iloc[i]["^GSPC_Close"]
            })

    return pd.DataFrame(predictions).set_index("Date")

# -----------------------------------------------------------------------------
# MAIN EXECUTION: This section runs the full pipeline for model training,
# backtesting, evaluation, and making a prediction for the next trading day.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load historical market data and daily sentiment averages.
    # Market data includes prices and volumes for S&P 500 and selected stocks.
    # Sentiment data is the daily average sentiment score from news analysis.
    print("Loading market data...")
    market_data = get_market_data(start_date="1990-01-01")

    print("Loading daily sentiment averages...")
    sentiment_df = get_daily_sentiment_average()

    # Step 2: Merge sentiment data with market data by date.
    # This creates a unified dataset for feature engineering and modeling.
    market_data["Date"] = pd.to_datetime(market_data["Date"])
    market_data = market_data.merge(sentiment_df, left_on="Date", right_index=True, how="left")
    market_data["daily_sentiment_avg"] = market_data["daily_sentiment_avg"].fillna(0)

    # Step 3: Add technical and engineered features to the dataset.
    print("Adding features...")
    market_data = add_features(market_data)

    # Step 4: Create the target variable for prediction.
    # Target is 1 if next day's close is up by more than 0.2%, else 0.
    print("Creating target variable...")
    market_data["Target"] = (market_data["^GSPC_Close"].shift(-1) > market_data["^GSPC_Close"] * 1.002).astype(int)
    market_data = market_data.dropna()

    # Step 5: Feature selection to reduce redundancy and improve model performance.
    print("Selecting features...")
    all_features = [col for col in market_data.columns if col not in ['Target', 'Date']]
    if "daily_sentiment_avg" not in all_features:
        all_features.append("daily_sentiment_avg")

    selected_features = select_uncorrelated_features(market_data, "Target", max_correlation=0.75)
    if "daily_sentiment_avg" not in selected_features:
        selected_features.append("daily_sentiment_avg")

    print(f"\nSelected {len(selected_features)} features:")
    print(selected_features)

    if len(selected_features) < 5:
        print("Warning: Very few features selected. Using default features.")
        selected_features = ['^GSPC_Close', '^GSPC_MA_5', '^GSPC_MA_20', '^GSPC_RSI_14', '^GSPC_Volume_MA_5', 'daily_sentiment_avg']
        selected_features = [f for f in selected_features if f in market_data.columns]

    # Step 6: Run rolling-window backtest and evaluate performance.
    # This simulates how the model would perform if deployed in real time.
    print("\nRunning backtest...")
    result = backtest(market_data, selected_features, window_size=1000, retrain_every=30)

    result["Correct"] = (result["True"] == result["Pred"]).astype(int)
    accuracy = result["Correct"].mean()
    print(f"\nBacktest Accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(result["True"], result["Pred"]))

    # Step 7: Calculate and plot cumulative returns for the model and the market.
    # This helps visualize the performance of the strategy compared to simply holding the market.
    result = result.join(market_data.set_index('Date')[["^GSPC_Returns_1d"]], how='left')
    result.rename(columns={"^GSPC_Returns_1d": "Returns_1d"}, inplace=True)

    result["Strategy_Return"] = 0.0
    buy_signals = result["Pred"] == 1
    result.loc[buy_signals, "Strategy_Return"] = result["Returns_1d"].shift(-1)
    result["Strategy_Return"] = result["Strategy_Return"].fillna(0)

    result["Cumulative_Strategy"] = (1 + result["Strategy_Return"]).cumprod()
    result["Cumulative_Market"] = (1 + result["Returns_1d"].fillna(0)).cumprod()

    total_strategy_return = result["Cumulative_Strategy"].iloc[-1] - 1
    total_market_return = result["Cumulative_Market"].iloc[-1] - 1

    print(f"\nTotal Strategy Return: {total_strategy_return:.2%}")
    print(f"Total Market Return: {total_market_return:.2%}")

    plt.figure(figsize=(12, 6))
    result[["Cumulative_Strategy", "Cumulative_Market"]].plot(title="Strategy vs Market Cumulative Returns", lw=2)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------------
    # PREDICTION FOR TOMORROW: Train on all data and predict the next day's market direction.
    # This provides a real-time prediction for the next trading day.
    # -----------------------------------------------------------------------------
    print("\nMaking prediction for tomorrow...")

    latest_row = market_data.iloc[[-1]].copy()

    missing_feats = [feat for feat in selected_features if feat not in latest_row.columns]
    if missing_feats:
        print(f"Missing features in latest data: {missing_feats}")
    else:
        X_latest = latest_row[selected_features]
        X_full = market_data[selected_features]
        y_full = market_data["Target"]

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_full), y=y_full)
        weight_dict = {0: weights[0], 1: weights[1]}

        model = RandomForestClassifier(n_estimators=200, class_weight=weight_dict, random_state=42,
                                       max_depth=5, min_samples_leaf=10, max_features='sqrt')
        model.fit(X_full, y_full)

        tomorrow_pred = model.predict(X_latest)[0]
        tomorrow_proba = model.predict_proba(X_latest)[0][1]

        direction = "UP" if tomorrow_pred == 1 else "DOWN"
        print(f"Predicted market direction for tomorrow: {direction} (Confidence: {tomorrow_proba:.2%})")

        # Prepare prediction document for database insertion.
        # This allows storing the prediction for future analysis and tracking.
        features_cleaned = {feat: float(latest_row.iloc[0][feat]) if isinstance(latest_row.iloc[0][feat], (np.floating, np.integer))
                            else latest_row.iloc[0][feat] for feat in selected_features}

        prediction_doc = {
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Predicted_Direction": direction,
            "Confidence": round(float(tomorrow_proba), 4),
            "Features": features_cleaned,
            "Created_At": datetime.now(timezone.utc)
        }

        # Insert prediction into MongoDB for record-keeping and later analysis.
        try:
            insert_prediction(collection_name="market_predictions", prediction_data=prediction_doc)
            print("Prediction successfully inserted.")
        except Exception as e:
            print(f"Failed to insert prediction: {e}")
