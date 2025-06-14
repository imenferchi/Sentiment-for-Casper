# Sentiment For-Casper: Financial News Sentiment Analysis & Forecasting

This project provides a complete pipeline for collecting financial news, analyzing sentiment, correlating with S&P 500 market data, forecasting market direction, and visualizing results in a web dashboard.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Variables](#environment-variables)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Certification](#certification)   <!-- Add this line -->

---

## Features

- Automated news collection from GNews API using macroeconomic and financial queries.
- Sentiment analysis of news articles using FinBERT (transformers).
- Correlation analysis between news sentiment and S&P 500 returns.
- Market forecasting using machine learning (Random Forest).
- Interactive dashboard (Flask) to visualize sentiment, market data, predictions, and statistics.

---

## Project Structure

```
Sentiment-For-Casper/
│
├── phase1_data_extraction/      # News extraction, database utilities, config, .env.example
├── phase2_sentiment_analysis/   # Sentiment analysis scripts (FinBERT)
├── phase3_correlation_index/    # Sentiment/market correlation scripts
├── phase4_forecasting_model/    # Forecasting model and backtesting
├── phase5_dashboard/            # Flask dashboard app and templates
├── requirements.txt             # All Python dependencies
├── README.md                    # This file
└── .gitignore                   # Files/folders to ignore in git
```

---

## Prerequisites

- Python 3.10 or higher  
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/atlas/database))
- GNews API Key ([Get one here](https://gnews.io/))
- (Recommended) Git for cloning the repository

---

## Installation & Setup

1. **Clone the repository**
    ```sh
    git clone https://github.com/yourusername/sentiment-for-casper.git
    cd sentiment-for-casper
    ```

2. **Create and activate a virtual environment**
    ```sh
    python -m venv venv
    # On Linux/Mac:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3. **Install all dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
    - Copy the example file and edit it:
      ```sh
      cp phase1_data_extraction/.env.example phase1_data_extraction/.env
      ```
    - Open `phase1_data_extraction/.env` and fill in:
      ```
      MONGODB_URI=your_mongodb_connection_string
      GNEWS_API_KEY=your_gnews_api_key
      MONGO_DB_NAME=fear_index_db
      ```

---

## Environment Variables

Your `.env` file (in `phase1_data_extraction/`) should look like:

```
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority
GNEWS_API_KEY=your_gnews_api_key
MONGO_DB_NAME=fear_index_db
```

---

## How to Run

**1. Extract News (Phase 1)**
```sh
cd phase1_data_extraction
python extract_news.py
```

**2. Run Sentiment Analysis (Phase 2)**
```sh
cd ../phase2_sentiment_analysis
python sentiment_analysis.py
```

**3. Compute Correlation Index (Phase 3)**
```sh
cd ../phase3_correlation_index
python sentiment_correlation_index.py
python sp500_daily_returns.py
```

**4. Run Forecasting Model (Phase 4)**
```sh
cd ../phase4_forecasting_model
python forecasting_model.py
```

**5. Launch the Dashboard (Phase 5)**
```sh
cd ../phase5_dashboard
python app.py
```
- The dashboard will open automatically in your browser at [http://localhost:5000](http://localhost:5000).

---

## Troubleshooting

- **Missing dependencies:**  
  Make sure you ran `pip install -r requirements.txt` in your virtual environment.
- **MongoDB connection errors:**  
  Double-check your `MONGODB_URI` in the `.env` file.
- **GNews API errors:**  
  Ensure your `GNEWS_API_KEY` is valid and not rate-limited.
- **Other issues:**  
  Check the terminal output for error messages and ensure all environment variables are set.

### Error: "Address already in use" when starting the Flask server

If you see an error like:

```
Address already in use
Port 5000 is in use by another program.
```

It means another process is already using port 5000. To fix this:

#### Option 1: Kill the process using port 5000

**On Windows:**

1. Find the process ID (PID) using port 5000:
    ```
    netstat -ano | findstr :5000
    ```
2. Kill the process (replace `<PID>` with the actual number):
    ```
    taskkill /PID <PID> /F
    ```

**On macOS/Linux:**

1. Find the process ID (PID):
    ```
    lsof -i :5000
    ```
2. Kill the process:
    ```
    kill -9 <PID>
    ```

#### Option 2: Run the server on a different port

You can start the Flask app on another port (e.g., 5001):

```
python phase5_dashboard/app.py --port 5001
```
Or modify your code to use a different port.

---

## Certification

We, Giovanni Celeste, Imen Ferchichi, and Robert Jones, hereby certify that we have written the program ourselves with the assistance of generative AI. We have thoroughly tested the program and confirm that it executed without errors or crashes.

---
