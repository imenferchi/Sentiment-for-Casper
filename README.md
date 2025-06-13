# Sentiment Analysis for Financial Markets (Sentiment-For-Casper)

![Project Banner](https://via.placeholder.com/800x200?text=Sentiment+Analysis+Dashboard)  
*You can replace this image with an actual screenshot of the dashboard.*

This project provides a multi-phase pipeline for collecting financial news, analyzing sentiment, correlating with market data, and forecasting trends. It includes data extraction, sentiment analysis, correlation analysis, forecasting, and a web dashboard for visualization.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- MongoDB (either local installation or [MongoDB Atlas](https://www.mongodb.com/atlas/database))
- [GNews API Key](https://gnews.io/) (a free tier is available)

### Installation

```sh
# 1. Clone the repository
git clone https://github.com/imenferchi/sentiment-for-casper.git
cd sentiment-for-casper

# 2. Set up a virtual environment (recommended)
python -m venv venv
# On Linux/MacOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install all required dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp phase1_data_extraction/.env.example phase1_data_extraction/.env
# Edit the .env file and fill in your MongoDB URI and GNews API key
```

---

**Project Structure and Usage**

- Each phase of the project is organized in its own directory (data extraction, sentiment analysis, correlation, forecasting, dashboard).
- To run a specific phase, navigate to its directory and execute the relevant Python script.
- The dashboard can be started by running `python app.py` inside the `phase5_dashboard` directory.

---

**Notes**

- Ensure your `.env` file is correctly configured with all required credentials before running the scripts.
- For any issues or questions, please refer to the documentation or open an issue on