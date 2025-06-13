# Sentiment-for-Casper
# Sentiment For-Casper

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/imenferchi/sentiment-for-casper.git
   cd sentiment-for-casper
   ```

2. **Create a virtual environment (recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your MongoDB URI and API keys.

5. **Run the application:**
   - For the dashboard:
     ```sh
     cd phase5_dashboard
     python app.py
     ```
   - For other scripts, run as needed:
     ```sh
     python phase1_data_extraction/extract_news.py
     ```

## Notes
- Python 3.10+ is recommended.
- Make sure you have a valid `.env` file with all required secrets.