import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# API key for GNews service, used to fetch news articles
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# Maximum number of articles to fetch per API request (default: 100)
GNEWS_MAX = int(os.getenv("GNEWS_MAX", "100"))

# MongoDB connection URI and database name for storing articles
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Minimum delay (in hours) before fetching new articles for the same topic (default: 12)
DEFAULT_DELAY_HOURS = int(os.getenv("DEFAULT_DELAY_HOURS", "12"))

