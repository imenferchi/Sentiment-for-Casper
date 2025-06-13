from pymongo import MongoClient
import certifi  
import config

# -----------------------------------------------------------------------------
# Initialize MongoDB client using credentials from config.py.
# certifi.where() ensures SSL certificate verification for secure connections.
# -----------------------------------------------------------------------------
client = MongoClient(config.MONGODB_URI, tlsCAFile=certifi.where())
db = client[config.MONGO_DB_NAME]

def insert_articles(collection_name, articles):
    """
    Insert new articles into the specified MongoDB collection.
    - Skips articles with URLs already present in the collection to avoid duplicates.
    - Returns the number of articles actually inserted.

    Args:
        collection_name (str): Name of the MongoDB collection to insert into.
        articles (list): List of article dictionaries to insert.

    Returns:
        int: Number of new articles inserted.
    """
    if not articles:
        return 0

    collection = db[collection_name]

    # Fetch all existing URLs in the collection to check for duplicates.
    try:
        existing_urls = set(doc["url"] for doc in collection.find({}, {"url": 1}) if "url" in doc)
    except Exception as e:
        print(f"Failed to fetch existing URLs: {e}")
        return 0

    # Filter out articles whose URLs are already in the database.
    new_articles = []
    for article in articles:
        url = article.get("url")
        if url and url not in existing_urls:
            new_articles.append(article)

    # Insert only the new, non-duplicate articles.
    if new_articles:
        try:
            collection.insert_many(new_articles)
        except Exception as e:
            print(f"Failed to insert articles: {e}")
            return 0

    return len(new_articles)

def insert_prediction(collection_name, prediction_data):
    """
    Insert a single prediction document into the specified MongoDB collection.
    - Expects prediction_data to be a dictionary.
    - Prints a confirmation message if successful.

    Args:
        collection_name (str): Name of the MongoDB collection to insert into.
        prediction_data (dict): The prediction data to insert.
    """
    collection = db[collection_name]

    if isinstance(prediction_data, dict):
        try:
            collection.insert_one(prediction_data)
            print(f"Inserted prediction for {prediction_data.get('Date')}")
        except Exception as e:
            print(f"Failed to insert prediction: {e}")
    else:
        print("Prediction data must be a dictionary")


#Database.py