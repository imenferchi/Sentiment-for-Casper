#!/usr/bin/env python3
import logging
from datetime import datetime, timedelta, timezone
import requests
import config
import database
from urllib.parse import urlparse, urlunparse

# -----------------------------------------------------------------------------
# Configure logging for status and error messages.
# This helps track the script's progress and diagnose issues.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# List of macroeconomic and financial news queries for GNews API.
# Each string is a search query for a specific macroeconomic topic.
# These queries are used to fetch relevant news articles for sentiment analysis.
# -----------------------------------------------------------------------------
MACRO_QUERIES = [
    # 1. U.S. Federal Reserve rate decisions and commentary
    '("Fed rate decision" OR "Fed policy statement" OR "federal reserve rate" '
    'OR "Fed funds rate" OR "interest rate hike" OR "rate cut") '
    'AND (outlook OR forecast OR commentary)',

    # 2. U.S. inflation readings (CPI, PPI, PCE)
    '("consumer price index" OR "producer price index" OR "personal consumption expenditures" '
    'OR inflation OR inflationary)',

    # 3. Q1 2025 GDP releases & growth commentary
    '("Q1 2025 GDP" OR "first quarter 2025 GDP" OR "Q1 economic growth 2025" '
    'OR "GDP revision")',

    # 4. U.S. jobs report & unemployment
    '("nonfarm payrolls" OR "unemployment rate" OR "jobs report" OR "labor market")',

    # 5. U.S. Manufacturing PMI & ISM readings
    '("manufacturing PMI" OR "ISM manufacturing" OR "factory PMI" OR "PMI index")',

    # 6. U.S. trade balance, imports, tariffs
    '("trade balance" OR "U.S. imports" OR tariffs OR "trade deficit" '
    'OR "trade policy" OR "import duty")',

    # 7. Federal budget & fiscal policy
    '("federal budget" OR "budget negotiations" OR "fiscal policy" '
    'OR "spending bill")',

    # 8. ECB rates & euro‐area yields
    '("ECB interest rate" OR "European Central Bank" OR "euro area yields" '
    'OR "euro bond yield")',

    # 9. BOJ policy & yen outlook
    '("Bank of Japan" OR "BOJ rate decision" OR "yen exchange rate" '
    'OR "yen outlook")',

    # 10. BoE inflation & gilt yields
    '("Bank of England" OR "BoE rate decision" OR gilts OR "gilt yield" '
    'OR "U.K. inflation")',

    # 11. China PMI & supply‐chain news
    '("China PMI" OR "manufacturing PMI China" OR "supply chain" '
    'OR "logistics disruption" OR "factory output")',

    # 12. Yield‐curve inversion & recession signals
    '("yield curve inversion" OR "inverted yield curve" OR recession '
    'OR "Treasury yield")',

    # 13. VIX & market volatility warnings
    '(VIX OR "volatility index" OR "equity correction" '
    'OR "market correction warning" OR volatility)',

    # 14. Bank stress tests & financial stability
    '("bank stress test" OR "capital adequacy" OR "banking stress" '
    'OR "systemic risk")',

    # 15. S&P 500 earnings vs. analyst estimates
    '("S&P 500 earnings" OR "earnings report" OR "earnings estimates" '
    'OR "EPS surprise")',

    # 16. M&A deals & equity reaction
    '(merger OR acquisition OR "M&A deal" OR "takeover")',

    # 17. Housing starts & mortgage‐rate news
    '("housing starts" OR "building permits" OR "mortgage rate" '
    'OR "housing market")',

    # 18. Retail sales & consumer confidence
    '("retail sales" OR "consumer confidence" OR "consumer sentiment" '
    'OR "shopping trends")',

    # 19. Oil price swings & energy costs
    '("crude oil price" OR "oil price" OR "energy costs" '
    'OR "oil supply" OR "OPEC")',

    # 20. Gold as inflation hedge & safe‐haven
    '("gold price" OR "safe-haven" OR "gold as hedge" OR "precious metals")',

    # 21. Emerging‐markets currency crises
    '("emerging market currency" OR "currency crisis" '
    'OR "capital outflows" OR "exchange controls")',

    # 22. Eurozone CPI & ECB commentary
    '("Eurozone CPI" OR "Euro area inflation" OR "ECB commentary" '
    'OR "euro inflation")',

    # 23. Jackson Hole & Fed speeches
    '("Jackson Hole" OR "Powell speech" OR "Fed Chair Powell" '
    'OR "central bank speech")',

    # 24. U.S. federal budget & market volatility
    '("federal budget" OR "budget negotiations" OR "federal spending" '
    'OR "deficit ceiling" OR "debt limit")',

    # 25. Fed minutes & transcripts
    '("Fed minutes" OR "FOMC minutes" OR "Fed transcript")',

    # 26. ISM Services PMI & service-sector data
    '("services PMI" OR "ISM services" OR "service sector PMI")',

    # 27. Consumer credit & household borrowing
    '("consumer credit" OR "household debt" OR "bank lending")',

    # 28. Corporate debt levels & bond yields
    '("corporate debt" OR "bond yield" OR "credit spread")',

    # 29. US–China trade tensions & tariffs
    '("U.S.-China trade" OR "trade tension" OR "China tariffs" '
    'OR "trade war")',

    # 30. Geopolitical risks & commodity supply
    '("geopolitical risk" OR "commodity supply" OR "oil embargo" '
    'OR "supply shock")',

    # 31. Climate policy & green energy investment
    '("climate policy" OR "green energy" OR "renewable investment" '
    'OR "carbon tax")',

    # 32. Digital currency & central bank digital currencies
    '("central bank digital currency" OR "CBDC" OR "digital currency" '
    'OR "crypto regulation")',
]

def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing query parameters and fragments.
    This helps with deduplication by ensuring URLs are compared in a standard format.
    Returns the cleaned URL string.
    """
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return url

def process_articles(raw: list) -> list:
    """
    Convert raw article data from GNews API into a standardized format.
    Adds a fetch timestamp and sets sentiment to None (to be filled later).
    Returns a list of processed article dictionaries.
    """
    out = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for a in raw:
        url = normalize_url(a.get("url") or "")
        out.append({
            "title":       a.get("title", ""),
            "description": a.get("description", ""),
            "content":     a.get("content", ""),
            "publishedAt": a.get("publishedAt", ""),
            "source":      a.get("source", {}).get("name", ""),
            "url":         url,
            "fetchedAt":   fetched_at,
            "sentiment":   None
        })
    return out

def collect_financial_news(query: str, collection_name: str, from_iso: str, to_iso: str):
    """
    Fetch news articles for a given query and date range from the GNews API.
    Deduplicates articles by URL, processes them into a standard format,
    and inserts new articles into MongoDB.
    """
    logger.info(f"Fetching '{query}' from {from_iso} to {to_iso}…")
    seen_urls = {doc["url"] for doc in database.db[collection_name].find({}, {"url": 1})}
    raw_articles, page, page_size = [], 1, config.GNEWS_MAX
    max_pages = getattr(config, "MAX_PAGES", 30)

    while page <= max_pages:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q":      query,
                "from":   from_iso,
                "to":     to_iso,
                "lang":   "en",
                "max":    page_size,
                "page":   page,
                "apikey": config.GNEWS_API_KEY
            }
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"GNews error {e.response.status_code}: {e.response.text}")
            return

        batch = resp.json().get("articles", [])
        if not batch:
            break

        for art in batch:
            u = normalize_url(art.get("url") or "")
            if u and u not in seen_urls:
                seen_urls.add(u)
                raw_articles.append(art)

        logger.info(f"  page {page}: fetched {len(batch)}, unique so far={len(raw_articles)}")
        if len(batch) < page_size:
            break
        page += 1

    filtered = process_articles(raw_articles)
    inserted = database.insert_articles(collection_name, filtered)
    logger.info(f"Inserted {inserted} new docs into '{collection_name}'")

if __name__ == "__main__":
    # Calculate the UTC date range for "yesterday"
    # This ensures we fetch news for the most recent complete day.
    now_utc = datetime.now(timezone.utc)
    yesterday_date = (now_utc - timedelta(days=1)).date()
    today_date     = now_utc.date()

    # Build ISO 8601 strings for the full "yesterday" window
    from_iso = datetime.combine(
        yesterday_date, datetime.min.time(), tzinfo=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_iso   = datetime.combine(
        today_date, datetime.min.time(), tzinfo=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"===== Collecting news for {yesterday_date} (UTC) =====")
    logger.info(f"From: {from_iso}  To: {to_iso}")

    # Fetch and store news for each macroeconomic query.
    # Each query targets a different aspect of the financial news landscape.
    for q in MACRO_QUERIES:
        collect_financial_news(q, "financial_news", from_iso, to_iso)

    logger.info("Finished collecting yesterday’s news.")
