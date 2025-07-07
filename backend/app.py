from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import pytz
from urllib.parse import urljoin
import logging
from functools import wraps
import time
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from collections import Counter
import heapq


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# Configuration
BASE_URL = "https://www.moneycontrol.com"
NEWS_URL = f"{BASE_URL}/news/business/markets"
ET_BASE_URL = "https://economictimes.indiatimes.com"
ET_NEWS_URL = f"{ET_BASE_URL}/markets/stocks/news"
TRADINGVIEW_URL = "https://in.tradingview.com/markets/stocks-india/ideas/"
CACHE_DURATION = 300  # 5 minutes

LIVE_DATA_CACHE_DURATION = 60


live_index_data = {}
data_lock = threading.Lock()


# Enhanced stock list with more Indian companies
INDIAN_STOCKS = [
    {'name': 'Reliance Industries', 'symbol': 'RIL', 'aliases': ['Reliance', 'RIL']},
    {'name': 'Tata Consultancy Services', 'symbol': 'TCS', 'aliases': ['TCS', 'Tata Consultancy']},
    {'name': 'Infosys', 'symbol': 'INFY', 'aliases': ['Infosys', 'INFY']},
    {'name': 'HDFC Bank', 'symbol': 'HDFCBANK', 'aliases': ['HDFC Bank', 'HDFCBANK', 'HDFC']},
    {'name': 'Wipro', 'symbol': 'WIPRO', 'aliases': ['Wipro', 'WIPRO']},
    {'name': 'Bharti Airtel', 'symbol': 'BHARTIARTL', 'aliases': ['Bharti Airtel', 'Airtel', 'BHARTIARTL']},
    {'name': 'ITC', 'symbol': 'ITC', 'aliases': ['ITC']},
    {'name': 'State Bank of India', 'symbol': 'SBIN', 'aliases': ['SBI', 'SBIN', 'State Bank']},
    {'name': 'Larsen & Toubro', 'symbol': 'LT', 'aliases': ['L&T', 'LT', 'Larsen']},
    {'name': 'HCL Technologies', 'symbol': 'HCLTECH', 'aliases': ['HCL', 'HCLTECH']},
    {'name': 'Axis Bank', 'symbol': 'AXISBANK', 'aliases': ['Axis Bank', 'AXISBANK']},
    {'name': 'Maruti Suzuki', 'symbol': 'MARUTI', 'aliases': ['Maruti', 'MARUTI']},
    {'name': 'Bajaj Finance', 'symbol': 'BAJFINANCE', 'aliases': ['Bajaj Finance', 'BAJFINANCE']},
    {'name': 'Asian Paints', 'symbol': 'ASIANPAINT', 'aliases': ['Asian Paints', 'ASIANPAINT']},
    {'name': 'Hindustan Unilever', 'symbol': 'HINDUNILVR', 'aliases': ['HUL', 'HINDUNILVR', 'Hindustan Unilever']},
    {'name': 'Mahindra & Mahindra', 'symbol': 'M&M', 'aliases': ['M&M', 'Mahindra']},
    {'name': 'Titan Company', 'symbol': 'TITAN', 'aliases': ['Titan', 'TITAN']},
    {'name': 'Nestle India', 'symbol': 'NESTLEIND', 'aliases': ['Nestle', 'NESTLEIND']},
    {'name': 'Adani Enterprises', 'symbol': 'ADANIENT', 'aliases': ['Adani', 'ADANIENT']},
    {'name': 'Tata Motors', 'symbol': 'TATAMOTORS', 'aliases': ['Tata Motors', 'TATAMOTORS']},
    {'name': 'NTPC', 'symbol': 'NTPC', 'aliases': ['NTPC']},
    {'name': 'Coal India', 'symbol': 'COALINDIA', 'aliases': ['Coal India', 'COALINDIA']},
    {'name': 'Power Grid Corporation', 'symbol': 'POWERGRID', 'aliases': ['Power Grid', 'POWERGRID']},
    {'name': 'Sun Pharmaceutical', 'symbol': 'SUNPHARMA', 'aliases': ['Sun Pharma', 'SUNPHARMA']},
    {'name': 'Dr. Reddy\'s Laboratories', 'symbol': 'DRREDDY', 'aliases': ['Dr Reddy', 'DRREDDY']},
    {'name': 'Tech Mahindra', 'symbol': 'TECHM', 'aliases': ['Tech Mahindra', 'TECHM']},
    {'name': 'UltraTech Cement', 'symbol': 'ULTRACEMCO', 'aliases': ['UltraTech', 'ULTRACEMCO']},
    {'name': 'Bajaj Auto', 'symbol': 'BAJAJ-AUTO', 'aliases': ['Bajaj Auto', 'BAJAJ-AUTO']},
    {'name': 'Cipla', 'symbol': 'CIPLA', 'aliases': ['Cipla', 'CIPLA']},
    {'name': 'Grasim Industries', 'symbol': 'GRASIM', 'aliases': ['Grasim', 'GRASIM']},
    {'name': 'JSW Steel', 'symbol': 'JSWSTEEL', 'aliases': ['JSW Steel', 'JSWSTEEL']},
    {'name': 'Tata Steel', 'symbol': 'TATASTEEL', 'aliases': ['Tata Steel', 'TATASTEEL']},
    {'name': 'Hero MotoCorp', 'symbol': 'HEROMOTOCO', 'aliases': ['Hero MotoCorp', 'HEROMOTOCO']},
    {'name': 'Britannia Industries', 'symbol': 'BRITANNIA', 'aliases': ['Britannia', 'BRITANNIA']},
    {'name': 'Eicher Motors', 'symbol': 'EICHERMOT', 'aliases': ['Eicher Motors', 'EICHERMOT']},
    {'name': 'Nifty 50', 'symbol': 'NIFTY', 'aliases': ['Nifty', 'NIFTY', 'Nifty 50']},
    {'name': 'Sensex', 'symbol': 'SENSEX', 'aliases': ['Sensex', 'SENSEX', 'BSE Sensex']},
]

# Sentiment keywords for financial news
POSITIVE_KEYWORDS = [
    'profit', 'gain', 'rise', 'increase', 'growth', 'surge', 'boost', 'up', 'higher', 'positive',
    'bullish', 'rally', 'strong', 'outperform', 'beat', 'exceed', 'improvement', 'expansion',
    'breakthrough', 'success', 'achievement', 'milestone', 'record', 'all-time high', 'soar',
    'optimistic', 'confident', 'upgrade', 'target', 'buy', 'overweight', 'recommend'
]

NEGATIVE_KEYWORDS = [
    'loss', 'decline', 'fall', 'decrease', 'drop', 'crash', 'plunge', 'down', 'lower', 'negative',
    'bearish', 'sell-off', 'weak', 'underperform', 'miss', 'below', 'concern', 'worry', 'problem',
    'issue', 'challenge', 'difficulty', 'crisis', 'risk', 'threat', 'volatile', 'uncertainty',
    'downgrade', 'sell', 'underweight', 'caution', 'warning', 'alert', 'disappointing'
]

# Simple in-memory cache (no Redis dependency)
cache = {}
rate_limit_store = {}


def get_market_status(timezone_str):
    """Get market status based on timezone"""
    try:
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz)
        current_hour = local_time.hour
        current_minute = local_time.minute
        weekday = local_time.weekday()
        
        # Weekend check
        if weekday >= 5:  
            return "closed"
        
        
        if timezone_str == 'Asia/Kolkata':  # NSE
            if 9 <= current_hour < 15 or (current_hour == 15 and current_minute <= 30):
                return "open"
        
        return "closed"
    
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return "unknown"

def cache_response(timeout=CACHE_DURATION):
    """Simple in-memory cache decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"api:{f.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()
            
            # Check if cached and not expired
            if cache_key in cache:
                cached_time, cached_result = cache[cache_key]
                if current_time - cached_time < timeout:
                    return cached_result
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            cache[cache_key] = (current_time, result)
            
            # Clean old cache entries
            for key in list(cache.keys()):
                if current_time - cache[key][0] > timeout:
                    del cache[key]
            
            return result
        return decorated_function
    return decorator

def safe_api_call(f):
    """Decorator for safe API calls with logging"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            logger.info(f"API call: {f.__name__}")
            result = f(*args, **kwargs)
            logger.info(f"API call successful: {f.__name__}")
            return result
        except Exception as e:
            logger.error(f"API call failed: {f.__name__} - {str(e)}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    return decorated_function

def rate_limit(max_requests=60, window=60):
    """Simple in-memory rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            key = f"rate_limit:{client_ip}:{f.__name__}"
            current_time = time.time()
            
            # Clean old rate limit entries
            for k in list(rate_limit_store.keys()):
                if current_time - rate_limit_store[k]['timestamp'] > window:
                    del rate_limit_store[k]
            
            # Check rate limit
            if key in rate_limit_store:
                if rate_limit_store[key]['count'] >= max_requests:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                rate_limit_store[key]['count'] += 1
            else:
                rate_limit_store[key] = {'count': 1, 'timestamp': current_time}
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def simple_summarize(text, max_sentences=3):
    """Simple extractive summarization using sentence scoring"""
    if not text or len(text.strip()) < 50:
        return text
    
    try:
        # Tokenize sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Simple approach: score sentences based on word frequency
        words = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalpha() and word not in stop_words]
        
        if not words:
            return ' '.join(sentences[:max_sentences])
        
        # Calculate word frequency
        word_freq = Counter(words)
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalpha() and word not in stop_words]
            
            if sentence_words:
                # Score based on word frequency and sentence position (earlier sentences get slight boost)
                word_score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
                position_score = 1.0 / (i + 1) * 0.1  # Small position boost
                sentence_scores[sentence] = word_score + position_score
        
        # Get top sentences
        if sentence_scores:
            top_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    summary_sentences.append(sentence)
            
            return ' '.join(summary_sentences)
        else:
            return ' '.join(sentences[:max_sentences])
    
    except Exception as e:
        print(f"Error in summarization: {e}")
        # Fallback: return first few sentences
        try:
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:max_sentences])
        except:
            return text[:500] + "..." if len(text) > 500 else text

def analyze_sentiment(text):
    """Analyze sentiment of text and return score"""
    if not text:
        return 0.0
    
    try:
        # Use TextBlob for basic sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # Enhance with keyword-based analysis
        text_lower = text.lower()
        positive_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text_lower)
        negative_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text_lower)
        
        # Combine TextBlob and keyword analysis
        if positive_count + negative_count > 0:
            keyword_score = (positive_count - negative_count) / (positive_count + negative_count)
            # Weighted average (favor keyword analysis for financial news)
            final_score = (polarity * 0.4) + (keyword_score * 0.6)
        else:
            final_score = polarity
        
        return max(-1.0, min(1.0, final_score))
    
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0.0

def extract_stock_names_with_sentiment(text):
    """Extract stock names and their sentiment from text"""
    if not text:
        return []
    
    found_stocks = []
    text_lower = text.lower()
    
    # Overall sentiment of the text
    overall_sentiment = analyze_sentiment(text)
    
    for stock in INDIAN_STOCKS:
        # Check if any alias matches
        for alias in stock['aliases']:
            if alias.lower() in text_lower:
                # Extract context around the stock mention for more accurate sentiment
                pattern = re.compile(r'.{0,150}' + re.escape(alias.lower()) + r'.{0,150}', re.IGNORECASE)
                matches = pattern.findall(text)
                
                if matches:
                    # Analyze sentiment of the context
                    context_sentiment = analyze_sentiment(' '.join(matches))
                    
                    # Use context sentiment if strong enough, otherwise use overall sentiment
                    final_sentiment = context_sentiment if abs(context_sentiment) > 0.15 else overall_sentiment
                    
                    # Determine sentiment label
                    if final_sentiment > 0.1:
                        sentiment_label = 'positive'
                    elif final_sentiment < -0.1:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    found_stocks.append({
                        'name': stock['name'],
                        'symbol': stock['symbol'],
                        'sentiment': round(final_sentiment, 3),
                        'sentiment_label': sentiment_label
                    })
                    break
    
    # Remove duplicates
    unique_stocks = []
    seen = set()
    for stock in found_stocks:
        if stock['symbol'] not in seen:
            unique_stocks.append(stock)
            seen.add(stock['symbol'])
    
    return unique_stocks

def extract_stock_names(text):
    """Legacy function - now returns stocks with sentiment"""
    stocks_with_sentiment = extract_stock_names_with_sentiment(text)
    # Convert to old format for backward compatibility
    return [{'name': stock['name'], 'symbol': stock['symbol']} for stock in stocks_with_sentiment]

TRADINGVIEW_URL = "https://in.tradingview.com/markets/stocks-india/ideas/"

from urllib.parse import urljoin

def get_tradingview_ideas():
    """Get TradingView ideas for Indian stocks"""
    try:
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        all_ideas = soup.find_all("a", class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl")
        all_conditions = soup.find_all("span", class_="visually-hidden-label-cbI7LT3N")

        ideas_list = []

        for i in range(min(len(all_ideas), len(all_conditions))):
            try:
                idea_tag = all_ideas[i]
                condition_tag = all_conditions[i]

                idea_href = idea_tag.get("href")
                if not idea_href:
                    # Skip this idea if no href
                    continue

                full_link = urljoin("https://in.tradingview.com", idea_href)
                stock_split = idea_href.strip("/").split("/")
                stock_symbol = stock_split[4] if len(stock_split) > 4 else "Unknown"
                title = idea_tag.get_text(strip=True) or stock_symbol

                condition_text = condition_tag.get_text(strip=True)

                if condition_text == "Long":
                    signal_label = "BUY"
                    signal_color = "green"
                elif condition_text == "Short":
                    signal_label = "SELL"
                    signal_color = "red"
                else:
                    signal_label = "EDUCATIONAL"
                    signal_color = "blue"

                ideas_list.append({
                    "stock_symbol": stock_symbol,
                    "title": title,
                    "link": full_link,
                    "condition": condition_text or "Educational",
                    "signal_label": signal_label,
                    "signal_color": signal_color
                })

            except Exception as e:
                print(f"[WARN] Skipped idea due to error: {e}")
                continue

        return ideas_list[:50]

    except Exception as e:
        print(f"[ERROR] Failed to fetch TradingView ideas: {e}")
        return []

    

@app.route("/api/health")
def health_check():
    """Enhanced health check endpoint"""
    global live_index_data
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_available": True,
        "live_indices_loaded": len(live_index_data),
        "sources": ["MoneyControl", "Economic Times", "TradingView", "Yahoo Finance"],
        "features": [
            "summarization", 
            "sentiment_analysis", 
            "stock_extraction", 
            "tradingview_ideas",
            "live_index_data",
            "historical_data",
            "market_comparison"
        ]
    })

@app.route("/api/tradingview/ideas")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=600)  # Cache for 10 minutes
def get_tradingview_ideas_api():
    """Get TradingView ideas for Indian stocks"""
    try:
        ideas = get_tradingview_ideas()
        return jsonify({
            "ideas": ideas,
            "total_count": len(ideas),
            "source": "TradingView",
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/indices/live")
@rate_limit(max_requests=120, window=60)  # Higher limit for live data
def get_live_indices():
    """Get live data for all major indices"""
    global live_index_data
    
    try:
        with data_lock:
            if not live_index_data:
                # If no cached data, fetch immediately
                update_all_indices()
            
            # Check if data is stale (older than 2 minutes)
            current_time = datetime.now()
            for index_key, data in live_index_data.items():
                if data and 'last_updated' in data:
                    last_updated = datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
                    if (current_time - last_updated.replace(tzinfo=None)).seconds > 120:
                        # Data is stale, trigger update in background
                        threading.Thread(target=update_all_indices).start()
                        break
        
        return jsonify({
            "indices": live_index_data,
            "total_count": len(live_index_data),
            "last_updated": datetime.now().isoformat(),
            "cache_duration": f"{LIVE_DATA_CACHE_DURATION} seconds"
        })
    
    except Exception as e:
        logger.error(f"Error in get_live_indices: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/indices/comparison")
@rate_limit(max_requests=30, window=60)
def get_indices_comparison():
    """Get comparison data for all major indices"""
    try:
        global live_index_data
        
        with data_lock:
            if not live_index_data:
                update_all_indices()
        
        comparison_data = []
        for index_key, data in live_index_data.items():
            if data:
                comparison_data.append({
                    'index': index_key,
                    'name': data['name'],
                    'current_price': data['current_price'],
                    'change': data['change'],
                    'change_percent': data['change_percent'],
                    'market_status': data['market_status'],
                    'currency': data['currency']
                })
        
        # Sort by change percentage (descending)
        comparison_data.sort(key=lambda x: x['change_percent'], reverse=True)
        
        return jsonify({
            "comparison": comparison_data,
            "best_performer": comparison_data[0] if comparison_data else None,
            "worst_performer": comparison_data[-1] if comparison_data else None,
            "last_updated": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in indices comparison: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/indices/summary")
@rate_limit(max_requests=60, window=60)
def get_market_summary():
    """Get overall market summary"""
    try:
        global live_index_data
        
        with data_lock:
            if not live_index_data:
                update_all_indices()
        
        summary = {
            "total_indices": len(live_index_data),
            "indices_up": 0,
            "indices_down": 0,
            "indices_unchanged": 0,
            "markets_open": 0,
            "markets_closed": 0,
            "average_change": 0,
            "indices_data": []
        }
        
        total_change = 0
        for index_key, data in live_index_data.items():
            if data:
                change_percent = data['change_percent']
                
                if change_percent > 0:
                    summary["indices_up"] += 1
                elif change_percent < 0:
                    summary["indices_down"] += 1
                else:
                    summary["indices_unchanged"] += 1
                
                if data['market_status'] == 'open':
                    summary["markets_open"] += 1
                else:
                    summary["markets_closed"] += 1
                
                total_change += change_percent
                
                summary["indices_data"].append({
                    'index': index_key,
                    'name': data['name'],
                    'change_percent': change_percent,
                    'market_status': data['market_status']
                })
        
        if live_index_data:
            summary["average_change"] = round(total_change / len(live_index_data), 2)
        
        summary["market_sentiment"] = "positive" if summary["average_change"] > 0 else "negative" if summary["average_change"] < 0 else "neutral"
        summary["last_updated"] = datetime.now().isoformat()
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error in market summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tradingview/ideas/by-condition")
@rate_limit(max_requests=30, window=60)
def get_ideas_by_condition():
    """Get TradingView ideas filtered by condition (Long/Short/Educational)"""
    condition = request.args.get('condition', '').title()
    
    if condition not in ['Long', 'Short', 'Educational']:
        return jsonify({"error": "Invalid condition. Use 'Long', 'Short', or 'Educational'"}), 400
    
    try:
        all_ideas = get_tradingview_ideas()
        filtered_ideas = [idea for idea in all_ideas if idea['condition'] == condition]
        
        return jsonify({
            "ideas": filtered_ideas,
            "condition": condition,
            "count": len(filtered_ideas),
            "source": "TradingView"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Add this new endpoint to your existing Flask app

@app.route("/api/news-with-ideas")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=300)  # Cache for 5 minutes


def get_news_with_tradingview_ideas():
    """Get news with summaries and TradingView trading ideas combined"""
    try:
        # Get news from both sources
        mc_news = get_moneycontrol_news()
        et_news = get_economictimes_news()
        
        # Combine news
        all_news = mc_news + et_news
        
        # Enhanced news with better summaries and sentiment
        enhanced_news = []
        for news_item in all_news[:15]:  # Limit to 15 articles
            enhanced_item = {
                'title': news_item.get('title', ''),
                'description': news_item.get('description', ''),
                'link': news_item.get('link', ''),
                'source': news_item.get('source', ''),
                'publishedAt': news_item.get('publishedAt', ''),
                'author': news_item.get('author', ''),
                'stocks': news_item.get('stocks', []),
                'stocks_with_sentiment': news_item.get('stocks_with_sentiment', [])
            }
            # Generate article_id
            enhanced_item['article_id'] = str(hash(news_item.get('title', '') + news_item.get('link', '')))
            
            # Generate better summary if description exists
            if news_item.get('description'):
                enhanced_item['summary'] = simple_summarize(news_item['description'], max_sentences=2)
            else:
                enhanced_item['summary'] = "Summary not available"
            
            # Add overall sentiment
            full_text = (news_item.get('title', '') + " " + news_item.get('description', ''))
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
            
            enhanced_news.append(enhanced_item)
        
        # Get TradingView ideas
        tradingview_ideas = get_enhanced_tradingview_ideas()
        
        return jsonify({
            "news": enhanced_news,
            "tradingview_ideas": tradingview_ideas,
            "metadata": {
                "news_count": len(enhanced_news),
                "ideas_count": len(tradingview_ideas),
                "last_updated": datetime.now().isoformat(),
                "sources": ["MoneyControl", "Economic Times", "TradingView"]
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def fetch_indices_from_moneycontrol():
    results = {}

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.93 Safari/537.36"
        )
    }

    try:
        url = "https://www.moneycontrol.com/stocksmarketsindia/"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        print("===== RAW HTML START =====")
        print(r.text)
        print("===== RAW HTML END =====")
        soup = BeautifulSoup(r.text, "lxml")

        # Row 1 - Nifty 50
        tr_nifty = soup.select_one("#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(1)")
        cells = tr_nifty.find_all("td")
        results["NIFTY50"] = {
            "name": cells[0].get_text(strip=True),
            "current_price": float(cells[1].get_text(strip=True).replace(",", "")),
            "change": float(cells[2].get_text(strip=True).replace(",", "")),
            "change_percent": float(cells[3].get_text(strip=True).replace("%", "")),
            "last_updated": datetime.now().isoformat(),
            "market_status": "open",
            "currency": "INR"
        }

        # Row 2 - Sensex
        tr_sensex = soup.select_one("#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(2)")
        cells = tr_sensex.find_all("td")
        results["SENSEX"] = {
            "name": cells[0].get_text(strip=True),
            "current_price": float(cells[1].get_text(strip=True).replace(",", "")),
            "change": float(cells[2].get_text(strip=True).replace(",", "")),
            "change_percent": float(cells[3].get_text(strip=True).replace("%", "")),
            "last_updated": datetime.now().isoformat(),
            "market_status": "open",
            "currency": "INR"
        }

        # Row 3 - Bank Nifty
        tr_banknifty = soup.select_one("#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(3)")
        cells = tr_banknifty.find_all("td")
        results["BANKNIFTY"] = {
            "name": cells[0].get_text(strip=True),
            "current_price": float(cells[1].get_text(strip=True).replace(",", "")),
            "change": float(cells[2].get_text(strip=True).replace(",", "")),
            "change_percent": float(cells[3].get_text(strip=True).replace("%", "")),
            "last_updated": datetime.now().isoformat(),
            "market_status": "open",
            "currency": "INR"
        }

    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        results["NIFTY50"] = None
        results["SENSEX"] = None
        results["BANKNIFTY"] = None

    return results
        
def get_enhanced_tradingview_ideas():
    """Get TradingView ideas with enhanced signal labeling"""
    try:
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the stock ideas
        all_ideas = soup.find_all(class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl")

        # Find all the conditions (long, short, educational, etc.)
        all_conditions = soup.find_all(class_="visually-hidden-label-cbI7LT3N", name="span")

        ideas_list = []

        for idea in all_ideas:
            try:
                idea_href = idea.get("href")
                if not idea_href:
                    continue  # Skip if href is missing

                full_link = urljoin("https://in.tradingview.com", idea_href)
                stock_split = idea_href.strip("/").split("/")
                stock_symbol = stock_split[4] if len(stock_split) > 4 else "Unknown"
                title = idea.get_text(strip=True) or stock_symbol

                ideas_list.append({
                    "stock_symbol": stock_symbol,
                    "title": title,
                    "link": full_link,
                    "condition": "Educational"
                })
            except Exception as e:
                print(f"Error processing idea: {e}")
                continue

        # Extract conditions
        conditions = []
        for span in all_conditions:
            try:
                condition_text = span.get_text(strip=True) if span.get_text() else ""
                if not condition_text:
                    conditions.append("Educational")
                elif condition_text in ["Long", "Short"]:
                    conditions.append(condition_text)
                else:
                    conditions.append("Educational")
            except Exception as e:
                print(f"Error processing condition: {e}")
                conditions.append("Educational")

        # Match conditions with ideas and add signal labeling
        enhanced_ideas = []
        for i, idea in enumerate(ideas_list):
            if i < len(conditions):
                idea["condition"] = conditions[i]

            condition = idea["condition"]
            if condition == "Long":
                idea["signal_label"] = "BUY"
                idea["signal_color"] = "green"
            elif condition == "Short":
                idea["signal_label"] = "SELL"
                idea["signal_color"] = "red"
            else:
                idea["signal_label"] = "EDUCATIONAL"
                idea["signal_color"] = "blue"

            enhanced_ideas.append(idea)

        return enhanced_ideas[:30]

    except Exception as e:
        print(f"Error fetching TradingView ideas: {e}")
        return []

# Enhanced version of your existing TradingView endpoint
@app.route("/api/tradingview/ideas/enhanced")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=600)  # Cache for 10 minutes
def get_enhanced_tradingview_ideas_api():
    """Get enhanced TradingView ideas with signal labeling"""
    try:
        ideas = get_enhanced_tradingview_ideas()
        
        # Group ideas by signal type for better organization
        buy_signals = [idea for idea in ideas if idea['signal_label'] == 'BUY']
        sell_signals = [idea for idea in ideas if idea['signal_label'] == 'SELL']
        educational = [idea for idea in ideas if idea['signal_label'] == 'EDUCATIONAL']
        
        return jsonify({
            "ideas": ideas,
            "grouped_ideas": {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "educational": educational
            },
            "summary": {
                "total_ideas": len(ideas),
                "buy_signals_count": len(buy_signals),
                "sell_signals_count": len(sell_signals),
                "educational_count": len(educational)
            },
            "source": "TradingView",
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper endpoint to test news summaries separately
@app.route("/api/news/enhanced-summary")
@rate_limit(max_requests=30, window=60)
def get_enhanced_news_summary():
    """Get news with enhanced summaries and sentiment analysis"""
    try:
        # Get news from both sources
        mc_news = get_moneycontrol_news()
        et_news = get_economictimes_news()
        
        # Combine news
        all_news = mc_news + et_news
        
        # Enhanced news with better summaries
        enhanced_news = []
        for news_item in all_news[:10]:  # Limit to 10 articles for testing
            enhanced_item = news_item.copy()
            
            # Generate better summary if description exists
            if news_item.get('description') and len(news_item.get('description', '')) > 50:
                enhanced_item['summary'] = simple_summarize(news_item['description'], max_sentences=2)
                enhanced_item['has_summary'] = True
            elif news_item.get('title'):
                # If no description, create a brief summary from title
                enhanced_item['summary'] = news_item['title']
                enhanced_item['has_summary'] = False
            else:
                enhanced_item['summary'] = "Summary not available"
                enhanced_item['has_summary'] = False
            
            # Add overall sentiment
            full_text = (news_item.get('title', '') + " " + news_item.get('description', ''))
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
            
            # Add text length info for debugging
            enhanced_item['debug_info'] = {
                'title_length': len(news_item.get('title', '')),
                'description_length': len(news_item.get('description', '')),
                'summary_length': len(enhanced_item['summary'])
            }
            
            enhanced_news.append(enhanced_item)
        
        return jsonify({
            "news": enhanced_news,
            "total_count": len(enhanced_news),
            "sources": ["MoneyControl", "Economic Times"],
            "last_updated": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this endpoint to test TradingView scraping separately
@app.route("/api/tradingview/test")
@rate_limit(max_requests=10, window=60)
def test_tradingview_scraping():
    """Test TradingView scraping functionality"""
    try:
        ideas = get_enhanced_tradingview_ideas()
        
        return jsonify({
            "status": "success",
            "ideas_found": len(ideas),
            "sample_ideas": ideas[:5],  # Show first 5 ideas
            "signal_distribution": {
                "buy": len([i for i in ideas if i['signal_label'] == 'BUY']),
                "sell": len([i for i in ideas if i['signal_label'] == 'SELL']),
                "educational": len([i for i in ideas if i['signal_label'] == 'EDUCATIONAL'])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/api/news")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_news():
    """Get news from both MoneyControl and Economic Times"""
    try:
        # Get news from both sources
        mc_news = get_moneycontrol_news()
        et_news = get_economictimes_news()
        
        # Combine and return
        all_news = mc_news + et_news
        
        # Sort by recency if possible
        return all_news[:20]  # Return top 20 articles
    except Exception as e:
        return {"error": str(e)}

def get_moneycontrol_news():
    """Get news from MoneyControl"""
    try:
        response = requests.get(NEWS_URL, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # Select the top 10 news links from the main page
        for a_tag in soup.select("h2 a")[:10]:
            title = a_tag.get('title') or a_tag.text.strip()
            href = a_tag.get('href')
            full_link = urljoin(BASE_URL, href)
            
            published_at, author, description = get_mc_article_metadata(full_link)
            
            # Extract stocks with sentiment
            stocks_with_sentiment = extract_stock_names_with_sentiment(title + " " + (description or ""))

            # Add chart data for first 3 stocks
           
            news_items.append({
                'title': title,
                'link': full_link,
                'publishedAt': published_at,
                'author': author,
                'description': description,
                'source': 'MoneyControl',
                'stocks': extract_stock_names(title + " " + (description or "")),  # Legacy format
                'stocks_with_sentiment': stocks_with_sentiment
            })
        
        return news_items
    except Exception as e:
        print(f"Error fetching MoneyControl news: {e}")
        return []

def get_economictimes_news():
    """Get news from Economic Times"""
    try:
        response = requests.get(ET_NEWS_URL, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # Try multiple selectors for Economic Times
        story_selectors = [
            ".eachStory",
            ".story-box",
            ".story",
            "article"
        ]
        
        stories = []
        for selector in story_selectors:
            stories = soup.select(selector)
            if stories:
                break
        
        for story in stories[:10]:
            title_tag = story.select_one("h3 a") or story.select_one("h2 a") or story.select_one("a")
            desc_tag = story.select_one("p")
            time_tag = story.select_one("time") or story.select_one(".date")
            
            if title_tag:
                title = title_tag.get_text(strip=True)
                relative_url = title_tag.get('href')
                if relative_url:
                    full_link = urljoin(ET_BASE_URL, relative_url)
                    description = desc_tag.get_text(strip=True) if desc_tag else None
                    published_at = time_tag.get_text(strip=True) if time_tag else None
                    
                    # Extract stocks with sentiment
                    stocks_with_sentiment = extract_stock_names_with_sentiment(title + " " + (description or ""))
                    
                    news_items.append({
                        "title": title,
                        "link": full_link,
                        "description": description,
                        "publishedAt": published_at,
                        "source": "Economic Times",
                        "stocks": extract_stock_names(title + " " + (description or "")),  # Legacy format
                        "stocks_with_sentiment": stocks_with_sentiment
                    })
        
        return news_items
    except Exception as e:
        print(f"Error fetching Economic Times news: {e}")
        return []

@app.route("/api/news/moneycontrol")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_moneycontrol_only():
    """Get news from MoneyControl only"""
    return get_moneycontrol_news()

@app.route("/api/news/economic-times")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_et_news():
    """Get news from Economic Times only"""
    return get_economictimes_news()

@app.route("/api/article")
@rate_limit(max_requests=30, window=60)
def get_article_content():
    """Get full article content with summary and sentiment analysis"""
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400
    
    try:
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch article"}), 500
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Determine source and extract accordingly
        if "economictimes.indiatimes.com" in url:
            return extract_et_article_content(soup, url)
        elif "moneycontrol.com" in url:
            return extract_mc_article_content(soup, url)
        else:
            return jsonify({"error": "Unsupported source"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_et_article_content(soup, url):
    """Extract article content from Economic Times with summary and sentiment"""
    try:
        # Extract main article content
        content_selectors = [
            ".artText p",
            ".Normal p",
            ".story-content p",
            "article p",
            ".content p"
        ]
        
        content = ""
        for selector in content_selectors:
            paragraphs = soup.select(selector)
            if paragraphs:
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                break
        
        # Generate summary
        summary = simple_summarize(content, max_sentences=3)
        
        # Extract author
        author = None
        author_selectors = [
            ".byline",
            ".author",
            ".writer",
            "[class*='author']"
        ]
        
        for selector in author_selectors:
            author_tag = soup.select_one(selector)
            if author_tag:
                author = author_tag.get_text(strip=True).replace("By", "").strip()
                break
        
        # Extract publish date
        published_at = None
        date_selectors = [
            ".publish_on",
            ".date",
            "time",
            "[class*='date']"
        ]
        
        for selector in date_selectors:
            date_tag = soup.select_one(selector)
            if date_tag:
                published_at = date_tag.get_text(strip=True)
                break
        
        # Extract stock names from content with sentiment
        all_text = content or soup.get_text()
        stocks_with_sentiment = extract_stock_names_with_sentiment(all_text)
       
        # Overall article sentiment
        article_sentiment = analyze_sentiment(content)
        
        return jsonify({
            "content": content,
            "summary": summary,
            "author": author,
            "publishedAt": published_at,
            "stocks": extract_stock_names(all_text),  # Legacy format
            "stocks_with_sentiment": stocks_with_sentiment,
            "article_sentiment": {
                "score": round(article_sentiment, 3),
                "label": "positive" if article_sentiment > 0.1 else "negative" if article_sentiment < -0.1 else "neutral"
            },
            "source": "MoneyControl",
            "url": url
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to extract MC article: {str(e)}"}), 500
    
def extract_mc_article_content(soup, url):
    """Extract article content from MoneyControl with summary, author, publish date, and sentiment"""
    try:
        # Extract main article content from MoneyControl
        content_selectors = [
            ".article_body p",
            ".content_wrapper p",
            ".clearfix p",
            "article p",
            ".content p"
        ]
        
        content = ""
        for selector in content_selectors:
            paragraphs = soup.select(selector)
            if paragraphs:
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                break
        
        # Fallback if content still empty
        if not content:
            content = soup.get_text(strip=True)

        # Generate summary from content
        summary = simple_summarize(content, max_sentences=3)

        # Extract author
        author = None
        author_selectors = [
            ".article_author",
            ".author",
            ".byline",
            "[class*='author']"
        ]
        for selector in author_selectors:
            tag = soup.select_one(selector)
            if tag:
                author = tag.get_text(strip=True).replace("By", "").strip()
                break

        # Extract publish date
        published_at = None
        date_selectors = [
            ".article_schedule",
            ".schedule",
            "time",
            "[class*='date']"
        ]
        for selector in date_selectors:
            tag = soup.select_one(selector)
            if tag:
                published_at = tag.get_text(strip=True)
                break

        # Extract stock mentions + sentiment
        all_text = content or soup.get_text()
        stocks_with_sentiment = extract_stock_names_with_sentiment(all_text)
        
        # Analyze sentiment
        article_sentiment = analyze_sentiment(content)

        return jsonify({
            "content": content,
            "summary": summary,
            "author": author,
            "publishedAt": published_at,
            "stocks": extract_stock_names(all_text),
            "stocks_with_sentiment": stocks_with_sentiment,
            "article_sentiment": {
                "score": round(article_sentiment, 3),
                "label": "positive" if article_sentiment > 0.1 else "negative" if article_sentiment < -0.1 else "neutral"
            },
            "source": "MoneyControl",
            "url": url
        })

    except Exception as e:
        return jsonify({"error": f"Failed to extract MC article: {str(e)}"}), 500


def get_mc_article_metadata(url):
    """Get MoneyControl article metadata"""
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None, None, None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract publish date
        published_at = None
        date_span = soup.find("div", class_="article_schedule")
        if date_span:
            span_element = date_span.find("span")
            if span_element:
                published_at = span_element.get_text(strip=True)
        
        # Extract author
        author = None
        author_tag = soup.find("div", class_="article_author")
        if author_tag:
            author = author_tag.get_text(strip=True).replace("By", "").strip()
        
        # Extract description/summary
        description = None
        desc_selectors = [
            ".content_wrapper p",
            ".article-content p",
            ".story-content p"
        ]
        
        for selector in desc_selectors:
            paragraphs = soup.select(selector)
            if paragraphs and len(paragraphs) > 0:
                # Get first paragraph as description
                first_para = paragraphs[0].get_text(strip=True)
                if len(first_para) > 50:  # Only use if substantial
                    description = first_para[:200] + "..." if len(first_para) > 200 else first_para
                    break
        
        return published_at, author, description
    
    except Exception as e:
        print(f"Error fetching MC metadata: {e}")
        return None, None, None

@app.route("/api/news/summary")
@rate_limit(max_requests=30, window=60)
def get_news_with_summary():
    """Get news with enhanced summaries and sentiment analysis"""
    try:
        # Get news from both sources
        mc_news = get_moneycontrol_news()
        et_news = get_economictimes_news()
        
        # Combine news
        all_news = mc_news + et_news
        
        # Enhanced news with better summaries
        enhanced_news = []
        for news_item in all_news[:15]:  # Limit to 15 articles
            enhanced_item = news_item.copy()
            
            # Generate better summary if description exists
            if news_item.get('description'):
                enhanced_item['summary'] = simple_summarize(news_item['description'], max_sentences=2)
            else:
                enhanced_item['summary'] = "Summary not available"
            
            # Add overall sentiment
            full_text = (news_item.get('title', '') + " " + news_item.get('description', ''))
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
            
            enhanced_news.append(enhanced_item)
        
        return jsonify({
            "news": enhanced_news,
            "total_count": len(enhanced_news),
            "sources": ["MoneyControl", "Economic Times"],
            "last_updated": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/by-stock")
@rate_limit(max_requests=30, window=60)
def get_news_by_stock():
    """Get news filtered by specific stock"""
    stock_symbol = request.args.get('stock', '').upper()
    
    if not stock_symbol:
        return jsonify({"error": "Missing 'stock' parameter"}), 400
    
    try:
        # Get all news
        all_news = get_news()
        
        # Filter news by stock
        filtered_news = []
        for news_item in all_news:
            # Check if stock is mentioned in stocks_with_sentiment
            if 'stocks_with_sentiment' in news_item:
                for stock in news_item['stocks_with_sentiment']:
                    if stock['symbol'].upper() == stock_symbol:
                        filtered_news.append(news_item)
                        break
        
        return jsonify({
            "news": filtered_news,
            "stock_symbol": stock_symbol,
            "count": len(filtered_news),
            "last_updated": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sentiment/analysis")
@rate_limit(max_requests=20, window=60)
def analyze_market_sentiment():
    """Analyze overall market sentiment from recent news"""
    try:
        # Get recent news
        all_news = get_news()
        
        if not all_news:
            return jsonify({"error": "No news available"}), 404
        
        # Analyze sentiment
        sentiment_scores = []
        stock_sentiments = {}
        
        for news_item in all_news:
            # Overall article sentiment
            full_text = (news_item.get('title', '') + " " + news_item.get('description', ''))
            sentiment_score = analyze_sentiment(full_text)
            sentiment_scores.append(sentiment_score)
            
            # Stock-specific sentiments
            if 'stocks_with_sentiment' in news_item:
                for stock in news_item['stocks_with_sentiment']:
                    symbol = stock['symbol']
                    if symbol not in stock_sentiments:
                        stock_sentiments[symbol] = []
                    stock_sentiments[symbol].append(stock['sentiment'])
        
        # Calculate overall market sentiment
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Calculate average sentiment for each stock
        stock_avg_sentiments = {}
        for symbol, sentiments in stock_sentiments.items():
            stock_avg_sentiments[symbol] = {
                'average_sentiment': round(sum(sentiments) / len(sentiments), 3),
                'sentiment_count': len(sentiments),
                'label': 'positive' if sum(sentiments) / len(sentiments) > 0.1 else 'negative' if sum(sentiments) / len(sentiments) < -0.1 else 'neutral'
            }
        
        return jsonify({
            "overall_market_sentiment": {
                "score": round(overall_sentiment, 3),
                "label": "positive" if overall_sentiment > 0.1 else "negative" if overall_sentiment < -0.1 else "neutral"
            },
            "stock_sentiments": stock_avg_sentiments,
            "analysis_date": datetime.now().isoformat(),
            "news_analyzed": len(all_news)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stocks/trending")
@rate_limit(max_requests=30, window=60)
def get_trending_stocks():
    """Get trending stocks based on news mentions and sentiment"""
    try:
        # Get recent news
        all_news = get_news()
        
        if not all_news:
            return jsonify({"error": "No news available"}), 404
        
        # Count stock mentions and calculate sentiment
        stock_data = {}
        
        for news_item in all_news:
            if 'stocks_with_sentiment' in news_item:
                for stock in news_item['stocks_with_sentiment']:
                    symbol = stock['symbol']
                    if symbol not in stock_data:
                        stock_data[symbol] = {
                            'name': stock['name'],
                            'symbol': symbol,
                            'mention_count': 0,
                            'sentiments': [],
                            'news_items': []
                        }
                    
                    stock_data[symbol]['mention_count'] += 1
                    stock_data[symbol]['sentiments'].append(stock['sentiment'])
                    stock_data[symbol]['news_items'].append({
                        'title': news_item.get('title', ''),
                        'source': news_item.get('source', ''),
                        'sentiment': stock['sentiment_label']
                    })
        
        # Calculate trending score and average sentiment
        trending_stocks = []
        for symbol, data in stock_data.items():
            avg_sentiment = sum(data['sentiments']) / len(data['sentiments']) if data['sentiments'] else 0
            trending_score = data['mention_count'] * (1 + abs(avg_sentiment))  # More mentions + stronger sentiment = higher trending
            
            trending_stocks.append({
                'name': data['name'],
                'symbol': symbol,
                'mention_count': data['mention_count'],
                'average_sentiment': round(avg_sentiment, 3),
                'sentiment_label': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'trending_score': round(trending_score, 2),
                'recent_news': data['news_items'][:3]  # Latest 3 news items
            })
        
        # Sort by trending score
        trending_stocks.sort(key=lambda x: x['trending_score'], reverse=True)
        
        return jsonify({
            "trending_stocks": trending_stocks[:10],  # Top 10 trending
            "analysis_date": datetime.now().isoformat(),
            "total_stocks_analyzed": len(trending_stocks)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def update_all_indices():
    """Fetch fresh data for Nifty, Dow Jones, and Hang Seng"""
    global live_index_data

    try:
        logger.info("Updating indices data from MoneyControl...")
        new_data = fetch_indices_from_moneycontrol()
        with data_lock:
            live_index_data = new_data
        logger.info("Indices updated successfully")
    except Exception as e:
        logger.error(f"Error updating indices: {e}")




    
scheduler = BackgroundScheduler()

def start_background_tasks():
    """Start background tasks for live data updates"""
    try:
        # Initial data fetch
        update_all_indices()
        
        # Schedule regular updates every 60 seconds
        scheduler.add_job(
            func=update_all_indices,
            trigger="interval",
            minutes=60,
            id='update_indices',
            name='Update live indices data',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("Background scheduler started for live data updates")
    
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")

# Start background tasks
start_background_tasks()

if __name__ == "__main__":
    
    
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if scheduler.running:
            scheduler.shutdown()
        logger.info("Scheduler stopped")