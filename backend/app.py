from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import pytz
from lxml import html
from urllib.parse import urljoin
import json
import os
import yfinance as yf
import sqlite3
import logging
from functools import lru_cache
from functools import wraps
import time
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import heapq
import sys
import io
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

# Database initialization
def init_db():
    conn = sqlite3.connect('bookmarks.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT UNIQUE,
            title TEXT,
            url TEXT,
            source TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

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

def get_tradingview_ideas():
    """Get TradingView ideas for Indian stocks"""
    try:
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the stock ideas and conditions
        all_ideas = soup.find_all("a", class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl")
        all_conditions = soup.find_all("span", class_="visually-hidden-label-cbI7LT3N")

        ideas_list = []

        for i in range(min(len(all_ideas), len(all_conditions))):
            try:
                idea_tag = all_ideas[i]
                condition_tag = all_conditions[i]

                idea_href = idea_tag.get("href", "")
                full_link = urljoin("https://in.tradingview.com", idea_href)
                stock_split = idea_href.split("/")
                stock_symbol = stock_split[4] if len(stock_split) > 4 else "Unknown"
                title = idea_tag.get_text(strip=True) or stock_symbol

                condition_text = condition_tag.get_text(strip=True)

                # ✅ Proper signal label & color mapping
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
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_available": True,
        "sources": ["MoneyControl", "Economic Times", "TradingView"],
        "features": ["summarization", "sentiment_analysis", "stock_extraction", "tradingview_ideas"]
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

def get_enhanced_tradingview_ideas():
    """Get TradingView ideas with enhanced signal labeling"""
    try:
        # Set up UTF-8 encoding
        # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all the stock ideas
        all_ideas = soup.find_all(class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl")
        
        # Find all the conditions (long, short, educational, etc.)
        all_conditions = soup.find_all(class_="visually-hidden-label-cbI7LT3N", name="span")
        
        # Initialize lists
        ideas_list = []
        
        # Extract stock names and links
        for idea in all_ideas:
            try:
                idea_href = idea.get("href")
                if idea_href:
                    stock_split = idea_href.split("/")
                    if len(stock_split) >= 5:
                        stock_symbol = stock_split[4]
                        full_link = "" + idea_href
                        title = idea.get_text(strip=True) or stock_symbol
                        
                        ideas_list.append({
                            'stock_symbol': stock_symbol,
                            'title': title,
                            'link': full_link,
                            'condition': 'Educational'  # Default condition
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
                idea['condition'] = conditions[i]
            
            # Add signal labeling based on condition
            condition = idea['condition']
            if condition == "Long":
                idea['signal_label'] = "BUY"
                idea['signal_color'] = "green"
            elif condition == "Short":
                idea['signal_label'] = "SELL"
                idea['signal_color'] = "red"
            else:  # Educational
                idea['signal_label'] = "EDUCATIONAL"
                idea['signal_color'] = "blue"
            
            enhanced_ideas.append(idea)
        
        return enhanced_ideas[:30]  # Return top 30 ideas
    
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
    
@app.route("/api/bookmarks", methods=['GET'])
@rate_limit(max_requests=30, window=60)
def get_bookmarks():
    """Get all bookmarked articles"""
    try:
        conn = sqlite3.connect('bookmarks.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT article_id, title, url, source, sentiment, created_at 
            FROM bookmarks 
            ORDER BY created_at DESC
        ''')
        bookmarks = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        bookmark_list = []
        for bookmark in bookmarks:
            bookmark_list.append({
                'article_id': bookmark[0],
                'title': bookmark[1],
                'url': bookmark[2],
                'source': bookmark[3],
                'sentiment': bookmark[4],
                'created_at': bookmark[5]
            })
        
        return jsonify({
            "bookmarks": bookmark_list,
            "total_count": len(bookmark_list),
            "last_updated": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bookmarks", methods=['POST'])
@rate_limit(max_requests=30, window=60)
def add_bookmark():
    """Add a new bookmark"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['article_id', 'title', 'url', 'source']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract sentiment if provided
        sentiment = data.get('sentiment', 'neutral')
        
        conn = sqlite3.connect('bookmarks.db')
        cursor = conn.cursor()
        
        # Check if bookmark already exists
        cursor.execute('SELECT id FROM bookmarks WHERE article_id = ?', (data['article_id'],))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return jsonify({"error": "Article already bookmarked"}), 409
        
        # Insert new bookmark
        cursor.execute('''
            INSERT INTO bookmarks (article_id, title, url, source, sentiment)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['article_id'], data['title'], data['url'], data['source'], sentiment))
        
        conn.commit()
        bookmark_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            "message": "Bookmark added successfully",
            "bookmark_id": bookmark_id,
            "article_id": data['article_id']
        }), 201
    
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/bookmarks/<article_id>", methods=['DELETE'])
@rate_limit(max_requests=30, window=60)
def delete_bookmark(article_id):
    """Delete a bookmark by article_id"""
    try:
        conn = sqlite3.connect('bookmarks.db')
        cursor = conn.cursor()
        
        # Check if bookmark exists
        cursor.execute('SELECT id FROM bookmarks WHERE article_id = ?', (article_id,))
        existing = cursor.fetchone()
        
        if not existing:
            conn.close()
            return jsonify({"error": "Bookmark not found"}), 404
        
        # Delete the bookmark
        cursor.execute('DELETE FROM bookmarks WHERE article_id = ?', (article_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Bookmark deleted successfully",
            "article_id": article_id
        })
    
    except Exception as e:
        logger.error(f"Error deleting bookmark: {e}")
        return jsonify({"error": str(e)}), 500



def get_stock_symbol_for_yahoo(symbol):
    """Convert Indian stock symbol to Yahoo Finance format"""
    special_cases = {
        'M&M': 'M&M.NS',
        'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
        'NIFTY': '^NSEI',
        'SENSEX': '^BSESN'
    }
    
    if symbol in special_cases:
        return special_cases[symbol]
    
    # For regular Indian stocks, append .NS (National Stock Exchange)
    return f"{symbol}.NS"

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period, timestamp):
    """Cached stock data fetching"""
    try:
        yahoo_symbol = get_stock_symbol_for_yahoo(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        
        # Get historical data
        hist = ticker.history(period=period)
        
        if hist.empty:
            return None
        
        # Convert to list of dictionaries for JSON serialization
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'timestamp': int(date.timestamp()),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        
        # Get current price info
        info = ticker.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        previous_close = info.get('previousClose', info.get('regularMarketPreviousClose', 0))
        
        # Calculate change
        change = current_price - previous_close if current_price and previous_close else 0
        change_percent = (change / previous_close * 100) if previous_close else 0
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'previous_close': round(previous_close, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'historical_data': data,
            'last_updated': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def get_stock_data(symbol, period='1mo'):
    """Get stock data with 5-minute caching"""
    try:
        # Create timestamp for 5-minute cache invalidation
        current_time = datetime.now()
        cache_timestamp = current_time.replace(second=0, microsecond=0)
        # Round to nearest 5 minutes
        cache_timestamp = cache_timestamp.replace(minute=(cache_timestamp.minute // 5) * 5)
        
        # Use cached function
        return fetch_stock_data_cached(symbol, period, cache_timestamp.timestamp())
    
    except Exception as e:
        logger.error(f"Error in get_stock_data for {symbol}: {e}")
        return None

@app.route("/api/chart/<symbol>")
@rate_limit(max_requests=60, window=60)
def get_chart_data(symbol):
    """Get chart data for a specific stock (default 1 month period)"""
    try:
        stock_data = get_stock_data(symbol.upper(), '1mo')
        
        if not stock_data:
            return jsonify({"error": f"No data found for symbol: {symbol}"}), 404
        
        return jsonify({
            "chart_data": stock_data,
            "symbol": symbol.upper(),
            "period": "1mo",
            "data_points": len(stock_data.get('historical_data', [])),
            "last_updated": stock_data.get('last_updated')
        })
    
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/<symbol>/<period>")
@rate_limit(max_requests=60, window=60)
def get_chart_data_with_period(symbol, period):
    """Get chart data for a specific stock with custom period"""
    try:
        # Validate period
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            return jsonify({
                "error": f"Invalid period: {period}. Valid periods are: {', '.join(valid_periods)}"
            }), 400
        
        stock_data = get_stock_data(symbol.upper(), period)
        
        if not stock_data:
            return jsonify({"error": f"No data found for symbol: {symbol}"}), 404
        
        # Calculate additional metrics based on period
        historical_data = stock_data.get('historical_data', [])
        analytics = {}
        
        if len(historical_data) >= 2:
            # Calculate period high/low
            highs = [data['high'] for data in historical_data]
            lows = [data['low'] for data in historical_data]
            
            analytics = {
                'period_high': max(highs) if highs else 0,
                'period_low': min(lows) if lows else 0,
                'period_volume_avg': sum(data['volume'] for data in historical_data) / len(historical_data),
                'volatility': round(
                    sum(abs(historical_data[i]['close'] - historical_data[i-1]['close']) 
                        for i in range(1, len(historical_data))) / len(historical_data), 2
                )
            }
        
        return jsonify({
            "chart_data": stock_data,
            "symbol": symbol.upper(),
            "period": period,
            "data_points": len(historical_data),
            "analytics": analytics,
            "last_updated": stock_data.get('last_updated')
        })
    
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol} with period {period}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)