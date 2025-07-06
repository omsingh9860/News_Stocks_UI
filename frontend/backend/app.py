from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import pytz
from lxml import html
import json
import os
from functools import wraps
import time

app = Flask(__name__)
CORS(app)

# Configuration
BASE_URL = "https://www.moneycontrol.com"
NEWS_URL = f"{BASE_URL}/news/business/markets"
CACHE_DURATION = 300  # 5 minutes

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

@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_available": True
    })

@app.route("/api/news")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_news():
    try:
        response = requests.get(NEWS_URL, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return {"error": "Failed to retrieve news", "status_code": response.status_code}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # Select the top 15 news links from the main page
        for a_tag in soup.select("h2 a")[:15]:
            title = a_tag.get('title') or a_tag.text.strip()
            href = a_tag.get('href')
            full_link = urljoin(BASE_URL, href)
            
            published_at, author, description = get_article_metadata(full_link)
            
            news_items.append({
                'title': title,
                'link': full_link,
                'publishedAt': published_at,
                'author': author,
                'description': description,
                'source': 'MoneyControl'
            })
        
        return news_items
    except Exception as e:
        return {"error": str(e)}

@app.route("/api/news/search")
@rate_limit(max_requests=30, window=60)
def search_news():
    """Search news by keyword"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    try:
        # Search in cached news first
        cache_key = "api:get_news:latest"
        current_time = time.time()
        
        if cache_key in cache:
            cached_time, cached_news = cache[cache_key]
            if current_time - cached_time < CACHE_DURATION:
                filtered_news = [
                    item for item in cached_news 
                    if query.lower() in item['title'].lower()
                ]
                return jsonify(filtered_news)
        
        # If no cache, fetch fresh data
        response = requests.get(NEWS_URL, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve news"}), 500
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        for a_tag in soup.select("h2 a"):
            title = a_tag.get('title') or a_tag.text.strip()
            if query.lower() in title.lower():
                href = a_tag.get('href')
                full_link = urljoin(BASE_URL, href)
                
                published_at, author, description = get_article_metadata(full_link)
                
                news_items.append({
                    'title': title,
                    'link': full_link,
                    'publishedAt': published_at,
                    'author': author,
                    'description': description,
                    'source': 'MoneyControl'
                })
        
        return jsonify(news_items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/market-summary")
@cache_response(timeout=600)  # 10 minutes cache
def get_market_summary():
    """Get market summary and indices"""
    try:
        # Mock market data - in real implementation, fetch from APIs
        market_data = {
            "indices": {
                "nifty50": {
                    "name": "NIFTY 50",
                    "value": 24855.85,
                    "change": 125.40,
                    "changePercent": 0.51,
                    "trend": "up"
                },
                "sensex": {
                    "name": "SENSEX",
                    "value": 81332.72,
                    "change": -45.22,
                    "changePercent": -0.06,
                    "trend": "down"
                }
            },
            "sector_performance": {
                "banking": {"change": 0.8, "trend": "up"},
                "technology": {"change": -0.3, "trend": "down"},
                "pharma": {"change": 1.2, "trend": "up"},
                "auto": {"change": 0.5, "trend": "up"}
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return market_data
    except Exception as e:
        return {"error": str(e)}

def get_article_metadata(article_url):
    try:
        res = requests.get(article_url, timeout=10)
        res.encoding = 'utf-8'
        
        if res.status_code != 200:
            return None, None, None
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Extract author
        author_div = soup.find("div", class_="article_author")
        author = author_div.get_text(strip=True).replace("By", "").strip() if author_div else None
        
        # Extract description
        description = None
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            description = meta_desc.get("content", "").strip()
        else:
            # Try to get first paragraph
            first_p = soup.find("p")
            if first_p:
                description = first_p.get_text(strip=True)[:200] + "..."
        
        published_at = None
        
        # Method 1: Try to find date in article_schedule span
        date_span = soup.find("div", class_="article_schedule")
        if date_span:
            span_element = date_span.find("span")
            if span_element:
                date_text = span_element.get_text(strip=True)
                published_at = parse_date_string(date_text)
        
        # Method 2: If not found, try alternative selectors
        if not published_at:
            # Try meta tags
            meta_date = soup.find("meta", {"property": "article:published_time"})
            if meta_date:
                published_at = meta_date.get("content")
            else:
                # Try other common date selectors
                date_selectors = [
                    ".article_schedule span",
                    ".article_date",
                    ".publish_date",
                    ".date",
                    "[class*='date']",
                    "[class*='time']"
                ]
                
                for selector in date_selectors:
                    date_element = soup.select_one(selector)
                    if date_element:
                        date_text = date_element.get_text(strip=True)
                        if date_text:
                            published_at = parse_date_string(date_text)
                            break
        
        # Method 3: If still not found, try XPath with lxml
        if not published_at:
            try:
                tree = html.fromstring(res.content)
                
                # Try multiple XPath patterns
                xpath_patterns = [
                    "//div[@class='article_schedule']//span/text()",
                    "//div[contains(@class, 'article_schedule')]//text()",
                    "//span[contains(@class, 'date')]//text()",
                    "//div[contains(@class, 'date')]//text()",
                    "//*[contains(text(), 'Published')]//text()",
                    "//*[contains(text(), 'IST')]//text()"
                ]
                
                for pattern in xpath_patterns:
                    results = tree.xpath(pattern)
                    for result in results:
                        if result and result.strip():
                            published_at = parse_date_string(result.strip())
                            if published_at:
                                break
                    if published_at:
                        break
            except Exception as e:
                print(f"XPath extraction failed: {e}")
        
        return published_at, author, description
    
    except Exception as e:
        print(f"⚠️ Error parsing article {article_url}: {e}")
        return None, None, None

def parse_date_string(date_text):
    """Parse various date formats commonly used by MoneyControl"""
    if not date_text:
        return None
    
    # Clean up the text
    date_text = date_text.strip()
    
    # Handle "Published: July 05, 2025 / 02:30 PM IST" format
    if "Published:" in date_text:
        published_str = date_text.split("Published:")[-1].strip()
        if "/" in published_str:
            date_part, time_part = published_str.split("/", 1)
            date_part = date_part.strip()
            time_part = time_part.strip().replace("IST", "").strip()
            combined = f"{date_part} {time_part}"
            
            try:
                dt = datetime.strptime(combined, "%B %d, %Y %I:%M %p")
                ist = pytz.timezone("Asia/Kolkata")
                dt_ist = ist.localize(dt)
                return dt_ist.isoformat()
            except ValueError:
                pass
    
    # Handle direct date formats
    date_formats = [
        "%B %d, %Y %I:%M %p",  # July 05, 2025 02:30 PM
        "%b %d, %Y %I:%M %p",  # Jul 05, 2025 02:30 PM
        "%Y-%m-%d %H:%M:%S",   # 2025-07-05 14:30:00
        "%d/%m/%Y %H:%M",      # 05/07/2025 14:30
        "%d-%m-%Y %H:%M",      # 05-07-2025 14:30
        "%B %d, %Y",           # July 05, 2025
        "%b %d, %Y",           # Jul 05, 2025
        "%Y-%m-%d",            # 2025-07-05
        "%d/%m/%Y",            # 05/07/2025
        "%d-%m-%Y"             # 05-07-2025
    ]
    
    # Remove IST, GMT, UTC etc.
    cleaned_date = date_text.replace("IST", "").replace("GMT", "").replace("UTC", "").strip()
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(cleaned_date, fmt)
            ist = pytz.timezone("Asia/Kolkata")
            dt_ist = ist.localize(dt)
            return dt_ist.isoformat()
        except ValueError:
            continue
    
    # If no format matches, return the original text
    return date_text

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)