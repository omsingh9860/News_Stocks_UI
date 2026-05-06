"""
All configuration constants for the Market Insights Dashboard.
"""

BASE_URL = "https://www.moneycontrol.com"
NEWS_URL = f"{BASE_URL}/news/business/markets"

ET_BASE_URL = "https://economictimes.indiatimes.com"
ET_NEWS_URL = f"{ET_BASE_URL}/markets/stocks/news"

TRADINGVIEW_URL = "https://in.tradingview.com/markets/stocks-india/ideas/"

CACHE_DURATION = 300           # 5 minutes
LIVE_DATA_CACHE_DURATION = 60  # 1 minute

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
    {'name': "Dr. Reddy's Laboratories", 'symbol': 'DRREDDY', 'aliases': ['Dr Reddy', 'DRREDDY']},
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
    'optimistic', 'confident', 'upgrade', 'target', 'buy', 'overweight', 'recommend',
]

NEGATIVE_KEYWORDS = [
    'loss', 'decline', 'fall', 'decrease', 'drop', 'crash', 'plunge', 'down', 'lower', 'negative',
    'bearish', 'sell-off', 'weak', 'underperform', 'miss', 'below', 'concern', 'worry', 'problem',
    'issue', 'challenge', 'difficulty', 'crisis', 'risk', 'threat', 'volatile', 'uncertainty',
    'downgrade', 'sell', 'underweight', 'caution', 'warning', 'alert', 'disappointing',
]

# Extended stock lists for Phase 1 SaaS endpoints
US_STOCKS = [
    {'name': 'S&P 500 Index', 'symbol': 'SPX', 'exchange': 'US'},
    {'name': 'Dow Jones Industrial Average', 'symbol': 'DJI', 'exchange': 'US'},
    {'name': 'NASDAQ Composite', 'symbol': 'IXIC', 'exchange': 'US'},
    {'name': 'Apple Inc.', 'symbol': 'AAPL', 'exchange': 'NASDAQ'},
    {'name': 'Microsoft Corporation', 'symbol': 'MSFT', 'exchange': 'NASDAQ'},
    {'name': 'Alphabet Inc.', 'symbol': 'GOOGL', 'exchange': 'NASDAQ'},
    {'name': 'Amazon.com Inc.', 'symbol': 'AMZN', 'exchange': 'NASDAQ'},
    {'name': 'Tesla Inc.', 'symbol': 'TSLA', 'exchange': 'NASDAQ'},
    {'name': 'NVIDIA Corporation', 'symbol': 'NVDA', 'exchange': 'NASDAQ'},
    {'name': 'Meta Platforms Inc.', 'symbol': 'META', 'exchange': 'NASDAQ'},
]

NSE_STOCKS_LIST = [
    {'name': s['name'], 'symbol': s['symbol'], 'exchange': 'NSE'}
    for s in INDIAN_STOCKS
    if s['symbol'] not in ('NIFTY', 'SENSEX')
] + [
    {'name': 'Nifty 50 Index', 'symbol': 'NIFTY50', 'exchange': 'NSE'},
    {'name': 'Bank Nifty Index', 'symbol': 'BANKNIFTY', 'exchange': 'NSE'},
    {'name': 'BSE Sensex Index', 'symbol': 'SENSEX', 'exchange': 'BSE'},
]
