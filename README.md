# 📈 Market Insights Dashboard

A production-ready, full-stack SaaS starter for real-time NSE & US stock news, personalized watchlists, trending analytics, sentiment analysis, and interactive candlestick charts.  
**No account required** — all personalization is stored locally in the browser.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 **Market Overview** | Live NSE/BSE/NASDAQ indices, auto-refreshed every 60 s |
| ⭐ **Watchlist** | Add/remove 40+ NSE stocks & 10 US stocks/indices; persisted in `localStorage` |
| 💼 **Holdings** | Add/edit/remove real portfolio holdings (symbol, qty, avg buy price, buy date, notes, manual current price); persisted in `localStorage` |
| 📰 **Personalized News** | Auto-filtered news feed for your watchlist **and holdings** stocks |
| 🔥 **Trending & Analytics** | Top-10 most-mentioned stocks by news volume + sentiment |
| 📊 **Analytics Dashboard** | Real portfolio P&L from your holdings (total invested, current value, unrealized P&L, top gainers/losers, allocation chart); falls back gracefully when live price is unavailable with manual override |
| 🕯️ **Candlestick Charts** | Interactive TradingView charts — click any stock or index to open |
| 🔖 **Bookmarks** | Bookmark articles; stored in `localStorage` |
| 🕐 **Recently Viewed** | Tracks stocks & articles you've viewed (local, no login) |
| 💬 **Feedback Form** | Star rating + category + message; saved locally |
| 🎓 **Onboarding Tour** | 6-step first-visit walkthrough |
| 🌙 **Dark / Light Mode** | Toggle with one click; preference persisted |
| 🦴 **Skeleton Loaders** | Animated placeholders while data loads |
| 🔄 **News Polling** | `useNewsPolling` hook ready for background refresh (configurable interval) |
| 📡 **Google Analytics** | `gtag` event tracking for watchlist, charts, tab switches |
| 🧠 **Sentiment Analysis** | TextBlob + keyword scoring on every news article |
| 💡 **TradingView Ideas** | BUY/SELL/EDUCATIONAL signals scraped from TradingView |

---

## 🗂 Project Structure

```
News_Stocks_UI/
├── backend/
│   ├── app/                         # Modular Flask package (new)
│   │   ├── __init__.py              # App factory (create_app)
│   │   ├── config.py                # All constants: URLs, stock lists, keywords
│   │   ├── api/
│   │   │   ├── indices.py           # /api/indices/* routes + APScheduler
│   │   │   ├── news.py              # /api/news/*, /api/news-with-ideas routes
│   │   │   ├── stocks.py            # /api/stocks/* routes
│   │   │   └── misc.py              # /api/health, /api/feedback, auth stubs
│   │   ├── scrapers/
│   │   │   ├── moneycontrol.py      # MoneyControl news + indices scraper
│   │   │   ├── economictimes.py     # Economic Times scraper
│   │   │   └── tradingview.py       # TradingView ideas scraper
│   │   └── utils/
│   │       ├── cache.py             # In-memory cache + rate limiting decorators
│   │       └── sentiment.py         # TextBlob/NLTK sentiment + summarization
│   ├── run.py                       # New modular entry point
│   ├── app.py                       # Original monolithic file (kept for compatibility)
│   └── requirements.txt
│
└── frontend/
    ├── public/
    │   └── index.html               # Google Analytics snippet placeholder
    └── src/
        ├── App.js                   # Main app with tab navigation
        ├── App.css                  # Full styles (dark + light themes)
        ├── index.js                 # Entry point — wraps App in ThemeProvider
        ├── services/
        │   └── api.js               # Centralized API calls (fetchIndices, fetchNewsWithIdeas…)
        ├── utils/
        │   └── helpers.js           # formatPrice, getSentimentColor, getSentimentIcon, etc.
        ├── context/
        │   └── ThemeContext.js      # Dark/light theme React context
        ├── hooks/
        │   ├── useWatchlist.js      # Watchlist state → localStorage
        │   ├── useHoldings.js       # Holdings state → localStorage
        │   ├── useRecentlyViewed.js # Recently-viewed → localStorage
        │   ├── useBookmarks.js      # News bookmarks → localStorage
        │   └── useNewsPolling.js    # Polling hook for background news refresh
        └── components/
            ├── Watchlist.js         # Watchlist UI (add/remove/search stocks)
            ├── HoldingsManager.js   # Holdings CRUD form (add/edit/remove holdings with P&L)
            ├── PersonalizedNews.js  # News filtered to watchlist + holdings stocks
            ├── TrendingSection.js   # Trending + recently viewed
            ├── AnalyticsDashboard.js# Real portfolio P&L from holdings + Recharts charts
            ├── StockChart.js        # TradingView candlestick modal
            ├── SkeletonLoader.js    # Animated skeleton placeholders
            ├── ThemeToggle.js       # Sun/Moon theme toggle button
            ├── FeedbackForm.js      # Feedback modal (star rating + message)
            └── OnboardingModal.js   # First-visit guided tour
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+, Flask, Flask-CORS, BeautifulSoup4, lxml |
| **NLP / Sentiment** | TextBlob, NLTK (punkt, stopwords, wordnet) |
| **Scheduling** | APScheduler (live index refresh every 60 s) |
| **Frontend** | React 19, Lucide React, Recharts, CSS3 |
| **Charts** | TradingView Widget (free embed — no API key needed) |
| **Analytics** | Google Analytics 4 (`gtag.js`) |
| **Data Sources** | MoneyControl, Economic Times (web scraping), TradingView |
| **Local Storage** | Watchlist, bookmarks, recently-viewed, feedback, theme, onboarding |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+

### Backend (Modular — recommended)

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Download NLTK data on first run (automatic, or run manually):
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python run.py                 # runs on http://localhost:5000
```

### Backend (Legacy — backward-compatible)

```bash
cd backend
python app.py                 # same as above, original monolithic entry point
```

### Frontend

```bash
cd frontend
npm install
# Configure the API URL:
echo "REACT_APP_API_URL=http://localhost:5000" > .env
npm start                     # runs on http://localhost:3000
```

---

## 🔌 API Endpoints

### Indices
| Method | Path | Description |
|---|---|---|
| GET | `/api/indices/live` | Live NSE/BSE/US indices |
| GET | `/api/indices/comparison` | Indices ranked by % change |
| GET | `/api/indices/summary` | Overall market summary |

### News
| Method | Path | Description |
|---|---|---|
| GET | `/api/news/summary` | News articles with AI summaries + sentiment |
| GET | `/api/news-with-ideas` | Combined news + TradingView ideas |
| GET | `/api/news/moneycontrol` | MoneyControl-only articles |
| GET | `/api/news/economic-times` | Economic Times-only articles |
| GET | `/api/news/for-tickers` | News filtered by tickers (`?tickers=TCS,INFY`) |
| GET | `/api/article` | Fetch full article content (`?url=…`) |

### Stocks
| Method | Path | Description |
|---|---|---|
| GET | `/api/stocks` | List all stocks (`?exchange=NSE`) |
| GET | `/api/stocks/trending` | Trending stocks by news mention count |
| GET | `/api/sentiment/analysis` | Overall market sentiment summary |

### TradingView Ideas
| Method | Path | Description |
|---|---|---|
| GET | `/api/tradingview/ideas` | Raw TradingView ideas |
| GET | `/api/tradingview/ideas/enhanced` | Ideas with BUY/SELL/EDUCATIONAL signal labels |
| GET | `/api/tradingview/ideas/by-condition` | Filter by condition (`?condition=Long`) |

### Misc
| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check & feature flags |
| POST | `/api/feedback` | Submit user feedback |
| POST | `/api/auth/register` | **[STUB]** — Phase 2 |
| POST | `/api/auth/login` | **[STUB]** — Phase 2 |

---

## 💾 Local Storage Keys

All personalization is stored in the browser — no login required.

| Key | Contents |
|---|---|
| `market_watchlist_v1` | User's watchlist (symbol, name, exchange) |
| `portfolio_holdings_v1` | User's holdings (symbol, qty, avgBuyPrice, buyDate, notes, manualCurrentPrice) |
| `recently_viewed_v1` | Last 10 viewed stocks/articles |
| `news_bookmarks_v1` | Bookmarked news articles (max 100) |
| `user_feedback_v1` | Submitted feedback entries (max 50) |
| `onboarding_completed_v1` | Flag — `true` once onboarding is seen |
| `app_theme_v1` | `"dark"` or `"light"` |

---

## 📊 Google Analytics Setup

In `frontend/public/index.html`, replace `GA_MEASUREMENT_ID` with your real [GA4 Measurement ID](https://support.google.com/analytics/answer/9304153):

```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

Tracked events: `tab_switch`, `watchlist_add`, `watchlist_remove`, `chart_view`, `news_view`.

---

## 🚀 Deployment

### Backend (Gunicorn)

```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
# or using the legacy entry point:
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Static Build)

```bash
cd frontend
npm run build
# Deploy the build/ folder to Vercel, Netlify, or any static host
```

---

## 🗺 Roadmap

| Phase | Status | Features |
|---|---|---|
| **Phase 1** | ✅ Complete | Watchlist, Personalized News, Trending, Candlestick Charts, Analytics Dashboard, Bookmarks, Dark/Light Mode, Skeleton Loaders, Feedback, Onboarding |
| **Phase 1.5** | ✅ Complete | Real Holdings input (add/edit/remove), localStorage persistence, real portfolio P&L metrics (total invested, current value, unrealized P&L, top gainers/losers, allocation chart), manual price override, holdings-aware Personalized News |
| **Phase 2** | 🔜 Planned | User Authentication (Google/Email), Cloud-synced Watchlists, Push Notifications |
| **Phase 3** | 🔜 Planned | Subscription Plans, Live Price Integration, AI Price Alerts, Team Workspaces |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'feat: add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Made with ❤️, React & Flask**
