# 📊 Market Insights Dashboard — Phase 1 SaaS Features

A modern, mobile-friendly web application for real-time NSE & US stock news, personalized watchlists, trending analytics, and interactive candlestick charts. **No account required** — all user data is saved locally in the browser.

---

## 🚀 What's New in Phase 1

| Feature | Description |
|---|---|
| ⭐ **Watchlist** | Add/remove NSE & US stocks; persisted in `localStorage` |
| 📰 **Personalized News** | News feed filtered to your watchlist stocks |
| 🔥 **Trending & Analytics** | Top-10 most-mentioned stocks by news volume + sentiment |
| 🕯️ **Candlestick Charts** | Interactive TradingView charts — click any stock/index to open |
| 🕐 **Recently Viewed** | Tracks stocks & articles you've viewed (local, no login) |
| 💬 **Feedback Form** | Rate + comment; saved locally & logged server-side |
| 🎓 **Onboarding Tour** | First-visit walkthrough of all features |
| 📊 **Google Analytics** | Tracks watchlist adds, chart opens, tab switches (opt-in) |
| 🌐 **US Stocks** | SPX, DJI, NASDAQ, AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META |

---

## ✨ All Features

### Core (existing)
- Real-time web-scraped news from **MoneyControl** & **Economic Times**
- Live NSE/BSE indices (Nifty 50, Sensex, Bank Nifty)
- AI sentiment analysis on news headlines
- TradingView ideas (BUY/SELL/EDUCATIONAL signals)
- Search & sentiment filters for news

### Phase 1 (new)
- **4-tab navigation**: Market Overview · Watchlist · Personalized News · Trending & Analytics
- **Watchlist**: 40+ NSE stocks + 10 US stocks/indices — fully searchable
- **Personalized News Feed**: Auto-filtered to your watchlist symbols
- **Trending Section**: Real-time mention count & average sentiment per stock
- **Stock Charts**: TradingView Advanced Chart (candlestick) opens as a modal
- **Feedback Form**: Star rating + category + message — stored in `localStorage` + console log
- **Onboarding Modal**: 6-step walkthrough shown once to new users
- **Responsive Design**: Fully mobile-friendly on all screen sizes
- **GA Integration**: `window.gtag` events for feature usage tracking

---

## 🗂 Project Structure

```
News_Stocks_UI/
├── backend/
│   ├── app.py            # Flask API (scraping + Phase 1 endpoints)
│   └── requirements.txt
└── frontend/
    ├── public/
    │   └── index.html    # Updated title + Google Analytics snippet
    └── src/
        ├── App.js        # Main app with tab navigation
        ├── App.css       # Full styles (dark theme + Phase 1 additions)
        ├── hooks/
        │   ├── useWatchlist.js      # Watchlist state → localStorage
        │   └── useRecentlyViewed.js # Recently-viewed → localStorage
        └── components/
            ├── Watchlist.js         # Watchlist UI
            ├── PersonalizedNews.js  # Filtered news by watchlist
            ├── TrendingSection.js   # Trending + recently viewed
            ├── StockChart.js        # TradingView candlestick chart
            ├── FeedbackForm.js      # Feedback form modal
            └── OnboardingModal.js   # First-visit tour
```

---

## 🛠 Tech Stack

| Layer | Stack |
|---|---|
| **Backend** | Python, Flask, BeautifulSoup4, TextBlob, NLTK, APScheduler |
| **Frontend** | React 19, Lucide React, CSS3 (dark theme) |
| **Charts** | TradingView Widget (free embed — no API key needed) |
| **Analytics** | Google Analytics 4 (`gtag.js`) |
| **Data** | MoneyControl & Economic Times (web scraping) |
| **Local Storage** | Watchlist, recently-viewed, feedback, onboarding flag |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend runs at `http://localhost:5000`

### Frontend Setup

```bash
cd frontend
npm install
# Set your backend URL in .env:
echo "REACT_APP_API_URL=http://localhost:5000" > .env
npm start
```

Frontend runs at `http://localhost:3000`

---

## 🔌 API Endpoints

### Existing Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/indices/live` | Live NSE/BSE indices |
| GET | `/api/news/summary` | News with summaries + sentiment |
| GET | `/api/news-with-ideas` | News + TradingView ideas (combined) |
| GET | `/api/tradingview/ideas/enhanced` | TradingView ideas with signals |
| GET | `/api/stocks/trending` | Trending stocks by news mentions |
| GET | `/api/sentiment/analysis` | Overall market sentiment |

### New Phase 1 Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/api/stocks` | List all stocks (NSE + US). Filter: `?exchange=NSE` |
| GET | `/api/news/for-tickers` | News filtered by tickers. e.g. `?tickers=TCS,INFY,AAPL` |
| GET | `/api/trending` | Trending stocks (mention count + sentiment) |
| POST | `/api/feedback` | Submit feedback (logs to console; DB in Phase 2) |
| POST | `/api/auth/register` | **[STUB]** Returns 501 — Phase 2 |
| POST | `/api/auth/login` | **[STUB]** Returns 501 — Phase 2 |

---

## 📊 Google Analytics Setup

In `frontend/public/index.html`, replace `GA_MEASUREMENT_ID` with your real [Google Analytics 4 Measurement ID](https://support.google.com/analytics/answer/9304153):

```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX', { send_page_view: true });
</script>
```

Tracked events: `tab_switch`, `watchlist_add`, `watchlist_remove`, `chart_view`, `news_view`.

---

## 💾 Local Data Storage

All user personalization is stored in the browser's `localStorage` — **no account or login required**.

| Key | Contents |
|---|---|
| `market_watchlist_v1` | User's watchlist (stock symbol, name, exchange) |
| `recently_viewed_v1` | Last 10 viewed stocks/articles |
| `user_feedback_v1` | Submitted feedback entries (max 50) |
| `onboarding_completed_v1` | Flag — `true` once onboarding is seen |

> **Note**: In Phase 2, authenticated accounts will sync watchlists and preferences to the cloud.

---

## 🚀 Deployment

### Backend (Gunicorn)
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Static Build)
```bash
cd frontend
npm run build
# Deploy the `build/` folder to Vercel, Netlify, or any static host
```

---

## 🗺 Roadmap

| Phase | Status | Features |
|---|---|---|
| Phase 1 | ✅ Complete | Watchlist, Personalized News, Trending, Charts, Feedback |
| Phase 2 | 🔜 Planned | User Authentication, Cloud-synced Watchlists, Notifications |
| Phase 3 | 🔜 Planned | Subscription Plans, Portfolio Tracking, AI Alerts |

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
 